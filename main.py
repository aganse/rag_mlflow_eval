"""Run the end-to-end MLflow QA evaluation workflow."""

### setting these env-vars is needed in MacOS on my MacBookPro:
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# and without these the default settings swamped system with lots of warnings
# on my MacBookPro:
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "3"
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "1"
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
###

import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.genai.datasets import create_dataset, get_dataset

import datasets
from retrieval_backends import get_supported_retrieval_backends
import scorers
import smoke_test
import utils

params = {
    "experiment_name": "History docs RAG",
    "mlflow_tracking_uri": "http://localhost:5000",
    "mode": "rag",
    "verbose": False,
    "retrieval_backend": "faiss",
    "dataset": "testB",
    "base_llm": "gpt-4o-mini",
    "judge_llm": "",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retrieval_top_k": 4,
}

_VALID_MODES = ("rag", "no_rag")



def normalize_optional_string(value: Any, field_name: str) -> str | None:
    """Return a stripped string value, or ``None`` when blank or missing.

    Args:
        value: The raw config value to normalize.
        field_name: The config field name used in validation errors.

    Returns:
        The stripped string value, or ``None`` if the input is ``None`` or blank.

    Raises:
        ValueError: If the value is not a string-like optional field.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{field_name}' must be a string when provided.")

    normalized = value.strip()
    return normalized or None



def validate_positive_int(value: Any, field_name: str) -> None:
    """Validate that a config value is a positive integer.

    Args:
        value: The config value to validate.
        field_name: The config field name used in error messages.

    Raises:
        ValueError: If the value is not a positive integer.
    """

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{field_name}' must be a positive integer.")



def validate_non_negative_int(value: Any, field_name: str) -> None:
    """Validate that a config value is a non-negative integer.

    Args:
        value: The config value to validate.
        field_name: The config field name used in error messages.

    Raises:
        ValueError: If the value is not a non-negative integer.
    """

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"'{field_name}' must be a non-negative integer.")



def validate_params(config: dict[str, Any]) -> None:
    """Validate the user-configurable workflow parameters.

    Args:
        config: The parameter dictionary defined near the top of ``main.py``.

    Raises:
        ValueError: If any parameter value is unsupported.
    """

    mode = config["mode"]
    if mode not in _VALID_MODES:
        valid_modes = ", ".join(_VALID_MODES)
        raise ValueError(
            f"Unsupported mode '{mode}'. Choose from: {valid_modes}"
        )

    datasets.get_dataset_module(config["dataset"])

    base_llm = config["base_llm"]
    if not isinstance(base_llm, str) or not base_llm.strip():
        raise ValueError("'base_llm' must be a non-empty string.")

    judge_llm = normalize_optional_string(config.get("judge_llm"), "judge_llm")
    if judge_llm is not None and ":/" not in judge_llm:
        raise ValueError(
            "'judge_llm' must be empty/missing or use MLflow judge model "
            "format like 'openai:/gpt-4o-mini'."
        )

    validate_positive_int(config["chunk_size"], "chunk_size")
    validate_non_negative_int(config["chunk_overlap"], "chunk_overlap")
    if config["chunk_overlap"] >= config["chunk_size"]:
        raise ValueError("'chunk_overlap' must be smaller than 'chunk_size'.")

    validate_positive_int(config["retrieval_top_k"], "retrieval_top_k")

    retrieval_backend = config["retrieval_backend"]
    supported_backends = get_supported_retrieval_backends()
    if retrieval_backend not in supported_backends:
        available_backends = ", ".join(supported_backends)
        raise ValueError(
            "Unsupported retrieval backend "
            f"'{retrieval_backend}'. Choose from: {available_backends}"
        )



def print_status(message: str, verbose: bool) -> None:
    """Print a lightweight progress message when verbose mode is enabled.

    Args:
        message: The short status message to print.
        verbose: Whether verbose logging is enabled.
    """

    if verbose:
        print(message)



def build_model_name(config: dict[str, Any]) -> str:
    """Build the concise model name used for MLflow logging.

    Args:
        config: The validated runtime configuration.

    Returns:
        The model name derived from the high-level app variant.
    """

    if config["mode"] == "rag":
        return (
            f"qa_{config['mode']}_{config['retrieval_backend']}_"
            f"{config['dataset']}"
        )
    return f"qa_{config['mode']}_{config['dataset']}"



def write_logged_model_config(config: dict[str, Any]) -> None:
    """Write the runtime config module packaged with the logged model.

    Args:
        config: The validated runtime configuration.
    """

    config_lines = [
        '"""Runtime configuration packaged with the logged MLflow model."""',
        "",
        f"BASE_LLM = {config['base_llm']!r}",
        f"RETRIEVAL_BACKEND = {config['retrieval_backend']!r}",
        f"RETRIEVAL_TOP_K = {config['retrieval_top_k']!r}",
    ]
    Path("logged_model_config.py").write_text(
        "\n".join(config_lines) + "\n"
    )



def prepare_faiss_index(
    url_listings: list[str],
    chunk_size: int,
    chunk_overlap: int,
    verbose: bool,
) -> None:
    """Build the FAISS index locally so it can be packaged with the model.

    Args:
        url_listings: The source URLs used to populate the vector store.
        chunk_size: The maximum chunk size used during splitting.
        chunk_overlap: The overlap between adjacent chunks.
        verbose: Whether to print lightweight progress information.
    """

    temporary_directory = tempfile.mkdtemp()
    persist_dir = Path(temporary_directory) / "faiss_index"
    try:
        print_status("Building FAISS index from source documents...", verbose)
        utils.create_faiss_database(
            url_listings,
            persist_dir.as_posix(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=verbose,
        )

        local_persist_dir = Path("faiss_index")
        if local_persist_dir.exists():
            shutil.rmtree(local_persist_dir)
        shutil.copytree(persist_dir, local_persist_dir)
        print_status("Copied FAISS index into local model artifacts.", verbose)
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)



def prepare_retrieval_artifacts(
    config: dict[str, Any],
    url_listings: list[str],
) -> None:
    """Prepare backend-specific retrieval artifacts for a RAG run.

    Args:
        config: The validated runtime configuration.
        url_listings: The source URLs used to populate the vector store.

    Raises:
        ValueError: If the backend is not yet supported by this helper.
    """

    if config["retrieval_backend"] == "faiss":
        prepare_faiss_index(
            url_listings,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            verbose=config["verbose"],
        )
        return

    raise ValueError(
        "Retrieval artifact preparation is not implemented for backend "
        f"'{config['retrieval_backend']}'."
    )



def get_logged_model_script(mode: str) -> str:
    """Return the Python module file used for MLflow model logging.

    Args:
        mode: The configured app mode.

    Returns:
        The filename passed to ``mlflow.langchain.log_model``.
    """

    if mode == "rag":
        return "retriever_chain.py"
    return "no_rag_chain.py"



def get_model_code_paths(mode: str, retrieval_backend: str) -> list[str]:
    """Return the extra code paths packaged with the logged model.

    Args:
        mode: The configured app mode.
        retrieval_backend: The configured retrieval backend short name.

    Returns:
        The list of files/directories needed by the logged model.
    """

    code_paths = [
        "chain_factory.py",
        "logged_model_config.py",
        "retrieval_backends",
    ]
    if mode == "rag" and retrieval_backend == "faiss":
        code_paths.append("faiss_index")
    return code_paths



def log_workflow_params(
    config: dict[str, Any],
    dataset_name: str,
    model_name: str,
    evaluation_label: str | None = None,
) -> None:
    """Log workflow parameters to the active MLflow run.

    Args:
        config: The validated runtime configuration.
        dataset_name: The long-form dataset name used for MLflow datasets.
        model_name: The model name used for MLflow logging.
        evaluation_label: Optional label describing the evaluation purpose.
    """

    logged_params = {
        "experiment_name": str(config["experiment_name"]),
        "mlflow_tracking_uri": str(config["mlflow_tracking_uri"]),
        "mode": str(config["mode"]),
        "verbose": str(config["verbose"]),
        "retrieval_backend": str(config["retrieval_backend"]),
        "dataset": str(config["dataset"]),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "base_llm": str(config["base_llm"]),
        "judge_llm": normalize_optional_string(
            config.get("judge_llm"),
            "judge_llm",
        )
        or "",
        "chunk_size": str(config["chunk_size"]),
        "chunk_overlap": str(config["chunk_overlap"]),
        "retrieval_top_k": str(config["retrieval_top_k"]),
        "enable_retrieval_eval": str(config["mode"] == "rag"),
    }
    if evaluation_label is not None:
        logged_params["evaluation_label"] = evaluation_label
    mlflow.log_params(logged_params)



def main() -> None:
    """Run the configured MLflow QA evaluation workflow."""

    validate_params(params)
    judge_llm = normalize_optional_string(params.get("judge_llm"), "judge_llm")

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "Please set the OPENAI_API_KEY environment variable."
        )

    dataset_module = datasets.get_dataset_module(params["dataset"])
    dataset_name = dataset_module.get_dataset_name()
    url_listings = dataset_module.get_url_listings()
    eval_records = dataset_module.get_eval_records()
    model_name = build_model_name(params)
    evaluation_label = "baseline"
    evaluation_run_name = f"eval_{model_name}_{evaluation_label}"

    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    mlflow.set_experiment(params["experiment_name"])
    mlflow.langchain.autolog()
    mlflow.tracing.disable_notebook_display()

    print_status(
        (
            "Running workflow with "
            f"mode={params['mode']}, "
            f"dataset={params['dataset']}, "
            f"retrieval_backend={params['retrieval_backend']}, "
            f"base_llm={params['base_llm']}, "
            f"judge_llm={judge_llm or 'mlflow-default'}"
        ),
        params["verbose"],
    )

    if params["mode"] == "rag":
        prepare_retrieval_artifacts(params, url_listings)
    else:
        print_status(
            "No-RAG mode selected; skipping retrieval index preparation.",
            params["verbose"],
        )

    write_logged_model_config(params)

    logged_model_script = get_logged_model_script(params["mode"])
    model_code_paths = get_model_code_paths(
        params["mode"],
        params["retrieval_backend"],
    )

    with mlflow.start_run(run_name=f"log_{model_name}"):
        log_workflow_params(params, dataset_name, model_name)
        print_status(
            f"Logging model '{model_name}' from {logged_model_script}...",
            params["verbose"],
        )
        model_info = mlflow.langchain.log_model(
            lc_model=logged_model_script,
            name=model_name,
            code_paths=model_code_paths,
        )

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

        sample_question = (
            "1. What is your knowledge cutoff date in your training data?"
        )
        sample_answer = loaded_model.predict([{"query": sample_question}])
        if params["verbose"]:
            print("Sample prediction for:", sample_question)
            utils.print_formatted_response(sample_answer)

        use_retrieval_scorers = False
        if params["mode"] == "rag":
            print_status(
                "Running retrieval smoke test for scorer eligibility...",
                params["verbose"],
            )
            try:
                use_retrieval_scorers = smoke_test.run_smoke_test(
                    loaded_model,
                    model_info,
                    verbose=params["verbose"],
                )
            except RuntimeError as exc:
                print(
                    "Warning: retrieval scorers were skipped because the "
                    f"smoke test failed: {exc}"
                )

    experiment = mlflow.get_experiment_by_name(params["experiment_name"])
    if experiment is None:
        raise RuntimeError(
            "Could not find MLflow experiment "
            f"{params['experiment_name']}"
        )

    try:
        dataset = get_dataset(name=dataset_name)
        print_status(
            f"Using existing dataset: {dataset.name} {dataset.dataset_id}",
            params["verbose"],
        )
    except Exception:
        dataset = create_dataset(
            name=dataset_name,
            experiment_id=experiment.experiment_id,
            tags={
                "app": "qa",
                "domain": "legal-rag",
                "source": "hand-written",
                "dataset_short_name": params["dataset"],
            },
        )
        print_status(
            f"Created dataset: {dataset.name} {dataset.dataset_id}",
            params["verbose"],
        )

    dataset = dataset.merge_records(eval_records)
    print_status(
        (
            f"Dataset ready: {dataset.name} {dataset.dataset_id}; "
            f"merged_records={len(eval_records)}"
        ),
        params["verbose"],
    )

    with mlflow.start_run(run_name=evaluation_run_name):
        log_workflow_params(
            params,
            dataset_name,
            model_name,
            evaluation_label=evaluation_label,
        )
        evaluation_results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=scorers.make_predict_fn(loaded_model),
            scorers=scorers.get_scorers(
                use_retrieval_scorers,
                judge_llm=judge_llm,
                verbose=params["verbose"],
            ),
            model_id=model_info.model_id,
        )

    print(evaluation_results)


if __name__ == "__main__":
    main()
