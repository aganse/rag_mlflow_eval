### setting these env-vars is needed in MacOS on my MacBookPro:
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# and without these the default settings swamped system with lots of warnings on my MacbookPro:
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "3"
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "1"
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"  # re typeerror in smoke_test
###

import os
from pathlib import Path
import shutil
import tempfile

from mlflow.genai.datasets import create_dataset, get_dataset
import mlflow

from eval_dataset import get_dataset_name, get_url_listings, get_eval_records
import scorers
import smoke_test
import utils


params = {
  "experiment_name": "History docs RAG",
  "mlflow_tracking_uri": "http://localhost:5000",
  "debug": False,
}


params["dataset_name"] = get_dataset_name()

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."
mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
mlflow.set_experiment(params["experiment_name"])

mlflow.langchain.autolog()  # log_traces=True, log_models=True, log_model_signatures=True, log_input_examples=True)
mlflow.tracing.disable_notebook_display()


# Set up FAISS vector database storage
temporary_directory = tempfile.mkdtemp()
persist_dir = os.path.join(temporary_directory, "faiss_index")
url_listings = get_url_listings()
vector_db = utils.create_faiss_database(url_listings, persist_dir)


if params["debug"]:
    print("Debug: verifying retrieval is set up correctly...")
    test_question = "Which amendment abolished slavery?"
    print(f"searching docs with question: {test_question}")
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(test_question)
    print("Retrieved docs:", len(docs))
    for i, d in enumerate(docs, 1):
        print(f"\nDoc {i} source:", d.metadata.get("source"))
        print("Length:", len(d.page_content))
        print(d.page_content[:300])
    print("\n")


local_persist_dir = Path("faiss_index")
if local_persist_dir.exists():
    shutil.rmtree(local_persist_dir)
shutil.copytree(persist_dir, local_persist_dir)


with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        lc_model="retriever_chain.py",
        name="retrieval_qa",
        code_paths=["faiss_index"],
    )

logged_model = mlflow.get_logged_model(model_info.model_id)
print("model_id:", logged_model.model_id)
print("artifact_location:", logged_model.artifact_location)

client = mlflow.tracking.MlflowClient()
arts = client.list_logged_model_artifacts(model_info.model_id)
print(arts)

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

test_question = "What amendment addresses protections on the right to vote?"
answer1 = loaded_model.predict([{"query": test_question}])
utils.print_formatted_response(answer1)

if params["debug"]:
    use_retrieval_scorers = smoke_test.run_smoke_test(loaded_model, model_info)


# Create MLflow Evaluation Dataset
eval_records = get_eval_records()

experiment = mlflow.get_experiment_by_name(params["experiment_name"])
if experiment is None:
    raise RuntimeError(f"Could not find MLflow experiment {params['experiment_name']}")
try:
    dataset = get_dataset(name=params["dataset_name"])
    print("Using existing dataset:", dataset.name, dataset.dataset_id)
except Exception:
    dataset = create_dataset(
        name=params["dataset_name"],
        experiment_id=experiment.experiment_id,
        tags={
            "app": "langchain-retriever2",
            "domain": "legal-rag",
            "source": "hand-written",
        },
    )
    print("Created dataset:", dataset.name, dataset.dataset_id)

dataset = dataset.merge_records(eval_records)
print("Dataset ready:", dataset.name, dataset.dataset_id)
print("Added / merged", len(eval_records), "records")


results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=scorers.make_predict_fn(loaded_model),
    scorers=scorers.get_scorers(use_retrieval_scorers),
    model_id=model_info.model_id,  # optional, but useful to keep tied to the logged model
)

print(results)  # the performance metrics which are logged into mlflow as well


# Clean up our temporary directory that we created with our FAISS instance
# shutil.rmtree(temporary_directory)
