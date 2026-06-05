"""Smoke test helpers for verifying retriever traces in MLflow."""

from time import sleep

from mlflow.entities import SpanType
import mlflow
from mlflow.models.model import ModelInfo


def run_smoke_test(
    loaded_model: mlflow.pyfunc.PyFuncModel,
    model_info: ModelInfo,
    verbose: bool = False,
) -> bool:
    """Verify that the logged model produces retriever spans in MLflow traces.

    Args:
        loaded_model: The loaded MLflow pyfunc model to invoke.
        model_info: Metadata for the logged model under test.
        verbose: Whether to print lightweight smoke-test progress details.

    Returns:
        ``True`` when retriever spans are present and retrieval scorers can run.

    Raises:
        RuntimeError: If no traces are available after invoking the model.
    """

    smoke_test_query = "2. Who is the latest current US president as of what date?"

    _ = loaded_model.predict([{"query": smoke_test_query}])

    sleep(1)

    recent_traces = mlflow.search_traces(
        model_id=model_info.model_id,
        max_results=5,
        order_by=["timestamp_ms DESC"],
        return_type="list",
        include_spans=True,
    )

    if not recent_traces:
        raise RuntimeError(
            "No traces were found for this model. "
            "Verify tracking/autologging before using retrieval scorers."
        )

    latest_trace = recent_traces[0]
    retriever_spans = latest_trace.search_spans(span_type=SpanType.RETRIEVER)
    use_retrieval_scorers = len(retriever_spans) > 0

    if verbose:
        print("Smoke test trace id:", latest_trace.info.trace_id)
        print("Retriever span count:", len(retriever_spans))

    return use_retrieval_scorers
