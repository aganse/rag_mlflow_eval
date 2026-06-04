# Smoke test: are retriever traces actually present?

from mlflow.entities import SpanType
import mlflow


def run_smoke_test(loaded_model, model_info):

    SMOKE_TEST_QUERY = "Which amendment involves birthright citizenship?"

    # Generate one fresh prediction / trace
    _ = loaded_model.predict([{"query": SMOKE_TEST_QUERY}])

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
            "If this happens, stop here and verify tracking/autologging before using retrieval scorers."
        )

    latest_trace = recent_traces[0]
    retriever_spans = latest_trace.search_spans(span_type=SpanType.RETRIEVER)

    print("Latest trace id:", latest_trace.info.trace_id)
    print("Retriever span count:", len(retriever_spans))

    for i, span in enumerate(retriever_spans[:2], start=1):
        docs = span.outputs or []
        print(f"\nRetriever span {i}: {len(docs)} docs")
        if docs:
            first_doc = docs[0]
            if isinstance(first_doc, dict):
                preview = first_doc.get("page_content", "")
            else:
                preview = getattr(first_doc, "page_content", "")
            print("First doc preview:", preview[:250])

    USE_RETRIEVAL_SCORERS = len(retriever_spans) > 0
    print("\nUSE_RETRIEVAL_SCORERS =", USE_RETRIEVAL_SCORERS)
    return USE_RETRIEVAL_SCORERS
