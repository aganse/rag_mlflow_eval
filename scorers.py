# Create prediction wrapper + scorer selection

from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    RetrievalRelevance,
    RetrievalGroundedness,
    RetrievalSufficiency,
)

def evaluate_logged_rag_model(query: str) -> str:
    pred = loaded_model.predict([{"query": query}])

    # Normalize the model output into a plain string for the judges
    if isinstance(pred, list) and len(pred) > 0:
        first = pred[0]
        if isinstance(first, dict):
            return (
                first.get("answer")
                or first.get("result")
                or first.get("response")
                or str(first)
            )
        return str(first)

    return str(pred)


def select_scorers(USE_RETRIEVAL_SCORERS=False):
    scorers = [
        Correctness(),
        RelevanceToQuery(),
    ]

    if USE_RETRIEVAL_SCORERS:
        scorers.extend([
            RetrievalRelevance(),
            RetrievalGroundedness(),
            RetrievalSufficiency(),
        ])

    print("Scorers selected:", [type(s).__name__ for s in scorers])
    return scorers
