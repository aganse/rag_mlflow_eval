# Create prediction wrapper + scorer selection

from collections.abc import Callable


import mlflow.pyfunc
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    RetrievalRelevance,
    RetrievalGroundedness,
    RetrievalSufficiency,
)


def evaluate_logged_rag_model(query: str, loaded_model: mlflow.pyfunc.PyFuncModel) -> str:
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


def make_predict_fn(loaded_model: mlflow.pyfunc.PyFuncMode) -> Callable[[str], str]:
    """Factory function to generate a predict_fn (for use in
    mlflow.genai.evaluate) for a given loaded_model."""
    def predict_fn(query: str) -> str:
        return evaluate_logged_rag_model(query, loaded_model)
    return predict_fn


def get_scorers(use_retrieval_scorers: bool = False) -> list[Scorer]:
    scorers = [
        Correctness(),
        RelevanceToQuery(),
    ]

    if use_retrieval_scorers:
        scorers.extend([
            RetrievalRelevance(),
            RetrievalGroundedness(),
            RetrievalSufficiency(),
        ])

    print("Scorers selected:", [type(s).__name__ for s in scorers])
    return scorers
