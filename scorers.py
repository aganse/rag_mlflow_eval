"""Prediction helpers and scorer selection for MLflow GenAI evaluation."""

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
    """Run the logged model for a query and normalize the response to a string.

    Args:
        query: The user query to evaluate.
        loaded_model: The loaded MLflow pyfunc model.

    Returns:
        A plain-text answer suitable for judge-based scorers.
    """

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


def make_predict_fn(loaded_model: mlflow.pyfunc.PyFuncModel) -> Callable[[str], str]:
    """Create the prediction callback expected by ``mlflow.genai.evaluate``.

    Args:
        loaded_model: The loaded MLflow pyfunc model.

    Returns:
        A callable that maps a query string to a normalized answer string.
    """

    def predict_fn(query: str) -> str:
        """Evaluate a single query with the logged model.

        Args:
            query: The user query to evaluate.

        Returns:
            A normalized answer string from the logged model.
        """

        return evaluate_logged_rag_model(query, loaded_model)

    return predict_fn


def get_scorers(use_retrieval_scorers: bool = False) -> list[Scorer]:
    """Return the configured set of GenAI scorers.

    Args:
        use_retrieval_scorers: Whether to include retrieval-specific scorers.

    Returns:
        The ordered list of scorers to pass to MLflow evaluation.
    """

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
