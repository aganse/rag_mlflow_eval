"""Retriever backend selection helpers."""

from typing import Any

from . import faiss_backend

SUPPORTED_RETRIEVAL_BACKENDS = ("faiss",)


def get_supported_retrieval_backends() -> tuple[str, ...]:
    """Return the currently supported retrieval backend names."""

    return SUPPORTED_RETRIEVAL_BACKENDS


def get_retriever(backend_name: str, retrieval_top_k: int) -> Any:
    """Build and return a retriever for the requested backend.

    Args:
        backend_name: The configured retrieval backend short name.
        retrieval_top_k: The number of chunks to retrieve per query.

    Returns:
        The backend-specific retriever object.

    Raises:
        ValueError: If the backend is not supported.
    """

    if backend_name == "faiss":
        return faiss_backend.get_retriever(retrieval_top_k)

    available_backends = ", ".join(get_supported_retrieval_backends())
    raise ValueError(
        f"Unsupported retrieval backend '{backend_name}'. "
        f"Choose from: {available_backends}"
    )
