"""FAISS retriever helpers for the logged RAG model."""

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from utils import build_embeddings


def get_faiss_persist_dir() -> Path:
    """Return the packaged FAISS directory used by the logged model.

    Returns:
        The filesystem path containing ``index.faiss`` and ``index.pkl``.

    Raises:
        FileNotFoundError: If no packaged FAISS directory can be found.
    """

    candidate_dirs = (
        Path("faiss_index"),
        Path(__file__).resolve().parent.parent / "faiss_index",
        Path(__file__).resolve().parent.parent / "code" / "faiss_index",
    )

    persist_dir = next(
        (path for path in candidate_dirs if (path / "index.faiss").exists()),
        None,
    )
    if persist_dir is None:
        checked_paths = ", ".join(str(path) for path in candidate_dirs)
        raise FileNotFoundError(
            "Could not find faiss_index. Checked: " + checked_paths
        )

    return persist_dir


def get_retriever(
    retrieval_top_k: int,
    embedding_model: str | None = None,
) -> Any:
    """Load the packaged FAISS vector store and return a retriever.

    Args:
        retrieval_top_k: The number of chunks to retrieve per query.
        embedding_model: Optional embedding model override.

    Returns:
        A retriever configured with the requested top-k setting.
    """

    vector_db = FAISS.load_local(
        get_faiss_persist_dir().as_posix(),
        build_embeddings(embedding_model),
        allow_dangerous_deserialization=True,
    )
    return vector_db.as_retriever(search_kwargs={"k": retrieval_top_k})
