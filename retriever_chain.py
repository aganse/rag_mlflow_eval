"""Logged MLflow wrapper for the retrieval-augmented QA model."""

import mlflow

from chain_factory import build_rag_model
from logged_model_config import BASE_LLM, RETRIEVAL_BACKEND, RETRIEVAL_TOP_K

model = build_rag_model(
    retrieval_backend=RETRIEVAL_BACKEND,
    base_llm=BASE_LLM,
    retrieval_top_k=RETRIEVAL_TOP_K,
)

mlflow.models.set_model(model)
