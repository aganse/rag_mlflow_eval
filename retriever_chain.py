"""Logged MLflow wrapper for the retrieval-augmented QA model."""

import mlflow

from chain_factory import build_rag_model
from logged_model_config import (
    ACTIVE_SYSTEM_PROMPT_TEMPLATE,
    ACTIVE_SYSTEM_PROMPT_URI,
    BASE_LLM,
    EMBEDDING_MODEL,
    RETRIEVAL_BACKEND,
    RETRIEVAL_TOP_K,
)

model = build_rag_model(
    retrieval_backend=RETRIEVAL_BACKEND,
    base_llm=BASE_LLM,
    retrieval_top_k=RETRIEVAL_TOP_K,
    system_prompt=ACTIVE_SYSTEM_PROMPT_TEMPLATE,
    system_prompt_uri=ACTIVE_SYSTEM_PROMPT_URI,
    embedding_model=EMBEDDING_MODEL,
)

mlflow.models.set_model(model)
