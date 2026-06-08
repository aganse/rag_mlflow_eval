"""Logged MLflow wrapper for the no-RAG QA baseline model."""

import mlflow

from chain_factory import build_no_rag_model
from logged_model_config import (
    ACTIVE_SYSTEM_PROMPT_TEMPLATE,
    ACTIVE_SYSTEM_PROMPT_URI,
    BASE_LLM,
)

model = build_no_rag_model(
    base_llm=BASE_LLM,
    system_prompt=ACTIVE_SYSTEM_PROMPT_TEMPLATE,
    system_prompt_uri=ACTIVE_SYSTEM_PROMPT_URI,
)

mlflow.models.set_model(model)
