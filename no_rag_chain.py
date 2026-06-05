"""Logged MLflow wrapper for the no-RAG QA baseline model."""

import mlflow

from chain_factory import build_no_rag_model
from logged_model_config import BASE_LLM

model = build_no_rag_model(base_llm=BASE_LLM)

mlflow.models.set_model(model)
