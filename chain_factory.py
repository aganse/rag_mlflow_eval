"""Helpers for constructing the logged LangChain QA models."""

import logging
from typing import Any

import mlflow

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from retrieval_backends import get_retriever

_LOGGER = logging.getLogger(__name__)


def build_chat_model(base_llm: str) -> ChatOpenAI:
    """Return the shared chat model configuration used by both modes.

    Args:
        base_llm: The OpenAI chat model name used for answer generation.

    Returns:
        The configured LangChain chat model.
    """

    return ChatOpenAI(
        model=base_llm,
        temperature=0,
        timeout=60,
        max_retries=1,
    )


def to_chain_input(payload: dict[str, str]) -> dict[str, str]:
    """Convert a pyfunc prediction record into the chain input shape.

    Args:
        payload: A single prediction record containing a ``query`` field.

    Returns:
        A dictionary with the ``input`` field expected by the chain.
    """

    return {"input": payload["query"]}


def extract_answer(payload: Any) -> str:
    """Normalize a chain output payload into a plain answer string.

    Args:
        payload: The raw object returned by the runnable chain.

    Returns:
        The final answer text.
    """

    if isinstance(payload, dict):
        if "answer" in payload:
            return str(payload["answer"])
        if "content" in payload:
            return str(payload["content"])

    content = getattr(payload, "content", None)
    if content is not None:
        return str(content)

    return str(payload)


def link_prompt_to_active_trace(payload: dict[str, str], prompt_uri: str) -> dict[str, str]:
    """Best-effort prompt load to let MLflow link prompts to active traces.

    Args:
        payload: The normalized chain input payload.
        prompt_uri: The versioned MLflow prompt URI for the baked prompt.

    Returns:
        The unchanged payload so the runnable pipeline can continue.
    """

    try:
        # Load the exact versioned prompt during traced execution for MLflow's
        # prompt-linking side effects, while still using the baked template text
        # for the actual chain prompt so the packaged model remains reproducible.
        mlflow.genai.load_prompt(prompt_uri)
    except Exception:
        _LOGGER.warning(
            "Failed to load prompt '%s' for MLflow trace linking.",
            prompt_uri,
            exc_info=True,
        )
    return payload


def build_rag_model(
    retrieval_backend: str,
    base_llm: str,
    retrieval_top_k: int,
    system_prompt: str,
    system_prompt_uri: str,
    embedding_model: str | None = None,
) -> Any:
    """Build the retrieval-augmented QA model.

    Args:
        retrieval_backend: The retrieval backend short name.
        base_llm: The OpenAI chat model name used for answer generation.
        retrieval_top_k: The number of retrieved chunks to supply to the model.
        system_prompt: The resolved system prompt template used for RAG answers.
        system_prompt_uri: The versioned MLflow prompt URI used for trace linking.
        embedding_model: Optional embedding model override.

    Returns:
        The runnable LangChain model used for RAG predictions.
    """

    retriever = get_retriever(
        retrieval_backend,
        retrieval_top_k,
        embedding_model,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(
        build_chat_model(base_llm),
        prompt,
    )
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return (
        RunnableLambda(to_chain_input)
        | RunnableLambda(
            lambda payload: link_prompt_to_active_trace(payload, system_prompt_uri)
        )
        | chain
        | RunnableLambda(extract_answer)
    )


def build_no_rag_model(
    base_llm: str,
    system_prompt: str,
    system_prompt_uri: str,
) -> Any:
    """Build the no-RAG QA model used for the closed-book baseline.

    Args:
        base_llm: The OpenAI chat model name used for answer generation.
        system_prompt: The resolved system prompt template used for no-RAG answers.
        system_prompt_uri: The versioned MLflow prompt URI used for trace linking.

    Returns:
        The runnable LangChain model used for no-RAG predictions.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    chain = prompt | build_chat_model(base_llm) | RunnableLambda(extract_answer)
    return (
        RunnableLambda(to_chain_input)
        | RunnableLambda(
            lambda payload: link_prompt_to_active_trace(payload, system_prompt_uri)
        )
        | chain
    )
