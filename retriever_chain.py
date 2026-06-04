"""LangChain retrieval pipeline used for the logged MLflow model."""

from pathlib import Path
from typing import Any

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import mlflow

candidate_dirs = [
    Path("faiss_index"),
    Path(__file__).resolve().parent / "faiss_index",
    Path(__file__).resolve().parent / "code" / "faiss_index",
]

persist_dir = next((p for p in candidate_dirs if (p / "index.faiss").exists()), None)
if persist_dir is None:
    raise FileNotFoundError(
        "Could not find faiss_index. Checked: "
        + ", ".join(str(p) for p in candidate_dirs)
    )

vector_db = FAISS.load_local(
    persist_dir.as_posix(),
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)

retriever = vector_db.as_retriever(search_kwargs={"k": 4})  # default is 4, just making explicit

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}",
        ),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    ChatOpenAI(model="gpt-4.1-mini", temperature=0, timeout=60, max_retries=1),
    prompt,
)

chain = create_retrieval_chain(retriever, question_answer_chain)

def to_chain_input(payload: dict[str, str]) -> dict[str, str]:
    """Convert the pyfunc input payload into the chain's expected shape.

    Args:
        payload: A single prediction record containing a ``query`` field.

    Returns:
        A dictionary with the ``input`` field expected by the retrieval chain.
    """

    return {"input": payload["query"]}


def extract_answer(payload: dict[str, Any]) -> str:
    """Extract the final answer text from the retrieval chain output.

    Args:
        payload: The retrieval chain output dictionary.

    Returns:
        The generated answer string.
    """

    return str(payload["answer"])

model = RunnableLambda(to_chain_input) | chain | RunnableLambda(extract_answer)

mlflow.models.set_model(model)
