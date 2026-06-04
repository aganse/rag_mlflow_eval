from pathlib import Path

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

def to_chain_input(x):
    return {"input": x["query"]}

def extract_answer(x):
    return x["answer"]

model = RunnableLambda(to_chain_input) | chain | RunnableLambda(extract_answer)

mlflow.models.set_model(model)
