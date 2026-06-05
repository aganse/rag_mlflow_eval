"""Utility helpers for fetching source documents and building the FAISS store."""

from collections.abc import Sequence

from bs4 import BeautifulSoup
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

### only necessary for MacOS:
faiss.omp_set_num_threads(1)


def fetch_federal_document(url: str, div_class: str) -> str:
    """Fetch the transcript text for a U.S. Milestones document page.

    Args:
        url: The Archives.gov page to scrape.
        div_class: The HTML class containing the transcript content.

    Returns:
        The extracted transcript text, or an error message when retrieval fails.
    """

    response = requests.get(url, timeout=60)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        transcript_section = soup.find("div", class_=div_class)
        if transcript_section:
            return transcript_section.get_text(separator="\n", strip=True)
        return "Transcript section not found."

    return f"Failed to retrieve the webpage. Status code: {response.status_code}"


def fetch_documents(url_list: Sequence[str]) -> list[Document]:
    """Fetch source pages and wrap each one in a LangChain document.

    Args:
        url_list: The source URLs to retrieve.

    Returns:
        A list of documents with source metadata preserved.
    """

    docs = []
    for url in url_list:
        text = fetch_federal_document(url, "col-sm-9")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": url},
            )
        )
    return docs


def create_faiss_database(
    url_list: Sequence[str],
    database_save_directory: str,
    chunk_size: int,
    chunk_overlap: int,
    verbose: bool = False,
) -> FAISS:
    """Create and persist a FAISS vector store from source URLs.

    Args:
        url_list: The source URLs to ingest.
        database_save_directory: The local directory where the FAISS index is
            saved.
        chunk_size: The maximum chunk size used during splitting.
        chunk_overlap: The overlap between adjacent chunks.
        verbose: Whether to print lightweight progress information.

    Returns:
        The populated FAISS vector store.
    """

    raw_documents = fetch_documents(url_list)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    document_chunks = splitter.split_documents(raw_documents)

    if verbose:
        print(
            "Built FAISS source chunks:",
            f"raw_documents={len(raw_documents)}",
            f"chunks={len(document_chunks)}",
            f"chunk_size={chunk_size}",
            f"chunk_overlap={chunk_overlap}",
        )

    embedding_generator = OpenAIEmbeddings()
    faiss_database = FAISS.from_documents(document_chunks, embedding_generator)
    faiss_database.save_local(database_save_directory)

    return faiss_database


def print_formatted_response(
    response_list: Sequence[str],
    max_line_length: int = 80,
) -> None:
    """Print response strings with a fixed maximum line length.

    Args:
        response_list: The response strings to print.
        max_line_length: The maximum number of characters per output line.
    """

    for response in response_list:
        words = response.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= max_line_length:
                line += word + " "
            else:
                print(line)
                line = word + " "
        print(line)
