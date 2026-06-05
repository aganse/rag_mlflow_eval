"""Utility helpers for fetching source documents and building the FAISS store."""

from collections.abc import Sequence
import re

from bs4 import BeautifulSoup
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

### only necessary for MacOS:
faiss.omp_set_num_threads(1)


def fetch_webpage_contents(url: str, div_class: str | None = None) -> str:
    """Fetch the main text content from a webpage.

    Args:
        url: The webpage URL to scrape.
        div_class: Optional HTML class to prefer when extracting content.
            This is kept for backward compatibility, but the function now
            falls back to a more general main-content extraction strategy.

    Returns:
        The extracted page text, or an error message when retrieval fails.
    """

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"Failed to retrieve the webpage: {exc}"

    soup = BeautifulSoup(response.text, "html.parser")

    for tag_name in [
        "script",
        "style",
        "noscript",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "svg",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    candidate_sections = []

    if div_class:
        preferred_section = soup.find(class_=div_class)
        if preferred_section:
            candidate_sections.append(preferred_section)

    for selector in [
        "main",
        "article",
        "[role='main']",
        "section",
        "div[id*='content']",
        "div[class*='content']",
        "div[id*='main']",
        "div[class*='main']",
    ]:
        candidate_sections.extend(soup.select(selector))

    if soup.body:
        candidate_sections.append(soup.body)
    else:
        candidate_sections.append(soup)

    best_section = max(
        candidate_sections,
        key=lambda tag: len(tag.get_text(" ", strip=True)),
    )
    text = best_section.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text or "No webpage text content found."


def fetch_documents(url_list: Sequence[str]) -> list[Document]:
    """Fetch source pages and wrap each one in a LangChain document.

    Args:
        url_list: The source URLs to retrieve.

    Returns:
        A list of documents with source metadata preserved.
    """

    docs = []
    for url in url_list:
        text = fetch_webpage_contents(url)
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
