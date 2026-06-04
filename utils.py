from bs4 import BeautifulSoup
import faiss
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

### only necessary for MacOS:
faiss.omp_set_num_threads(1)


def fetch_federal_document(url, div_class):
    """
    Scrapes the transcript of the US Milestones document from the given Archives.gov URL.

    Args:
    url (str): URL of the webpage to scrape.

    Returns:
    str: The transcript text of the document.
    """
    # Sending a request to the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Parsing the HTML content of the page
        soup = BeautifulSoup(response.text, "html.parser")

        # Finding the transcript section by its HTML structure
        if transcript_section := soup.find("div", class_=div_class):
            transcript_text = transcript_section.get_text(separator="\n", strip=True)
            return transcript_text
        else:
            return "Transcript section not found."
    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"


def fetch_documents(url_list):
    """
    Fetch documents from URLs and keep each source as its own LangChain Document.
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


def create_faiss_database(url_list, database_save_directory, chunk_size=500, chunk_overlap=50):
    """
    Creates and saves a FAISS database from separately chunked source documents.
    """
    raw_documents = fetch_documents(url_list)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    document_chunks = splitter.split_documents(raw_documents)

    print("Raw documents:", len(raw_documents))
    print("Chunks:", len(document_chunks))
    print("Max chunk length:", max(len(d.page_content) for d in document_chunks))
    print("First 3 chunk lengths:", [len(d.page_content) for d in document_chunks[:3]])

    embedding_generator = OpenAIEmbeddings()
    faiss_database = FAISS.from_documents(document_chunks, embedding_generator)
    faiss_database.save_local(database_save_directory)

    return faiss_database


def print_formatted_response(response_list, max_line_length=80):
    """
    Formats and prints responses with a maximum line length for better readability.

    Args:
    response_list (list): A list of strings representing responses.
    max_line_length (int): Maximum number of characters in a line. Defaults to 80.
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
