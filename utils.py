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
    Scrapes the transcript of the Act Establishing Yellowstone National Park from the given URL.

    Args:
    url (str): URL of the webpage to scrape.

    Returns:
    str: The transcript text of the Act.
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


# def fetch_and_save_documents(url_list, doc_path):
#     """
#     Fetches documents from given URLs and saves them to a specified file path.

#     Args:
#         url_list (list): List of URLs to fetch documents from.
#         doc_path (str): Path to the file where documents will be saved.
#     """
#     for url in url_list:
#         document = fetch_federal_document(url, "col-sm-9")
#         with open(doc_path, "a") as file:
#             file.write(document)


# def create_faiss_database(document_path, database_save_directory, chunk_size=500, chunk_overlap=10):
#     """
#     Creates and saves a FAISS database using documents from the specified file.

#     Args:
#         document_path (str): Path to the file containing documents.
#         database_save_directory (str): Directory where the FAISS database will be saved.
#         chunk_size (int, optional): Size of each document chunk. Default is 500.
#         chunk_overlap (int, optional): Overlap between consecutive chunks. Default is 10.

#     Returns:
#         FAISS database instance.
#     """
#     # Load documents from the specified file
#     # document_loader = TextLoader(document_path)
#     # raw_documents = document_loader.load()
#     with open(document_path, "r", encoding="utf-8") as f:
#         text = f.read()
#     raw_documents = [
#         Document(
#             page_content=text,
#             metadata={"source": document_path},
#         )
#     ]

#     # Split documents into smaller chunks with specified size and overlap
#     document_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     document_chunks = document_splitter.split_documents(raw_documents)

#     # Generate embeddings for each document chunk
#     embedding_generator = OpenAIEmbeddings()
#     faiss_database = FAISS.from_documents(document_chunks, embedding_generator)

#     # Save the FAISS database to the specified directory
#     faiss_database.save_local(database_save_directory)

#     return faiss_database


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
