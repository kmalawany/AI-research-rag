import os
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_core.messages import AIMessage
from embeddings.data_embedding import pgvector, generate_embeddings_locally
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.documents import Document
import pymupdf
import os

@tool
def get_paper(paper_id: str):
    """
    Download and extract the text of a research paper from arXiv
    :param paper_id: paper id
    :return text: extracted text from pdf
    """
    url = 'https://arxiv.org/pdf/' + paper_id

    response = requests.get(url)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file_path = tmp_file.name

    load_pdf = PyPDFLoader(tmp_file_path)
    paper = load_pdf.load()
    os.remove(tmp_file_path)

    return {"paper_text": paper}


def generate_paper_embeddings(state):
    vector_store = pgvector(connection=os.getenv('CONNECTION_NAME'),
                            collection_name=os.getenv("PAPERS_COLLECTION"))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200,
                                                   add_start_index=True
                                                   )
    cleaned_docs = []
    for doc in state["paper_text"]:
        cleaned_text = doc.page_content.replace("\x00", "")
        cleaned_docs.append(Document(page_content=cleaned_text, metadata=doc.metadata))
    splits = text_splitter.split_documents(cleaned_docs)
    vector_store.add_documents(splits)
    retrieved_docs = vector_store.max_marginal_relevance_search(state['question'], k=5)
    return {'context': retrieved_docs}
