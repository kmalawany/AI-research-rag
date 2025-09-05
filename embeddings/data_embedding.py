from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings


def load_data(file_path):

    loader = CSVLoader(file_path=file_path, metadata_columns=['id', 'year', 'categories'], encoding="utf-8")
    data = loader.load()
    return data


def generate_embeddings_locally():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def pgvector(connection, collection_name):
    vector_store = PGVector(
        embeddings=generate_embeddings_locally(),
        collection_name=collection_name,
        connection=connection,

    )
    return vector_store


# docs = load_data('D://ArxivLLM//arxiv_ai_updated.csv')
# vector_store = pgvector(connection="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
#                         collection_name="arxiv_dataset")
# print(docs[0])
# batch_size = 500
# for i in tqdm(range(0, len(docs), batch_size)):
#     _ = vector_store.add_documents(documents=docs[i: i + batch_size])
