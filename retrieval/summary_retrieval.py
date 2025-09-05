from embeddings.data_embedding import pgvector
import os


def summary_retrieve(state):
    vector_store = pgvector(connection=os.getenv("CONNECTION_NAME"),
                            collection_name=os.getenv("ABSTRACT_COLLECTION"))
    v = state["question"]
    retrieved_docs = vector_store.max_marginal_relevance_search(state["question"], k=5)
    return {'context': retrieved_docs}
