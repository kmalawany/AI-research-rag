from embeddings.data_embedding import pgvector
import os


def retrieve_topic(state):
    vector_store = pgvector(connection=os.getenv("CONNECTION_NAME"),
                            collection_name=os.getenv("ABSTRACT_COLLECTION"))
    v = state["question"]
    retrieved_docs = vector_store.max_marginal_relevance_search(state["question"], k=10)
    return {'context': retrieved_docs}




