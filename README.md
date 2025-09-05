# AI-research-rag
RAG-powered question answering system over AI research papers with topic search and paper summarization.

- **Backend:** FastAPI (serves the RAG graph)  
- **Frontend:** Streamlit (chat-style UI)  
- **Database:** pgvector (PostgreSQL extension for vector storage)

- Features
- Query classification (topic search / summarization / out-of-context)
- Semantic search over abstracts and full-text papers
- Summarization and explanation of research papers
- **Chunking strategy** for full papers to improve retrieval quality
- Chat history with interactive UI
- Dockerized pgvector database

-  System Diagram
![System Diagram](diagram.png)
