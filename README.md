# AI-research-rag
RAG-powered question answering system over AI research papers with topic search and paper summarization.

- **Backend:** FastAPI (serves the RAG graph)  
- **Frontend:** Streamlit (chat-style UI)  
- **Database:** pgvector (PostgreSQL extension for vector storage)

----------------------------------------------------------------------
- ## Features
- Query classification (topic search / summarization / out-of-context)
- Semantic search over abstracts and full-text papers
- Summarization and explanation of research papers
- **Chunking strategy** for full papers to improve retrieval quality
- Chat history with interactive UI
- Dockerized pgvector database
----------------------------------------------------------------------
- ## System Diagram
![System Diagram](Figure_1.png)
----------------------------------------------------------------------
## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ai-research-rag.git
cd ai-research-rag
```
### 2. Create .env file
#### Create a .env file in the project root with the following variables:
Example:
CONNECTION_NAME=your_connection_string_here
ABSTRACT_COLLECTION=abstracts
PAPERS_COLLECTION=papers
