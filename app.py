import streamlit as st
import requests
import uuid

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="AI Research Papers RAG", layout="centered")

st.title("📄 AI Research Papers RAG System")
st.write("Ask questions about AI research papers.")

# Generate a unique thread_id per session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

query = st.text_input("Enter your research question:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                FASTAPI_URL,
                json={"question": query, "thread_id": st.session_state.thread_id}
            )
            if response.status_code == 200:
                answer = response.json()
                st.subheader("🔎 Answer")
                st.write(answer)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")