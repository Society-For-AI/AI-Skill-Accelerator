import streamlit as st
import requests

st.title("🔍 RAG Q&A App with Local LLM")

query = st.text_input("Ask a question:")

if query:
    response = requests.get("http://localhost:8000/ask", params={"query": query})
    st.markdown(f"**Answer:** {response.json()['answer']}")


# run with: streamlit run streamlit_ui.py