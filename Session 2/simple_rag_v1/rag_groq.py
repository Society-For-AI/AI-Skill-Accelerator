from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from .data import vectorstore  # same vectorstore setup as before
import os


# 🔁 Use Groq's LLM instead of OpenAI
llm = ChatGroq(
    model_name="llama3-70b-8192",  # (or llama3-8b-8192), You can also try "mixtral-8x7b-32768" or "gemma-7b-it"
    groq_api_key=os.getenv("GROQ_API_KEY"),  # or set via env var GROQ_API_KEY
    temperature=0.2
)

# Use same retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# RAG Chain using Groq
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

def get_rag_answer(query: str) -> str:
    return qa_chain.run(query)