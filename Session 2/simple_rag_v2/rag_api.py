from fastapi import FastAPI, Query
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

app = FastAPI()

# Load model and vector store
embedding_fn = OllamaEmbeddings(model="mistral")
vectordb = Chroma(persist_directory="db", embedding_function=embedding_fn)
retriever = vectordb.as_retriever()

llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.get("/ask")
async def ask(query: str = Query(...)):
    answer = qa_chain.run(query)
    return {"query": query, "answer": answer}


# run with: uvicorn rag_api:app --reload