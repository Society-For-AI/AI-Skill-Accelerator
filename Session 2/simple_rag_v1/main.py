from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from .rag_groq import get_rag_answer

load_dotenv(".env")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str

@app.get("/")
def root():
    return {"message": "RAG API using FastAPI + Chroma + LangChain"}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        answer = get_rag_answer(request.query)
        return QueryResponse(answer=answer, query=request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))