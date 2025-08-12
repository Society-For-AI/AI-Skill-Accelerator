# Step 0: Install dependencies
pip install langchain chromadb fastapi pydantic langchain_community langchain_groq sentence-transformers openai

# Step 1: Run fastapi server
uvicorn main:app --reload

# Step 2: load the swagger page on your browser
http://localhost:8000/docs

# Step 3: Send in a request
eg: {
  "query": "Who developed the theory of relativity?"
}