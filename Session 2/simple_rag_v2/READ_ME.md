# Step 0: Create a virtual codespace
python -m venv venvName
source venvName/bin/activate #activate you venv

# Step 1: install the necessay dependencies
pip install langchain chromadb fastapi streamlit pydantic python-dotenv

# Step 2: install ollama and pull model to your local
curl -fsSL https://ollama.com/install.sh | sh
ollama run mistral  # or any model you have pulled 
on windows: Go to the Ollama website:
👉 https://ollama.com/download

Confirm install: ollama --version

ollama run mistral or ollama run llama2



Ollama is a local tool to run large language models (like Mistral, LLaMA, Gemma, etc.) on your own machine, without needing OpenAI or other external APIs.

# Step 3: run ingestion
python ingest_docs.py

# Step 4: run fastapi
uvicorn rag_api:app --reload

# Step 5: run streamlit ui
streamlit run streamlit_ui.py