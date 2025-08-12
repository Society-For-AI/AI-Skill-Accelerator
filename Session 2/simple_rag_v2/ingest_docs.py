from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load document
loader = TextLoader("data/sample.txt")
docs = loader.load()

# Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Generate embeddings
embedding_fn = OllamaEmbeddings(model="mistral")
Chroma.from_documents(chunks, embedding=embedding_fn, persist_directory="db")
print("Ingested and stored in vector DB.")

# run with:  python ingest_docs.py