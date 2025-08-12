from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import os

# Setup embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # or bge-small-en

# Sample documents
texts = [
    "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "The capital of France is Paris, known for its art, fashion, and culture.",
    "Python is a programming language known for simplicity and readability.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Photosynthesis allows plants to convert sunlight into energy.",
    "The mitochondria is known as the powerhouse of the cell.",
    "The theory of evolution was proposed by Charles Darwin.",
    "The speed of sound is approximately 343 meters per second in air.",
    "The human brain contains about 86 billion neurons.",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water.",
    "The water cycle involves processes such as evaporation, condensation, precipitation, and collection.",
    "Newton's laws of motion describe the relationship between a body and the forces acting on it.",
    "Mitosis is a type of cell division that results in two daughter cells identical to the parent cell.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "The French Revolution began in 1789 and led to significant political and social change in France.",
    "World War I lasted from 1914 to 1918 and involved many of the world's major powers.",
    "The Roman Empire was one of the largest empires in history, centered around the Mediterranean Sea.",
    "The American Civil War was fought from 1861 to 1865 over issues including states' rights and slavery.",
    "The Great Depression was a severe worldwide economic depression that took place mostly during the 1930s.",
    "Mount Everest is the highest mountain in the world, located in the Himalayas.",
    "The Nile River is the longest river in the world, flowing through northeastern Africa.",
    "The Sahara is the largest hot desert in the world, located in North Africa.",
    "The Amazon Rainforest is the world's largest tropical rainforest and a vital part of the Earth's ecosystem.",
    "Antarctica is the coldest continent on Earth and is mostly covered by ice.",
    "Artificial Intelligence refers to the simulation of human intelligence in machines.",
    "Blockchain is a decentralized ledger used in cryptocurrencies like Bitcoin.",
    "Machine learning is a subset of AI that allows computers to learn from data.",
    "The Internet of Things (IoT) involves connecting physical devices to the internet.",
    "Cloud computing allows users to access data and applications over the internet.",
    "Shakespeare is widely regarded as one of the greatest writers in the English language.",
    "The human body has 206 bones in adulthood.",
    "The heart pumps blood throughout the body via the circulatory system.",
    "Languages like Mandarin Chinese, Spanish, and English are among the most spoken worldwide.",
    "Recycling helps reduce waste and conserve natural resources."
]


# Wrap text into LangChain Documents
docs = [Document(page_content=t) for t in texts]

# Create Chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="rag_collection",
    persist_directory="chroma_db"
)