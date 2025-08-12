from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from data import vectorstore

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# Setup retriever from vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

def get_rag_answer(query: str) -> str:
    result = qa_chain.run(query)
    return result