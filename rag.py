from langchain.chains import RetrievalQA
from app.config import llm
from app.retriever import load_retriever

def get_rag_chain(index_path: str):
    retriever = load_retriever(index_path)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
