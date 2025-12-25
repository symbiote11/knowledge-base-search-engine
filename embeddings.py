from langchain_community.vectorstores import FAISS
from app.config import embeddings

def create_faiss_index(docs, save_path: str):
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)
