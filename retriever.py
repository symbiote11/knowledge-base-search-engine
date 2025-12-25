from langchain_community.vectorstores import FAISS
from app.config import embeddings

def load_retriever(index_path: str):
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3})
