import os, shutil
from fastapi import FastAPI, UploadFile, File
from app.ingestion import load_and_split_docs
from app.embeddings import create_faiss_index
from app.rag import get_rag_chain

app = FastAPI()
DATA_DIR = "data/documents"
INDEX_DIR = "vectorstore/faiss_index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

@app.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    path = f"{DATA_DIR}/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    docs = load_and_split_docs(path)
    create_faiss_index(docs, INDEX_DIR)
    return {"message": "Indexed"}

@app.post("/ask")
async def ask(question: str):
    chain = get_rag_chain(INDEX_DIR)
    return {"answer": chain.run(question)}
