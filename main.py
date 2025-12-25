import os
import shutil
from fastapi import FastAPI, UploadFile, File
from ingestion import load_and_split_docs
from embeddings import create_faiss_index
from rag import get_rag_chain

app = FastAPI()

DATA_DIR = "data/documents"
INDEX_DIR = "vectorstore/faiss_index"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

@app.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"{DATA_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = load_and_split_docs(file_path)
    create_faiss_index(docs, INDEX_DIR)

    return {"message": "Document uploaded and indexed successfully"}

@app.post("/ask")
async def ask_question(question: str):
    qa_chain = get_rag_chain(INDEX_DIR)
    answer = qa_chain.run(question)
    return {"answer": answer}
