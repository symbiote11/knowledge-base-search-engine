# Knowledge-base Search Engine (RAG)

## Objective
Search across uploaded documents and generate synthesized answers using an LLM-based Retrieval-Augmented Generation (RAG).

## Tech Stack
- FastAPI
- LangChain
- FAISS
- OpenAI GPT-3.5

## Setup
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key
uvicorn app.main:app --reload
