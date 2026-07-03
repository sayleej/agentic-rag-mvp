"""FastAPI backend — the single phone line between the UI and the RAG pipeline.

Run from the project root:
    .venv/bin/uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from backend import config
from backend.embeddings import embed_query
from backend.responder import answer
from backend.vector_store import count_chunks, search

app = FastAPI(title="Agentic RAG MVP")


class ChatMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


@app.get("/health")
def health():
    """Lets the UI show whether the backend and its index are alive."""
    missing = config.validate()
    if missing:
        return {"status": "misconfigured", "missing": missing}
    try:
        chunks = count_chunks()
    except Exception as e:
        return {"status": "qdrant_unreachable", "error": str(e)}
    return {"status": "ok", "indexed_chunks": chunks}


@app.post("/query")
def query(request: QueryRequest):
    """One chat turn: embed the question, search Qdrant, ask Groq."""
    query_vector = embed_query(request.question)
    chunks = search(query_vector)
    history = [m.model_dump() for m in request.history]
    reply = answer(request.question, chunks, history)
    return {
        "answer": reply,
        "sources": [
            {"source": c["source"], "score": c["score"], "text": c["text"]}
            for c in chunks
        ],
    }
