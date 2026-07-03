"""FastAPI backend — the single phone line between the UI and the RAG pipeline.

Run from the project root:
    .venv/bin/uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from backend import config
from backend.embeddings import embed_query
from backend.guardrails import BLOCKED_MESSAGE, check
from backend.planner import plan
from backend.reranker import rerank
from backend.responder import answer, answer_conversational
from backend.vector_store import count_chunks, search

app = FastAPI(title="Agentic RAG MVP")


class ChatMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    include_noisy: bool = False  # search the noise documents too


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
    from backend.llm import gateway_status

    return {"status": "ok", "indexed_chunks": chunks, "llm_via": gateway_status()}


@app.post("/query")
def query(request: QueryRequest):
    """One chat turn: plan, then either search + grounded answer, or chat."""
    history = [m.model_dump() for m in request.history]

    # Gate 0: guardrails — hostile input stops here, before any other spend.
    verdict = check(request.question)
    if not verdict["allowed"]:
        return {
            "answer": BLOCKED_MESSAGE,
            "sources": [],
            "plan": {"intent": f"blocked ({verdict['category']})", "search_query": None},
        }

    decision = plan(request.question, history)

    if decision["intent"] == "conversational":
        reply = answer_conversational(request.question, history)
        return {
            "answer": reply,
            "sources": [],
            "plan": {"intent": "conversational", "search_query": None},
        }

    query_vector = embed_query(decision["search_query"])
    candidates = search(
        query_vector,
        limit=config.CANDIDATES,
        include_noisy=request.include_noisy,
    )
    chunks = rerank(decision["search_query"], candidates)
    reply = answer(request.question, chunks, history)
    return {
        "answer": reply,
        "sources": [
            {"source": c["source"], "score": c["score"], "text": c["text"]}
            for c in chunks
        ],
        "plan": {"intent": "technical", "search_query": decision["search_query"]},
    }
