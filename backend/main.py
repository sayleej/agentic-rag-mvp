"""FastAPI backend — the single phone line between the UI and the RAG pipeline.

Run from the project root:
    .venv/bin/uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

import logfire
from fastapi import FastAPI
from pydantic import BaseModel

from backend import config
from backend.embeddings import embed_query
from backend.guardrails import BLOCKED_MESSAGE, check
from backend.planner import plan
from backend.reranker import rerank
from backend.responder import answer, answer_conversational
from backend.vector_store import count_chunks, search

# Sends traces to the Logfire dashboard only when LOGFIRE_TOKEN is set;
# otherwise spans are no-ops and nothing leaves the machine.
logfire.configure(service_name="agentic-rag-backend", send_to_logfire="if-token-present")

app = FastAPI(title="Agentic RAG MVP")
logfire.instrument_fastapi(app)


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
    """One chat turn: guard, plan, then either search + grounded answer, or chat."""
    history = [m.model_dump() for m in request.history]
    steps: list[str] = []

    # Gate 0: guardrails — hostile input stops here, before any other spend.
    with logfire.span("guardrails"):
        verdict = check(request.question)
    if not verdict["allowed"]:
        steps.append(f"Guardrails: blocked ({verdict['category']}) — pipeline stopped")
        return {
            "answer": BLOCKED_MESSAGE,
            "sources": [],
            "plan": {"intent": f"blocked ({verdict['category']})", "search_query": None},
            "steps": steps,
        }
    steps.append("Guardrails: passed")

    with logfire.span("planner"):
        decision = plan(request.question, history)

    if decision["intent"] == "conversational":
        steps.append("Planner: conversational — retrieval skipped")
        with logfire.span("responder (conversational)"):
            reply = answer_conversational(request.question, history)
        steps.append("Response generated from conversation memory")
        return {
            "answer": reply,
            "sources": [],
            "plan": {"intent": "conversational", "search_query": None},
            "steps": steps,
        }

    steps.append(f"Planner: technical — search query: '{decision['search_query']}'")
    with logfire.span("embed query"):
        query_vector = embed_query(decision["search_query"])
    with logfire.span("qdrant search"):
        candidates = search(
            query_vector,
            limit=config.CANDIDATES,
            include_noisy=request.include_noisy,
        )
    steps.append(f"Retrieved {len(candidates)} candidates from Qdrant (vector search)")
    with logfire.span("rerank"):
        chunks = rerank(decision["search_query"], candidates)
    steps.append(f"Reranked to top {len(chunks)} chunks (cross-encoder)")
    with logfire.span("responder"):
        reply = answer(request.question, chunks, history)
    steps.append("Grounded answer generated with citations")
    return {
        "answer": reply,
        "sources": [
            {
                "source": c["source"],
                "score": c["score"],
                "vector_score": c.get("vector_score"),
                "rerank_score": c.get("rerank_score"),
                "text": c["text"],
            }
            for c in chunks
        ],
        "plan": {"intent": "technical", "search_query": decision["search_query"]},
        "steps": steps,
    }
