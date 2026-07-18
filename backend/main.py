"""FastAPI backend — the single phone line between the UI and the RAG pipeline.

Run from the project root:
    .venv/bin/uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

from typing import Optional

import logfire
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from backend import config
from backend.graph import rag_agent
from backend.guardrails import BLOCKED_MESSAGE, TOO_LONG_MESSAGE, check
from backend.vector_store import count_chunks

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
    thread_id: str = "default"   # keys the graph's server-side memory per conversation


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Gate every non-health route behind a shared secret.

    Skipped entirely when API_KEY isn't set in config — keeps local dev
    and the free public demo frictionless, while giving any deployment
    that DOES set the key a real auth boundary.
    """
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


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


@app.post("/query", dependencies=[Depends(require_api_key)])
def query(request: QueryRequest):
    """One chat turn: guard, then run the LangGraph agent (plan -> retrieve? -> respond)."""
    history = [m.model_dump() for m in request.history]

    # Gate 0: guardrails — hostile input stops here, before any other spend.
    # Kept outside the graph deliberately: a blocked message shouldn't touch
    # the agent's state or its checkpointed memory at all.
    with logfire.span("guardrails"):
        verdict = check(request.question)
    if not verdict["allowed"]:
        message = TOO_LONG_MESSAGE if verdict["category"] == "too_long" else BLOCKED_MESSAGE
        return {
            "answer": message,
            "sources": [],
            "plan": {"intent": f"blocked ({verdict['category']})", "search_query": None},
            "steps": [f"Guardrails: blocked ({verdict['category']}) — pipeline stopped"],
        }

    initial_state = {
        "question": request.question,
        "history": history,
        "include_noisy": request.include_noisy,
        "intent": "",
        "search_query": "",
        "chunks": [],
        "answer": "",
        "steps": ["Guardrails: passed"],
    }
    graph_config = {"configurable": {"thread_id": request.thread_id}}

    with logfire.span("agent graph"):
        final_state = rag_agent.invoke(initial_state, config=graph_config)

    chunks = final_state.get("chunks", [])
    return {
        "answer": final_state["answer"],
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
        "plan": {
            "intent": final_state["intent"],
            "search_query": final_state["search_query"] or None,
        },
        "steps": final_state["steps"],
    }
