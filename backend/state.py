"""Shared state passed between LangGraph nodes.

Every node reads from and writes to this one dict-like object — it's how
the planner tells the responder what it decided, and how the retriever
hands the responder its chunks, without any node calling another directly.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    question: str
    history: list[dict]            # prior turns: [{"role": ..., "content": ...}]
    include_noisy: bool

    intent: str                    # "conversational" | "technical"
    search_query: str              # the (possibly rewritten) query to embed
    chunks: list[dict]             # reranked chunks, populated by the retriever node
    answer: str                    # the final response
    steps: list[str]               # human-readable trace of what happened
