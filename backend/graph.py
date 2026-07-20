"""The LangGraph agent: planner -> (retriever ->) responder, with memory.

This replaces the hand-written if/else orchestration that used to live in
main.py. Same logic, now expressed as an explicit state machine:

    START -> planner --conversational--> responder -> END
                  \--technical--> retriever -> responder -> END

Why add this now: the flow gained a real second branch worth naming as a
graph (routing + retrieval + generation), and a checkpointed graph gives us
server-side conversation memory (per thread_id) instead of re-sending the
whole history from the browser on every request.
"""

from __future__ import annotations

import logfire
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend import config
from backend.embeddings import embed_query
from backend.planner import plan
from backend.reranker import rerank
from backend.responder import answer, answer_conversational
from backend.state import AgentState
from backend.vector_store import search


def planner_node(state: AgentState) -> dict:
    with logfire.span("planner"):
        decision = plan(state["question"], state["history"])
        step = (
            "Planner: conversational — retrieval skipped"
            if decision["intent"] == "conversational"
            else f"Planner: technical — search query: '{decision['search_query']}'"
        )
        return {
            "intent": decision["intent"],
            "search_query": decision["search_query"] or "",
            "steps": state["steps"] + [step],
        }


def route_after_planner(state: AgentState) -> str:
    return "retriever" if state["intent"] == "technical" else "responder"


def retriever_node(state: AgentState) -> dict:
    with logfire.span("retriever"):
        with logfire.span("embed + vector search"):
            query_vector = embed_query(state["search_query"])
            candidates = search(
                query_vector,
                limit=config.CANDIDATES,
                include_noisy=state["include_noisy"],
            )
        with logfire.span("rerank"):
            chunks = rerank(state["search_query"], candidates)
        steps = state["steps"] + [
            f"Retrieved {len(candidates)} candidates from Qdrant (vector search)",
            f"Reranked to top {len(chunks)} chunks (cross-encoder)",
        ]
        return {"chunks": chunks, "steps": steps}


def responder_node(state: AgentState) -> dict:
    with logfire.span("responder"):
        if state["intent"] == "conversational":
            reply = answer_conversational(state["question"], state["history"])
            step = "Response generated from conversation memory"
        else:
            reply = answer(state["question"], state["chunks"], state["history"])
            step = "Grounded answer generated with citations"
        return {"answer": reply, "steps": state["steps"] + [step]}


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("responder", responder_node)

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {"retriever": "retriever", "responder": "responder"},
    )
    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)

    # In-memory checkpointer keyed by thread_id — gives each conversation
    # its own persisted state across turns. Swappable for a Postgres
    # checkpointer later without touching any node.
    return workflow.compile(checkpointer=MemorySaver())


rag_agent = build_graph()
