"""Planner — decide whether a message needs a document search.

The "agentic" step: before doing any retrieval, a fast LLM call classifies
the message and, when a search IS needed, rewrites it into a sharp,
self-contained search query (resolving pronouns like "it" or "them"
using the conversation history).
"""

from __future__ import annotations

import json

from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


PLANNER_PROMPT = """You are the routing step of a document Q&A assistant.
The document library covers Kubernetes operations: jobs, cron jobs, job
monitoring, parallel work queues, and pod autoscaling.

Classify the user's latest message:
- "technical": ANY information-seeking question — even if it seems outside
  the library's topics. The search step decides what is answerable; your job
  is only to distinguish questions from chit-chat.
- "conversational": greetings, thanks, chit-chat, or questions about the
  conversation itself ("what did I just ask?").

If technical, also write a self-contained search query: resolve pronouns
and references using the conversation history, and phrase it as a clear
topic description.

Reply with ONLY a JSON object, no other text:
{"intent": "technical" | "conversational", "search_query": "<query or empty string>"}"""


def plan(question: str, history: list[dict]) -> dict:
    """Return {"intent": ..., "search_query": ...} for the message.

    Fails safe: any error means we treat the message as technical and
    search with the original wording — a wasted search beats a lost answer.
    """
    history_text = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in history[-6:]
    ) or "(no history)"

    try:
        response = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": f"CONVERSATION HISTORY:\n{history_text}\n\nLATEST MESSAGE:\n{question}",
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        decision = json.loads(response.choices[0].message.content)
        intent = decision.get("intent", "technical")
        search_query = (decision.get("search_query") or "").strip() or question
        if intent not in ("technical", "conversational"):
            intent = "technical"
        return {"intent": intent, "search_query": search_query}
    except Exception:
        return {"intent": "technical", "search_query": question}
