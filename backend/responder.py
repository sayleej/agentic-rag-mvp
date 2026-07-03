"""Responder — hand retrieved chunks + the question to Groq for an answer.

This is the "generation" half of RAG. The prompt tells the model to
answer ONLY from the provided context, which is what keeps answers
grounded in your documents instead of the model's imagination.
"""

from __future__ import annotations

from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


CONVERSATIONAL_PROMPT = """You are a friendly assistant for a Kubernetes documentation Q&A product.
The user's message is small talk or about the conversation itself — no document
search was needed. Reply naturally and briefly, using the conversation history
for context. If asked what you can do, say you answer questions about the
Kubernetes operations document library (jobs, cron jobs, monitoring, work
queues, pod autoscaling)."""

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a document library.

Rules:
- Answer using ONLY the context provided below. Do not use outside knowledge.
- If the context does not contain the answer, say so plainly and suggest \
what the user might ask instead. Never invent an answer.
- When you use information from the context, mention which source document \
it came from (the source name appears above each context block).
- Keep answers clear and concise."""


def build_prompt(question: str, chunks: list[dict], history: list[dict]) -> list[dict]:
    """Assemble the message list Groq receives."""
    context_blocks = [
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in chunks
    ]
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context found)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent conversation so follow-up questions make sense.
    # Last 6 turns is plenty for an MVP and keeps the prompt small.
    messages.extend(history[-6:])

    messages.append(
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
        }
    )
    return messages


def answer(question: str, chunks: list[dict], history: list[dict]) -> str:
    """Generate a grounded answer from the retrieved chunks."""
    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=build_prompt(question, chunks, history),
        temperature=0.1,  # low = factual and consistent, not creative
    )
    return response.choices[0].message.content


def answer_conversational(question: str, history: list[dict]) -> str:
    """Reply to small talk — no documents involved."""
    messages = [{"role": "system", "content": CONVERSATIONAL_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})
    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.5,  # a bit warmer — this is conversation, not citation
    )
    return response.choices[0].message.content
