"""RAGAS-style metrics — the four standard RAG quality dimensions, computed
with our own LLM judge instead of the ragas package (which pulls in torch
and sentence-transformers — too heavy for this project's free-tier design).

Same idea as the industry framework, lighter dependency:

  faithfulness       — is every claim in the answer actually supported by
                        the retrieved context? (catches hallucination)
  answer_relevancy    — does the answer actually address the question asked,
                        independent of whether it's factually correct?
  context_precision   — of the chunks retrieved, how many are relevant to
                        the question? (measures retrieval noise)
  context_recall      — does the retrieved context contain everything the
                        reference answer needed? (measures retrieval gaps)

All four are 0.0-1.0. One LLM call computes all four at once to keep API
cost down — a single request instead of four.
"""

from __future__ import annotations

import json

from backend.llm import chat

RAGAS_PROMPT = """You are a RAG evaluation judge. Score four dimensions of a
retrieval-augmented answer, each from 0.0 to 1.0.

1. faithfulness: Of the factual claims made in the ANSWER, what fraction
   are actually supported by the RETRIEVED CONTEXT? 1.0 = every claim is
   grounded in the context. 0.0 = the answer invents things not in the
   context. This is the hallucination check — score ONLY against the
   context, ignore whether the answer happens to be true in the world.

2. answer_relevancy: Does the ANSWER actually address the QUESTION asked,
   regardless of whether it's correct? 1.0 = directly on-topic and
   answers what was asked. 0.0 = off-topic or answers a different question.

3. context_precision: Of the chunks in RETRIEVED CONTEXT, what fraction
   are actually relevant to answering the QUESTION? 1.0 = every retrieved
   chunk is useful. Low score = a lot of retrieved noise.

4. context_recall: Does RETRIEVED CONTEXT contain all the information
   the REFERENCE ANSWER needed? Compare against REFERENCE ANSWER, not the
   model's answer. 1.0 = nothing needed was missing from retrieval.

Reply with ONLY a JSON object:
{"faithfulness": <0-1>, "answer_relevancy": <0-1>, "context_precision": <0-1>, "context_recall": <0-1>}"""


def score(question: str, contexts: list[str], model_answer: str, reference_answer: str) -> dict:
    """Compute all four RAGAS-style metrics in one judge call."""
    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context retrieved)"
    reply = chat(
        messages=[
            {"role": "system", "content": RAGAS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"RETRIEVED CONTEXT:\n{context_block}\n\n"
                    f"ANSWER:\n{model_answer}\n\n"
                    f"REFERENCE ANSWER:\n{reference_answer}"
                ),
            },
        ],
        temperature=0.0,
        json_mode=True,
    )
    result = json.loads(reply)
    return {
        "faithfulness": round(float(result.get("faithfulness", 0)), 3),
        "answer_relevancy": round(float(result.get("answer_relevancy", 0)), 3),
        "context_precision": round(float(result.get("context_precision", 0)), 3),
        "context_recall": round(float(result.get("context_recall", 0)), 3),
    }
