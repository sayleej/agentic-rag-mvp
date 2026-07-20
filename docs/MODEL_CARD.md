# Model Card — Agentic RAG MVP

*Last updated: 2026-07-20, based on eval run `evals/results/20260718-171232.json`*

## Model Details

This isn't a single trained model — it's a **pipeline of models and components**
working together:

| Component | What it does |
|---|---|
| Groq-hosted LLM (via Portkey gateway) | Intent classification, guardrail classification, response generation, eval judging |
| Google Gemini embeddings | Turns questions and documents into vectors for search |
| Qdrant | Vector database — stores and searches document embeddings |
| FlashRank (local cross-encoder) | Reranks candidate chunks for precision, no external API call |
| LangGraph | Orchestrates the pipeline as a stateful graph, with per-conversation memory |

No custom model was trained. This is a **retrieval-augmented generation (RAG)
system built on existing, off-the-shelf models**, orchestrated and evaluated
by custom code.

## Intended Use

Answering questions about a specific **Kubernetes operations documentation
library** — jobs, cron jobs, monitoring, parallel work queues, and pod
autoscaling.

**Not intended for:**
- General-purpose chat or advice outside the document library's scope
- Any use case requiring guaranteed factual accuracy without human review
- Multi-tenant or multi-user production deployment (see Limitations)

## Data

- **Curated corpus**: Kubernetes documentation (`.docx`, `.html`, `.txt`, `.pdf`)
- **Noisy corpus** (optional, toggleable): a deliberately mixed-in set of
  off-topic documents, used to test retrieval scope precision — not part of
  the default search unless explicitly enabled
- **Golden evaluation set**: 18 hand-written items — 9 answerable questions
  with reference answers, 3 out-of-scope trap questions, 6 guardrail test
  cases (injection, abuse, and legitimate-off-topic)

## Evaluation Results

*From the most recent full eval run (2026-07-18):*

| Metric | Score |
|---|---|
| Retrieval hit rate | 100% |
| Answer quality (LLM judge, 1-5) | 5.0/5 |
| Refusal accuracy (out-of-scope) | 100% |
| Guardrail accuracy | 100% |
| Guardrail precision | 1.0 |
| Guardrail recall | 1.0 |
| Guardrail confusion matrix | TP=3, TN=3, FP=0, FN=0 |
| RAGAS faithfulness | 0.93 |
| RAGAS answer relevancy | 1.0 |
| RAGAS context precision | 0.383 |
| RAGAS context recall | 0.978 |
| Overall pass rate | 100% |

Scored using a hand-written golden dataset, an LLM-as-judge compared against
reference answers, and the real `ragas` library. Full methodology in
[`evals/run.py`](../evals/run.py).

## Known Limitations

Stated honestly, not glossed over:

- **Context precision (0.383) is the weakest metric.** Retrieval correctly
  surfaces the right document every time (100% hit rate), but often pulls in
  additional chunks that aren't actually relevant to the specific question.
  The reranker narrows candidates but doesn't fully solve this — a clear next
  optimization target, not a hidden problem.
- **Golden dataset is small (18 items).** A 100% pass rate reflects strong
  performance on these specific, fairly clear-cut test cases — it does not
  mean the system is flawless on arbitrary real-world inputs or adversarial
  edge cases beyond what's tested.
- **No long-term or episodic memory.** Conversation memory (via LangGraph's
  `MemorySaver`) is session-scoped only — it doesn't persist across sessions
  or server restarts, and there's no cross-session user memory.
- **Guardrail fails open.** If the safety classifier's LLM call itself
  errors, the message is allowed through rather than blocked. This is a
  deliberate availability-over-strictness tradeoff for a demo product — a
  regulated deployment should flip this to fail-closed.
- **Input length is capped (2000 characters) but not otherwise rate-limited.**
  There's no per-user request throttling or abuse-pattern detection beyond
  the guardrail's hostile-content classification.
- **Single-tenant only.** No role-based access control, no multi-tenant data
  isolation — not needed for a single-user public demo, but a real gap if
  this were extended to serve multiple organizations.
- **Retrieval is Kubernetes-domain-specific.** The embedding model and corpus
  are not tuned or validated for any other domain; this is not a
  general-purpose RAG template without rework.

## Responsible AI Notes

- **Guardrail scope is deliberately narrow**: it blocks hostile input (abuse,
  prompt injection, harmful requests) only. It does *not* block off-topic
  questions — that's handled separately, by the planner routing and the
  responder's honest-refusal instruction. This separation is intentional:
  conflating "hostile" and "off-topic" would make both harder to test and
  reason about independently.
- **Grounding, not the guardrail, prevents hallucination.** The responder's
  system prompt requires answers to be sourced only from retrieved context,
  with citations, and to refuse honestly when context doesn't cover the
  question. RAGAS faithfulness (0.93) is the primary check on whether this
  is actually working.
- **No human-in-the-loop review step exists.** All guardrail and retrieval
  decisions are fully automated; there is no manual approval or override
  path in the current design.
