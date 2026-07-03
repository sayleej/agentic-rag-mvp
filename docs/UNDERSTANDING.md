# Deep Understanding Guide — Agentic RAG MVP (Phase 1)

A plain-English walkthrough of what this system is, how every piece works,
and the product decisions behind it. Written for a product-minded reader;
no coding background assumed.

---

## 1. What problem does this product solve?

Large language models are brilliant writers with two flaws: they don't know
*your* documents, and when they don't know something, they sometimes make
it up (hallucination). **RAG — Retrieval-Augmented Generation** — fixes both
by splitting the job in two:

- **Retrieval**: find the exact passages in your document library that
  relate to the question.
- **Generation**: have the LLM write an answer *using only those passages*,
  with citations.

The result: answers grounded in your documents, with receipts.

---

## 2. The architecture in one picture

```
                    INGESTION (run once per library update)
data/ ──► loaders ──► chunker ──► Gemini embeddings ──► Qdrant Cloud
(PDF, DOCX,  (to plain   (1500-char                       (vector database)
 HTML, TXT)   text)       paragraphs)

                    CHAT (every question)
Streamlit UI ──► FastAPI backend ──► 1. Gemini: fingerprint the question
                                 ──► 2. Qdrant: find 5 closest chunks
                                 ──► 3. Groq: write a grounded answer
```

Three external services, in this exact order, every single question:
**Gemini → Qdrant → Groq**. The UI never talks to any of them directly —
only to our own backend. That separation means any service can be swapped
without touching the UI.

---

## 3. Component by component

### Loaders (`backend/loaders.py`) — the translators
Documents arrive in four "languages" (PDF, Word, HTML, plain text).
Each loader knows how to read one format and returns the same thing:
plain text. The HTML loader is a clipping service — it keeps the article
and throws away the menus, scripts, and footers.

### Chunker (`backend/chunker.py`) — the portion cutter
Instead of shelving whole books, we cut them into **index cards** of up to
1,500 characters — never cutting mid-paragraph, so each card stays one
coherent thought.

**Why not bigger chunks?** Two reasons:
1. *Retrieval precision* — you want the three paragraphs that answer the
   question, not a 40-page document that mentions it somewhere.
2. *Sharp fingerprints* — the embedding of a whole book is "about
   everything," so it matches nothing well. The embedding of two focused
   paragraphs is sharp.

### Embeddings (`backend/embeddings.py`) — the fingerprint machine
Gemini's `gemini-embedding-001` turns any text into **3,072 numbers that
capture its meaning**. Texts with similar meaning get similar numbers.
This is the entire trick behind searching by meaning instead of keywords.

Detail worth knowing: we embed documents and questions in **different
modes** (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`). Gemini tunes the
fingerprints for each side of the search, which measurably improves
match quality. Most tutorials skip this.

### Vector store (`backend/vector_store.py`) — the card catalog
Qdrant Cloud stores every card with its fingerprint. "Closest match" is
measured by **cosine similarity** — roughly, the angle between two
fingerprints; small angle = similar meaning. Scores run 0 to 1; our
relevant matches score ~0.7.

### Responder (`backend/responder.py`) — the ghostwriter
Takes the question + the 5 retrieved cards and asks Groq's
`openai/gpt-oss-120b` to write the answer. The intelligence is in the
**system prompt**, which enforces three rules:
1. **Answer ONLY from the provided context** (the anti-hallucination rule).
2. **Say "I don't know" when the context doesn't cover it** — an honest
   refusal beats a confident fabrication.
3. **Cite the source document** for every claim — trust through transparency.

Temperature is 0.1: factual mode, not creative-writing mode.

### Backend (`backend/main.py`) — the phone line
A FastAPI server with two endpoints: `/health` (are you alive, how many
chunks indexed?) and `/query` (one full chat turn). The entire RAG
pipeline is four lines here — everything else in the codebase exists so
those four lines can be that simple.

### UI (`ui/app.py`, `streamlit_app.py`) — the storefront
A Streamlit chat window with three product features:
- **Sources expander** under every answer — users can audit where each
  answer came from.
- **Health indicator** — turns "mystery failure" into "clear next step."
- **Empty state** — a welcome screen with clickable sample questions, so
  first-time users are never staring at a blank box wondering what to ask.

`streamlit_app.py` is a combined-mode twin used only for cloud hosting
(Streamlit Community Cloud can't run a separate backend process).

---

## 4. Key product decisions (and the reasoning)

| Decision | Choice | Why |
|---|---|---|
| Search every question, or add a planner? | Always search (Phase 1) | Fewer moving parts; planner is a deliberate Phase 2 upgrade |
| One app or two? | Two (backend + UI) | Clean separation; the backend could serve other clients later |
| Qdrant hosting | Cloud | Free tier, survives reinstalls, closer to production reality |
| Chunk size | 1,500 chars, paragraph-aware | Balance between precision and context |
| Noise in the index | Yes, deliberately | Proof that retrieval stays precise with 85% junk in the library |
| Temperature | 0.1 | Factual consistency over creativity |

---

## 5. War stories (real incidents from the build)

**The deprecated model.** The reference project used Groq's
`llama-3.3-70b-versatile` — which Groq deprecated in June 2026, while we
were building. Lesson: in AI products, model lifecycles are a product
risk. Pin versions consciously and keep the model name in one config file
so swapping takes one line.

**The rate limit.** Gemini's free tier allows 100 embedding requests per
minute. Ingesting a 288-chunk document blew through it mid-run. First fix
(retry after 1–4 seconds) failed — the quota window is a minute long, so
the retries have to be *patient* (20–60 seconds). Lesson: read the error;
Google literally tells you how long to wait.

**The oversized upload.** After embedding succeeded, uploading 288
fingerprints (~30 KB each) to Qdrant in one request timed out mid-transfer.
Fix: batch uploads, 32 at a time. Lesson: fixing one bottleneck reveals
the next one downstream — pipelines fail in sequence.

**The noise experiment.** With 411 chunks indexed and 85% of them
deliberate junk (malloc tutorials, graphics pipeline docs), every test
question returned only chunks from the correct document. Semantic search
doesn't get distracted by volume.

---

## 6. Hallucination vs. scope — a distinction that wins interviews

If a user asks "how do I write a memory allocator?" and the system answers
correctly from the malloc tutorial we ingested as noise — **that is not
hallucination**. Every word is grounded and citable. It's a **corpus scope
problem**: our curation let that document in.

- Hallucination = the model invents things found in no source.
  Fixed with prompts and guardrails.
- Scope creep = the system truthfully answers from documents that
  shouldn't be in (or should be filtered out of) the library.
  Fixed with curation or metadata filtering.

"First, let's check whether it's actually hallucination or a
retrieval/scope issue" is a differentiated answer to the interview
question "how would you reduce hallucinations?"

---

## 7. Phase 2 — agentic retrieval (shipped)

### The planner (`backend/planner.py`) — the receptionist
Before any search, a fast LLM call does two jobs:
1. **Routing** — small talk ("hi", "thanks") skips retrieval entirely and
   gets a conversational reply; real questions go to the library.
2. **Query rewriting** — vague follow-ups become self-contained search
   queries. "What are the downsides of the vertical one?" (after a pod
   autoscaling discussion) becomes "downsides of vertical pod autoscaling."
   Without this, we'd embed the literal words "the vertical one," which
   match nothing.

It fails safe: if the planner call errors, the message is treated as
technical and searched with its original wording — a wasted search beats
a lost answer.

**War story — the emergent guardrail.** The first planner prompt described
the library as "Kubernetes docs," and the planner started silently
classifying *any* non-Kubernetes question as small talk — an accidental
scope guardrail that broke the noise-toggle demo. Fix: one prompt
paragraph clarifying that the planner only separates questions from
chit-chat, and the search results decide what's answerable. Lesson:
prompts are product specs, and they get bugs like any spec. Test them.

### Source filtering (`data/true/`, `data/noisy/`) — the stickers
Every chunk now carries a `source_type` label, assigned from its folder
at ingestion. Search defaults to curated ("true") documents only; a UI
toggle opens the full library for the noise demonstration. The 411
already-indexed chunks were relabeled **in place** — updating a card's
sticker doesn't require re-fingerprinting it, so this cost zero embedding
quota. (Qdrant detail: filtering on a field requires a payload index on
that field, like a database column index.)

This feature is the productized version of the hallucination-vs-scope
lesson in section 6: off-topic answers weren't a model problem, so the
fix isn't a model fix — it's metadata and filtering.

### Reranking (`backend/reranker.py`) — the careful second reader
Two-stage retrieval: Qdrant fetches 15 candidates fast (comparing
pre-computed embeddings — quick but nearsighted), then FlashRank's
cross-encoder re-reads the query and each candidate *together* and keeps
the best 5. Analogy: skim 100 resumes to shortlist 15, then do real
interviews for the final 5. FlashRank runs locally on the CPU — no API,
no key, no per-question cost. It fails safe too: if the model can't load,
results pass through in vector order.

---

## 8. Phase 3 — production trust (shipped)

### Evals (`evals/`) — the fixed exam
A golden dataset of 12 questions: 9 answerable (each pinned to the
document that must be retrieved, with a reference answer) and 3
out-of-scope traps where refusing IS the correct behavior. The runner
(`python -m evals.run`) measures three things: retrieval hit rate
(objective — right document in the sources?), answer quality (an
LLM-as-judge grades against the reference answers, 1–5), and refusal
accuracy. Results are saved with timestamps so every future change gets
judged by numbers, not vibes. First run: 100% hit rate, 4.9/5 quality,
100% refusal accuracy — a score that good partly means the exam is too
easy; real eval discipline is adding questions until some fail.

### LLM gateway (`backend/llm.py`) — the switchboard
Every LLM call (planner, responder, judge, guard) goes through one
doorway. If Portkey keys are configured, calls route through the Portkey
gateway, which records latency, tokens, cost, and errors in a dashboard;
without keys, calls fall back to Groq directly and nothing breaks. The
"single doorway" refactor is why adding the gateway took minutes —
centralize the things you'll want to instrument.

### Guardrails (`backend/guardrails.py`) — the bouncer
Gate zero, before the planner: an LLM screening call blocks abuse,
prompt-injection attempts ("ignore your instructions..."), and harmful
requests — and deliberately does NOT block merely off-topic questions
(the planner and responder handle those gracefully; off-topic is not
hostile). Blocked messages stop at one cheap call instead of the full
pipeline. It fails open: if the guard errors, the message passes and the
responder's grounding rules remain the second line of defense —
availability over strictness for a demo; a bank would choose the
opposite.

---

## 9. Glossary

- **RAG** — Retrieval-Augmented Generation; search your documents first,
  then have an LLM answer from what was found.
- **Embedding / vector** — a list of numbers (here 3,072) representing
  the *meaning* of a piece of text.
- **Vector database** — a search engine for embeddings; finds
  nearest-in-meaning, not matching-in-keywords.
- **Cosine similarity** — the closeness measure between two embeddings.
- **Chunk** — one index card of text (≤1,500 characters here).
- **Hallucination** — an LLM stating things found in no source.
- **System prompt** — standing instructions the model receives before
  every question; where the product's behavioral rules live.
- **Temperature** — the model's creativity dial; low = consistent and
  factual.
- **Rate limit** — an API's requests-per-minute ceiling; free tiers are
  tight, production code must retry patiently.
- **Empty state** — what a user sees before their first interaction;
  one of the highest-leverage screens in any product.
