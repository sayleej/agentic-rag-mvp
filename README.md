# Agentic RAG MVP

A retrieval-augmented generation (RAG) assistant that answers questions about
your documents — grounded, cited, and honest when it doesn't know.

**Phase 1 scope:** document ingestion → Gemini embeddings → Qdrant vector
search → Groq answer generation → Streamlit chat UI.
(Planned later phases: query planner, reranking, guardrails, LLM gateway, evals.)

📖 **New to RAG?** Read the [deep understanding guide](docs/UNDERSTANDING.md) —
a plain-English walkthrough of every component, the product decisions behind
them, and the real incidents hit during the build.

## Architecture

```
                      INGESTION (run once)
data/  ──►  loaders  ──►  chunker  ──►  Gemini embeddings  ──►  Qdrant Cloud
(PDF, DOCX,  (to plain    (1500-char                            (411 vectors,
 HTML, TXT)   text)        paragraphs)                           cosine search)

                      CHAT (every question)
Streamlit UI ──► FastAPI backend ──► Gemini (embed question)
                                 ──► Qdrant (top-5 closest chunks)
                                 ──► Groq gpt-oss-120b (grounded answer)
```

- **Embeddings:** `gemini-embedding-001`, 3072 dimensions, separate
  document/query task types for better retrieval.
- **Answering:** `openai/gpt-oss-120b` on Groq — instructed to answer only
  from retrieved context, cite sources, and refuse when the answer isn't there.

## Setup

1. Python 3.9+ and a virtualenv:
   ```
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in free API keys from
   [Google AI Studio](https://aistudio.google.com/apikey),
   [Groq](https://console.groq.com/keys), and
   [Qdrant Cloud](https://cloud.qdrant.io).
3. Drop documents into `data/` (PDF, DOCX, HTML, TXT, MD).

## Run

```
# 1. Ingest documents (re-run whenever data/ changes)
.venv/bin/python -m backend.ingest

# 2. Start the backend
.venv/bin/uvicorn backend.main:app --port 8000

# 3. Start the UI (new terminal)
.venv/bin/streamlit run ui/app.py
```

Open http://localhost:8501 and ask away.
