"""Combined-mode app for Streamlit Community Cloud.

Runs the whole RAG pipeline inside one Streamlit process — no separate
FastAPI backend. Used only for cloud hosting; local development keeps
the two-app setup (backend/main.py + ui/app.py).

Deploy: share.streamlit.io -> New app -> this repo, main branch,
streamlit_app.py. Paste API keys into the app's Secrets settings.
"""

import os

import streamlit as st

# On Streamlit Cloud, keys live in st.secrets (not a .env file).
# Copy them into environment variables BEFORE importing backend modules,
# because backend/config.py reads the environment at import time.
for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from backend import config
from backend.embeddings import embed_query
from backend.responder import answer
from backend.vector_store import count_chunks, search

st.set_page_config(page_title="Docs Assistant", page_icon="📚")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    with st.expander("📄 Sources"):
        for s in sources:
            st.markdown(f"**{s['source']}** (relevance {s['score']:.2f})")
            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))


with st.sidebar:
    st.title("📚 Docs Assistant")
    st.caption("Agentic RAG MVP — Phase 1")

    missing = config.validate()
    if missing:
        st.error(f"Missing secrets: {', '.join(missing)}")
    else:
        try:
            st.success(f"Knowledge base online — {count_chunks()} chunks indexed")
        except Exception as e:
            st.error(f"Cannot reach Qdrant: {e}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

question = st.chat_input("Ask about Kubernetes jobs, cron jobs, or autoscaling...")

# Empty state: tell first-time users what this assistant knows.
if not st.session_state.messages:
    st.title("📚 Kubernetes Docs Assistant")
    st.markdown(
        "I answer questions about a **Kubernetes operations** document library — "
        "jobs, cron jobs, monitoring, parallel work queues, and pod autoscaling. "
        "Every answer cites the documents it came from, and I'll say so plainly "
        "if the docs don't cover your question."
    )
    st.markdown("**Try one of these:**")
    for sample in (
        "How does pod autoscaling work?",
        "What is a CronJob and when would I use one?",
        "How do I monitor a running job?",
    ):
        if st.button(sample):
            question = sample

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and writing an answer..."):
            try:
                chunks = search(embed_query(question))
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                reply = answer(question, chunks, history)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        st.markdown(reply)
        render_sources(chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": chunks}
    )
