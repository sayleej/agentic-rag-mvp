"""Streamlit chat UI — the face of the RAG assistant.

Run from the project root (backend must be running first):
    .venv/bin/streamlit run ui/app.py
"""

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Docs Assistant", page_icon="📚")

# --- Session state: the chat transcript survives across reruns ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: backend status + controls ---
with st.sidebar:
    st.title("📚 Docs Assistant")
    st.caption("Agentic RAG MVP — Phase 1")

    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        if health.get("status") == "ok":
            st.success(f"Backend online — {health['indexed_chunks']} chunks indexed")
        else:
            st.warning(f"Backend problem: {health}")
    except Exception:
        st.error("Backend offline. Start it with:\n\n`uvicorn backend.main:app --port 8000`")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

question = st.chat_input("Ask about Kubernetes jobs, cron jobs, or autoscaling...")

# --- Empty state: tell first-time users what this assistant knows ---
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

# --- Replay the transcript so far ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for s in message["sources"]:
                    st.markdown(f"**{s['source']}** (relevance {s['score']:.2f})")
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

# --- Handle a new question ---
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and writing an answer..."):
            try:
                # Send prior turns (role + content only) so follow-ups work.
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": question, "history": history},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                st.error(f"Could not reach the backend: {e}")
                st.stop()

        plan_info = data.get("plan", {})
        if plan_info.get("intent") == "technical":
            st.caption(f"🔎 Searched the library for: *{plan_info['search_query']}*")
        elif plan_info.get("intent") == "conversational":
            st.caption("💬 Conversational — no document search needed")
        st.markdown(data["answer"])
        if data["sources"]:
            with st.expander("📄 Sources"):
                for s in data["sources"]:
                    st.markdown(f"**{s['source']}** (relevance {s['score']:.2f})")
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
    )
