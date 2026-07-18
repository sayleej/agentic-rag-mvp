"""Combined-mode app for Streamlit Community Cloud.

Runs the whole RAG pipeline inside one Streamlit process — no separate
FastAPI backend. Used only for cloud hosting; local development keeps
the two-app setup (backend/main.py + ui/app.py).

Deploy: share.streamlit.io -> New app -> this repo, main branch,
streamlit_app.py. Paste API keys into the app's Secrets settings.
"""

import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

# On Streamlit Cloud, keys live in st.secrets (not a .env file).
# Copy them into environment variables BEFORE importing backend modules,
# because backend/config.py reads the environment at import time.
for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from backend import config
from backend.graph import rag_agent
from backend.guardrails import BLOCKED_MESSAGE, TOO_LONG_MESSAGE, check
from backend.vector_store import count_chunks

st.set_page_config(page_title="Docs Assistant", page_icon="📚")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def render_sources(sources):
    with st.expander("📄 Sources"):
        for s in sources:
            if s.get("rerank_score") is not None:
                label = f"vector {s['vector_score']:.2f} · rerank {s['rerank_score']:.2f}"
            else:
                label = f"vector {s['score']:.2f}"
            st.markdown(f"**{s['source']}** ({label})")
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

    include_noisy = st.toggle(
        "Include noise documents",
        value=False,
        help="The index deliberately contains off-topic 'noise' documents "
        "(85% of all chunks) to demonstrate retrieval precision. Off = search "
        "only the curated Kubernetes docs.",
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

chat_tab, analytics_tab = st.tabs(["💬 Chat", "📊 Analytics"])

with analytics_tab:
    st.subheader("Evaluation Results")
    st.caption(
        "Read-only view of this project's eval suite — a hand-written golden dataset "
        "scored with an LLM judge and the real `ragas` library. Live eval runs happen "
        "privately (they cost real API calls); this tab just displays saved results."
    )

    EVALS_DIR = Path(__file__).resolve().parent / "evals"

    golden_path = EVALS_DIR / "golden.json"
    if golden_path.exists():
        golden = json.loads(golden_path.read_text())["items"]
        answerable = [i for i in golden if i["type"] == "answerable"]
        oos = [i for i in golden if i["type"] == "out_of_scope"]
        guard = [i for i in golden if i["type"] == "guardrail"]

        with st.expander(f"📋 Golden dataset — {len(golden)} items "
                          f"({len(answerable)} answerable, {len(oos)} out-of-scope, {len(guard)} guardrail)"):
            st.markdown("**Answerable**")
            st.dataframe(
                pd.DataFrame([{"ID": i["id"], "Question": i["question"]} for i in answerable]),
                use_container_width=True, hide_index=True,
            )
            st.markdown("**Out-of-scope (should be refused)**")
            st.dataframe(
                pd.DataFrame([{"ID": i["id"], "Question": i["question"]} for i in oos]),
                use_container_width=True, hide_index=True,
            )
            st.markdown("**Guardrail (safety classifier tests)**")
            st.dataframe(
                pd.DataFrame([
                    {"ID": i["id"], "Input": i["question"],
                     "Expected": "Block" if i["should_be_blocked"] else "Pass"}
                    for i in guard
                ]),
                use_container_width=True, hide_index=True,
            )

    results_dir = EVALS_DIR / "results"
    saved_runs = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    if saved_runs:
        latest = json.loads(saved_runs[-1].read_text())
        st.markdown(f"**Latest run — {latest.get('timestamp', saved_runs[-1].stem)}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval Hit Rate", f"{latest.get('retrieval_hit_rate', 0):.0%}")
        c2.metric("Answer Quality", f"{latest.get('avg_answer_quality', 0):.1f}/5")
        c3.metric("Refusal Accuracy", f"{latest.get('refusal_accuracy', 0):.0%}")
        if latest.get("guardrail_accuracy") is not None:
            c4.metric("Guardrail Accuracy", f"{latest['guardrail_accuracy']:.0%}")

        gm = latest.get("guardrail_metrics")
        if gm:
            g1, g2, g3 = st.columns(3)
            g1.metric("Guardrail Precision", gm.get("precision", "—"))
            g2.metric("Guardrail Recall", gm.get("recall", "—"))
            g3.metric("TP / TN / FP / FN", f"{gm.get('tp')} / {gm.get('tn')} / {gm.get('fp')} / {gm.get('fn')}")

            with st.expander("ℹ️ What do TP / TN / FP / FN mean here?"):
                st.markdown(
                    "The guardrail's job is to block hostile input (abuse, prompt injection, "
                    "harmful requests) while letting everything else — including off-topic but "
                    "harmless questions — pass through. \"Should be blocked\" is treated as the "
                    "positive case:\n\n"
                    "- **TP (True Positive)** — a message that *should* be blocked, *was* blocked. "
                    "The guardrail caught a real threat.\n"
                    "- **TN (True Negative)** — a message that *should* pass, *did* pass. "
                    "No false alarm.\n"
                    "- **FP (False Positive)** — a legitimate message got blocked by mistake. "
                    "This is *over-blocking* — annoying, but not dangerous.\n"
                    "- **FN (False Negative)** — a hostile message *slipped through*. "
                    "This is the dangerous failure mode — a missed threat.\n\n"
                    "**Precision** = of everything blocked, how much was a real threat "
                    "(TP / (TP+FP)) — low precision means the guardrail is too trigger-happy.\n\n"
                    "**Recall** = of everything that should have been blocked, how much was "
                    "actually caught (TP / (TP+FN)) — low recall means real threats are getting "
                    "through, which is the more serious failure mode for a safety system.\n\n"
                    "A single \"accuracy\" number can hide which of these two failure modes is "
                    "happening — that's why precision and recall are tracked separately."
                )

        if latest.get("ragas_avg"):
            st.markdown("**RAGAS metrics** (faithfulness, relevancy, context precision/recall)")
            r1, r2, r3, r4 = st.columns(4)
            ra = latest["ragas_avg"]
            r1.metric("Faithfulness", ra.get("faithfulness", "—"))
            r2.metric("Answer Relevancy", ra.get("answer_relevancy", "—"))
            r3.metric("Context Precision", ra.get("context_precision", "—"))
            r4.metric("Context Recall", ra.get("context_recall", "—"))

        st.divider()
        st.markdown("**History across all saved runs**")
        history_rows = []
        for path in saved_runs:
            data = json.loads(path.read_text())
            row = {
                "Run": path.stem,
                "Retrieval Hit": data.get("retrieval_hit_rate"),
                "Answer Quality": data.get("avg_answer_quality"),
                "Refusal Acc.": data.get("refusal_accuracy"),
                "Guardrail Acc.": data.get("guardrail_accuracy"),
                "Overall Pass": data.get("overall_pass_rate"),
            }
            if data.get("ragas_avg"):
                row["Faithfulness"] = data["ragas_avg"].get("faithfulness")
            history_rows.append(row)
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No saved eval runs found yet.")

with chat_tab:
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
            with st.spinner("Thinking..."):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    verdict = check(question)
                    if not verdict["allowed"]:
                        decision = {"intent": "blocked", "search_query": None}
                        chunks = []
                        reply = TOO_LONG_MESSAGE if verdict["category"] == "too_long" else BLOCKED_MESSAGE
                        steps = [f"Guardrails: blocked ({verdict['category']}) — pipeline stopped"]
                    else:
                        initial_state = {
                            "question": question,
                            "history": history,
                            "include_noisy": include_noisy,
                            "intent": "",
                            "search_query": "",
                            "chunks": [],
                            "answer": "",
                            "steps": ["Guardrails: passed"],
                        }
                        graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        final_state = rag_agent.invoke(initial_state, config=graph_config)
                        decision = {
                            "intent": final_state["intent"],
                            "search_query": final_state["search_query"] or None,
                        }
                        chunks = final_state.get("chunks", [])
                        reply = final_state["answer"]
                        steps = final_state["steps"]
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.stop()

            if decision["intent"] == "technical":
                st.caption(f"🔎 Searched the library for: *{decision['search_query']}*")
            elif decision["intent"] == "blocked":
                st.caption("🛡️ Blocked by guardrails")
            else:
                st.caption("💬 Conversational — no document search needed")
            if steps:
                with st.expander("⚙️ Thought process"):
                    for step in steps:
                        st.write(f"• {step}")
            st.markdown(reply)
            if chunks:
                render_sources(chunks)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "sources": chunks}
        )
