"""Interactive eval dashboard — a Streamlit UI over evals/run.py's pipeline.

Run from the project root:
    streamlit run evals/app.py

Three tabs:
  1. Golden Dataset — inspect the fixed exam (answerable / out-of-scope / guardrail items)
  2. Run Live Eval  — run every item through the real pipeline, with progress feedback
  3. Results        — scored metrics for the run, plus history across saved runs

This is a UI wrapper — all scoring logic still lives in evals/run.py and
evals/ragas_metrics.py, so the CLI (`python -m evals.run`) and this dashboard
always agree on numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from evals.run import EVALS_DIR, compute_guardrail_metrics, judge, run_pipeline
from evals import ragas_metrics
from backend.guardrails import check as check_guardrail

st.set_page_config(page_title="Agentic RAG — Eval Suite", page_icon="🧪", layout="wide")

GRADE_COLORS = {"green": "#d4edda", "yellow": "#fff3cd", "red": "#f8d7da"}


def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    elif score >= 0.5:
        return "⚠️ Fair"
    return "❌ Poor"


def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"


golden = json.loads((EVALS_DIR / "golden.json").read_text())["items"]
answerable_items = [i for i in golden if i["type"] == "answerable"]
oos_items = [i for i in golden if i["type"] == "out_of_scope"]
guard_items = [i for i in golden if i["type"] == "guardrail"]

for key in ("run_results", "guard_results"):
    if key not in st.session_state:
        st.session_state[key] = None

st.title("🧪 Agentic RAG — Evaluation Suite")
st.caption("Golden dataset → live pipeline run → RAGAS + guardrail scoring")
st.divider()

tab1, tab2, tab3 = st.tabs(["📋 Golden Dataset", "🚀 Run Live Eval", "📊 Results"])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Golden Dataset
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Answerable Questions")
    st.dataframe(
        pd.DataFrame([
            {"ID": i["id"], "Question": i["question"], "Expected Source": i["expected_source"]}
            for i in answerable_items
        ]),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"{len(answerable_items)} answerable items — judged against a hand-written reference answer.")

    st.divider()
    st.subheader("Out-of-Scope Questions (should be honestly refused)")
    st.dataframe(
        pd.DataFrame([{"ID": i["id"], "Question": i["question"]} for i in oos_items]),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"{len(oos_items)} trap questions — correct when the assistant refuses instead of bluffing.")

    st.divider()
    st.subheader("Guardrail Test Cases")
    st.dataframe(
        pd.DataFrame([
            {
                "ID": i["id"],
                "Input": i["question"],
                "Expected": "🛡️ Block" if i["should_be_blocked"] else "✅ Pass",
                "Category": i["expected_category"],
            }
            for i in guard_items
        ]),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"{len(guard_items)} guardrail items — tests the safety classifier in isolation, before the pipeline runs.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Run Live Eval
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Guardrail Tests (fast — no rate limits, run these freely)")
    if st.button("▶️ Run Guardrail Tests", type="primary"):
        progress = st.progress(0, text="Testing guardrail...")
        g_results = []
        for i, item in enumerate(guard_items):
            verdict = check_guardrail(item["question"])
            g_results.append({
                "id": item["id"],
                "allowed": verdict["allowed"],
                "expected_blocked": item["should_be_blocked"],
                "category": verdict["category"],
                "expected_category": item["expected_category"],
            })
            progress.progress(int((i + 1) / len(guard_items) * 100), text=f"[{i+1}/{len(guard_items)}] {item['id']}")
        st.session_state.guard_results = g_results
        progress.progress(100, text="✅ Guardrail tests complete!")
        st.rerun()

    if st.session_state.guard_results:
        gm = compute_guardrail_metrics(st.session_state.guard_results)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{gm['accuracy']:.0%}" if gm["accuracy"] is not None else "—")
        c2.metric("Precision", gm["precision"] if gm["precision"] is not None else "—")
        c3.metric("Recall", gm["recall"] if gm["recall"] is not None else "—")
        c4.metric("TP / TN / FP / FN", f"{gm['tp']} / {gm['tn']} / {gm['fp']} / {gm['fn']}")
        st.dataframe(pd.DataFrame(st.session_state.guard_results), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Full Pipeline Run (slow — real LLM + RAGAS calls, rate-limited)")
    st.info(
        "Runs every answerable + out-of-scope question through the real retrieval/response "
        "pipeline, judges the answer, and scores it with RAGAS. This is the same thing "
        "`python -m evals.run` does — expect several minutes due to Groq's free-tier rate limits.",
        icon="⚠️",
    )
    n_items = st.slider("Number of answerable items to run (fewer = faster)", 1, len(answerable_items), 3)

    if st.button("▶️ Run Live Pipeline"):
        run_items = answerable_items[:n_items] + oos_items
        progress = st.progress(0, text="Starting...")
        table_slot = st.empty()
        rows = []
        results = []
        for i, item in enumerate(run_items):
            progress.progress(int(i / len(run_items) * 100), text=f"[{i+1}/{len(run_items)}] {item['id']}")
            result = run_pipeline(item["question"])
            verdict = judge(item["question"], item["reference_answer"], result["answer"])

            ragas_scores = None
            if item["type"] == "answerable":
                hit = item["expected_source"] in result["sources"]
                passed = hit and verdict["score"] >= 4
                ragas_scores = ragas_metrics.score(
                    item["question"], result["contexts"], result["answer"], item["reference_answer"]
                )
            else:
                hit = None
                passed = verdict["refused"]

            results.append({
                "id": item["id"], "type": item["type"], "passed": passed,
                "retrieval_hit": hit, "quality_score": verdict["score"],
                "refused": verdict["refused"], "ragas": ragas_scores,
                "answer": result["answer"], "sources": result["sources"],
            })
            rows.append({
                "#": i + 1, "ID": item["id"], "Status": "✅" if passed else "❌",
                "Quality": f"{verdict['score']}/5",
            })
            table_slot.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.session_state.run_results = results
        progress.progress(100, text="✅ Pipeline run complete!")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Results
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    if st.session_state.run_results:
        results = st.session_state.run_results
        answerable = [r for r in results if r["type"] == "answerable"]
        oos = [r for r in results if r["type"] == "out_of_scope"]

        st.subheader("This Run")
        c1, c2, c3 = st.columns(3)
        if answerable:
            hit_rate = sum(1 for r in answerable if r["retrieval_hit"]) / len(answerable)
            avg_quality = sum(r["quality_score"] for r in answerable) / len(answerable)
            c1.metric("Retrieval Hit Rate", f"{hit_rate:.0%}")
            c2.metric("Avg Answer Quality", f"{avg_quality:.1f}/5")
        if oos:
            refusal_rate = sum(1 for r in oos if r["refused"]) / len(oos)
            c3.metric("Refusal Accuracy", f"{refusal_rate:.0%}")

        ragas_items = [r["ragas"] for r in answerable if r["ragas"]]
        if ragas_items:
            st.markdown("**RAGAS metrics**")
            ragas_avg = {
                m: sum(r[m] for r in ragas_items) / len(ragas_items)
                for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
            }
            cols = st.columns(4)
            for col, (name, score) in zip(cols, ragas_avg.items()):
                col.metric(f"{_badge(score)} {name.replace('_', ' ').title()}", f"{score:.2f}", _grade(score))

        st.dataframe(pd.DataFrame(results)[["id", "type", "passed", "retrieval_hit", "quality_score"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Run the live pipeline in the previous tab to see results here.")

    st.divider()
    st.subheader("History — Saved Runs")
    results_dir = EVALS_DIR / "results"
    saved_runs = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    if saved_runs:
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
            gm = data.get("guardrail_metrics")
            if gm:
                row["Guardrail Precision"] = gm.get("precision")
                row["Guardrail Recall"] = gm.get("recall")
            if data.get("ragas_avg"):
                row["Faithfulness"] = data["ragas_avg"].get("faithfulness")
            history_rows.append(row)
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No saved runs yet — run `python -m evals.run` from the project root to generate one.")
