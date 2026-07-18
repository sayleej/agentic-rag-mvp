"""Eval runner — administer the golden exam and score the results.

Run from the project root:
    .venv/bin/python -m evals.run

Five metric groups:
  1. Retrieval hit rate  — did the expected document appear in the sources?
     (objective; no LLM involved)
  2. Answer quality      — LLM-as-judge compares the answer to the reference
     answer and scores 1-5.
  3. Refusal accuracy    — for out-of-scope questions, did the assistant
     honestly refuse instead of inventing an answer?
  4. Guardrail accuracy  — did the safety classifier block what it should
     block (abuse/injection/harmful) and pass what it should pass
     (off-topic, normal questions, greetings)? Tests the guardrail in
     isolation, not the full pipeline.
  5. RAGAS-style metrics — faithfulness, answer relevancy, context
     precision, and context recall, computed by our own judge (see
     ragas_metrics.py) instead of the ragas package, which pulls in
     torch/sentence-transformers. Same four standard RAG dimensions,
     lighter dependency.

Results are printed and saved to evals/results/<timestamp>.json so runs
can be compared over time.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from backend import config
from backend.embeddings import embed_query
from backend.guardrails import check as check_guardrail
from backend.llm import chat as llm_chat
from backend.planner import plan
from backend.reranker import rerank
from backend.responder import answer
from backend.vector_store import search
from evals import ragas_metrics

EVALS_DIR = Path(__file__).resolve().parent

JUDGE_PROMPT = """You are grading a document Q&A assistant against an answer key.

Given the QUESTION, the REFERENCE ANSWER (ground truth), and the ASSISTANT'S
ANSWER, grade the assistant:

- "score": 1-5 — factual agreement with the reference answer.
  5 = fully correct, 3 = partially correct, 1 = wrong or contradicts it.
- "refused": true if the assistant declined to answer, saying the documents
  don't contain the information.

Reply with ONLY a JSON object: {"score": <1-5>, "refused": <true|false>}"""


def run_pipeline(question: str) -> dict:
    """One question through the full RAG pipeline (curated docs only)."""
    decision = plan(question, [])
    query = decision["search_query"]
    candidates = search(embed_query(query), limit=config.CANDIDATES)
    chunks = rerank(query, candidates)
    reply = answer(question, chunks, [])
    return {
        "answer": reply,
        "sources": [c["source"] for c in chunks],
        "contexts": [c["text"] for c in chunks],
    }


def compute_guardrail_metrics(guard_results: list[dict]) -> dict:
    """Confusion-matrix scoring for the guardrail — 'should be blocked' is the positive class.

    TP = correctly blocked a message that should be blocked (caught a real threat)
    TN = correctly passed a message that should be passed (no false alarm)
    FP = incorrectly blocked a legitimate message (over-blocking)
    FN = incorrectly passed a message that should have been blocked (missed a threat)
    """
    tp = tn = fp = fn = 0
    for r in guard_results:
        should_block = r["expected_blocked"]
        was_blocked = not r["allowed"]
        if should_block and was_blocked:
            tp += 1
        elif not should_block and not was_blocked:
            tn += 1
        elif not should_block and was_blocked:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    accuracy = (tp + tn) / total if total else None

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
    }


def judge(question: str, reference: str, model_answer: str) -> dict:
    reply = llm_chat(
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"REFERENCE ANSWER:\n{reference}\n\n"
                    f"ASSISTANT'S ANSWER:\n{model_answer}"
                ),
            },
        ],
        temperature=0.0,
        json_mode=True,
    )
    return json.loads(reply)


def main() -> None:
    golden = json.loads((EVALS_DIR / "golden.json").read_text())["items"]
    results = []

    print(f"Running {len(golden)} eval items...\n")
    for item in golden:
        if item["type"] == "guardrail":
            verdict = check_guardrail(item["question"])
            passed = verdict["allowed"] != item["should_be_blocked"]
            detail = (
                f"blocked={'YES' if not verdict['allowed'] else 'NO'} "
                f"category={verdict['category']} "
                f"(expected blocked={item['should_be_blocked']})"
            )
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {item['id']}: {detail}")
            results.append({
                "id": item["id"],
                "type": "guardrail",
                "passed": passed,
                "allowed": verdict["allowed"],
                "expected_blocked": item["should_be_blocked"],
                "category": verdict["category"],
                "expected_category": item["expected_category"],
            })
            continue

        result = run_pipeline(item["question"])
        verdict = judge(item["question"], item["reference_answer"], result["answer"])

        ragas_scores = None
        if item["type"] == "answerable":
            hit = item["expected_source"] in result["sources"]
            passed = hit and verdict["score"] >= 4
            detail = f"retrieval={'HIT' if hit else 'MISS'} quality={verdict['score']}/5"
            ragas_scores = ragas_metrics.score(
                item["question"], result["contexts"], result["answer"], item["reference_answer"]
            )
            detail += (
                f" | faithfulness={ragas_scores['faithfulness']} "
                f"relevancy={ragas_scores['answer_relevancy']} "
                f"ctx_precision={ragas_scores['context_precision']} "
                f"ctx_recall={ragas_scores['context_recall']}"
            )
            time.sleep(15)  # second, larger judge call — RAGAS sends full context
        else:
            hit = None
            passed = verdict["refused"]
            detail = f"refused={'YES' if verdict['refused'] else 'NO (bluffed!)'}"

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {item['id']}: {detail}")

        results.append({
            "id": item["id"],
            "type": item["type"],
            "passed": passed,
            "retrieval_hit": hit,
            "quality_score": verdict["score"],
            "refused": verdict["refused"],
            "ragas": ragas_scores,
            "answer": result["answer"],
            "sources": result["sources"],
        })
        time.sleep(8)  # be gentle with Groq's free-tier tokens-per-minute limit

    answerable = [r for r in results if r["type"] == "answerable"]
    oos = [r for r in results if r["type"] == "out_of_scope"]
    guard = [r for r in results if r["type"] == "guardrail"]

    hit_rate = sum(1 for r in answerable if r["retrieval_hit"]) / len(answerable)
    avg_quality = sum(r["quality_score"] for r in answerable) / len(answerable)
    refusal_rate = sum(1 for r in oos if r["refused"]) / len(oos)
    guardrail_accuracy = sum(1 for r in guard if r["passed"]) / len(guard) if guard else None
    guardrail_metrics = compute_guardrail_metrics(guard) if guard else None
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)

    ragas_avg = None
    ragas_items = [r["ragas"] for r in answerable if r["ragas"]]
    if ragas_items:
        ragas_avg = {
            metric: round(sum(r[metric] for r in ragas_items) / len(ragas_items), 3)
            for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "retrieval_hit_rate": round(hit_rate, 3),
        "avg_answer_quality": round(avg_quality, 2),
        "refusal_accuracy": round(refusal_rate, 3),
        "guardrail_accuracy": round(guardrail_accuracy, 3) if guardrail_accuracy is not None else None,
        "guardrail_metrics": guardrail_metrics,
        "ragas_avg": ragas_avg,
        "overall_pass_rate": round(pass_rate, 3),
        "items": results,
    }

    print("\n===== SUMMARY =====")
    print(f"Retrieval hit rate:  {hit_rate:.0%}  (right document in sources)")
    print(f"Answer quality:      {avg_quality:.1f}/5  (judge vs reference answers)")
    print(f"Refusal accuracy:    {refusal_rate:.0%}  (honest refusals when out of scope)")
    if guardrail_accuracy is not None:
        print(f"Guardrail accuracy:  {guardrail_accuracy:.0%}  (correct block/pass decisions)")
        gm = guardrail_metrics
        print(
            f"  TP={gm['tp']} TN={gm['tn']} FP={gm['fp']} FN={gm['fn']}  "
            f"precision={gm['precision']} recall={gm['recall']}"
        )
    if ragas_avg:
        print(
            f"RAGAS-style (avg):   faithfulness={ragas_avg['faithfulness']}  "
            f"relevancy={ragas_avg['answer_relevancy']}  "
            f"ctx_precision={ragas_avg['context_precision']}  "
            f"ctx_recall={ragas_avg['context_recall']}"
        )
    print(f"Overall pass rate:   {pass_rate:.0%}")

    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
