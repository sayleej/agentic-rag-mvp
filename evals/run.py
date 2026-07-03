"""Eval runner — administer the golden exam and score the results.

Run from the project root:
    .venv/bin/python -m evals.run

Three metrics:
  1. Retrieval hit rate  — did the expected document appear in the sources?
     (objective; no LLM involved)
  2. Answer quality      — LLM-as-judge compares the answer to the reference
     answer and scores 1-5.
  3. Refusal accuracy    — for out-of-scope questions, did the assistant
     honestly refuse instead of inventing an answer?

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
from backend.llm import chat as llm_chat
from backend.planner import plan
from backend.reranker import rerank
from backend.responder import answer
from backend.vector_store import search

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
    return {"answer": reply, "sources": [c["source"] for c in chunks]}


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
        result = run_pipeline(item["question"])
        verdict = judge(item["question"], item["reference_answer"], result["answer"])

        if item["type"] == "answerable":
            hit = item["expected_source"] in result["sources"]
            passed = hit and verdict["score"] >= 4
            detail = f"retrieval={'HIT' if hit else 'MISS'} quality={verdict['score']}/5"
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
            "answer": result["answer"],
            "sources": result["sources"],
        })
        time.sleep(1)  # be gentle with rate limits

    answerable = [r for r in results if r["type"] == "answerable"]
    oos = [r for r in results if r["type"] == "out_of_scope"]

    hit_rate = sum(1 for r in answerable if r["retrieval_hit"]) / len(answerable)
    avg_quality = sum(r["quality_score"] for r in answerable) / len(answerable)
    refusal_rate = sum(1 for r in oos if r["refused"]) / len(oos)
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "retrieval_hit_rate": round(hit_rate, 3),
        "avg_answer_quality": round(avg_quality, 2),
        "refusal_accuracy": round(refusal_rate, 3),
        "overall_pass_rate": round(pass_rate, 3),
        "items": results,
    }

    print("\n===== SUMMARY =====")
    print(f"Retrieval hit rate:  {hit_rate:.0%}  (right document in sources)")
    print(f"Answer quality:      {avg_quality:.1f}/5  (judge vs reference answers)")
    print(f"Refusal accuracy:    {refusal_rate:.0%}  (honest refusals when out of scope)")
    print(f"Overall pass rate:   {pass_rate:.0%}")

    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
