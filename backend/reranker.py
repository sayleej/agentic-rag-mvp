"""Reranker — precise second-stage sorting of search candidates.

Stage 1 (Qdrant) compares pre-computed embeddings: fast but approximate.
Stage 2 (this module) uses FlashRank's cross-encoder, which reads the
query and each candidate TOGETHER — slower per document but far more
precise, so we only run it on the shortlist. It's a small local model
running on the CPU: no API, no key, no per-question cost.

Fails safe: if the model can't load or errors, we return the candidates
in their original vector-score order.
"""

from __future__ import annotations

from backend.config import TOP_K

_ranker = None
_ranker_failed = False


def _get_ranker():
    global _ranker, _ranker_failed
    if _ranker is None and not _ranker_failed:
        try:
            from pathlib import Path

            from flashrank import Ranker

            # Small (~34 MB) cross-encoder, downloaded once and cached.
            # Cache lives in the home directory, NOT /tmp — the OS clears
            # /tmp, which leaves a broken half-cache that fails to load.
            cache = Path.home() / ".cache" / "flashrank"
            cache.mkdir(parents=True, exist_ok=True)
            _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(cache))
        except Exception as e:
            print(f"FlashRank unavailable ({e}) — falling back to vector order.")
            _ranker_failed = True
    return _ranker


def rerank(query: str, chunks: list[dict], top_n: int = TOP_K) -> list[dict]:
    """Re-sort candidate chunks by cross-encoder relevance; keep top_n."""
    if len(chunks) <= top_n:
        return chunks

    ranker = _get_ranker()
    if ranker is None:
        return chunks[:top_n]

    try:
        from flashrank import RerankRequest

        passages = [
            {"id": i, "text": c["text"], "meta": c} for i, c in enumerate(chunks)
        ]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        reranked = []
        for r in results[:top_n]:
            chunk = dict(r["meta"])
            # Keep both scales: cosine similarity from Qdrant (~0.7 = good)
            # and the cross-encoder's confidence (can legitimately reach 1.0).
            chunk["vector_score"] = chunk.get("score")
            chunk["rerank_score"] = float(r["score"])
            chunk["score"] = chunk["rerank_score"]
            reranked.append(chunk)
        return reranked
    except Exception as e:
        print(f"Reranking failed ({e}) — falling back to vector order.")
        return chunks[:top_n]
