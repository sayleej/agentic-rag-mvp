"""Vector store — store and search chunk fingerprints in Qdrant Cloud."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.config import (
    EMBEDDING_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TOP_K,
)

_client = None

# 3072-dim vectors are heavy (~30 KB each as JSON), so uploads go in
# small batches — one huge request can time out mid-transfer.
UPLOAD_BATCH_SIZE = 32


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return _client


def reset_collection() -> None:
    """Delete and recreate the collection — a clean slate before ingestion."""
    client = _get_client()
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM,
            distance=models.Distance.COSINE,
        ),
    )


def add_chunks(chunks: list[str], vectors: list[list[float]], source: str) -> None:
    """Store chunks with their vectors; `source` records which file each came from."""
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, "source": source},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client = _get_client()
    for i in range(0, len(points), UPLOAD_BATCH_SIZE):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i : i + UPLOAD_BATCH_SIZE],
        )


def search(query_vector: list[float], limit: int = TOP_K) -> list[dict]:
    """Return the chunks whose vectors are closest to the query vector."""
    response = _get_client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return [
        {
            "text": point.payload.get("text", ""),
            "source": point.payload.get("source", "unknown"),
            "score": point.score,
        }
        for point in response.points
    ]


def count_chunks() -> int:
    """How many chunks are currently indexed (for status displays)."""
    return _get_client().count(QDRANT_COLLECTION).count
