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
    # Qdrant requires indexes on fields used in search filters.
    for field in ("source", "source_type"):
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def add_chunks(
    chunks: list[str],
    vectors: list[list[float]],
    source: str,
    source_type: str = "true",
) -> None:
    """Store chunks with their vectors; `source` records which file each came
    from and `source_type` labels it as curated ("true") or noise ("noisy")."""
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, "source": source, "source_type": source_type},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client = _get_client()
    for i in range(0, len(points), UPLOAD_BATCH_SIZE):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i : i + UPLOAD_BATCH_SIZE],
        )


def search(
    query_vector: list[float],
    limit: int = TOP_K,
    include_noisy: bool = False,
) -> list[dict]:
    """Return the chunks whose vectors are closest to the query vector.

    By default only curated ("true") documents are searched; pass
    include_noisy=True to search the whole library.
    """
    query_filter = None
    if not include_noisy:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value="true"),
                )
            ]
        )
    response = _get_client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
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
