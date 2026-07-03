"""Embeddings — turn text into vectors ("fingerprints") using Gemini.

Two entry points, because Gemini tunes the vector differently depending
on which side of the search the text is on:
  - embed_documents(): for chunks going INTO the library (RETRIEVAL_DOCUMENT)
  - embed_query():     for a user question searching the library (RETRIEVAL_QUERY)
"""

from __future__ import annotations

import time

from google import genai
from google.genai import types

from backend.config import EMBEDDING_MODEL, GEMINI_API_KEY

# Gemini accepts up to 100 texts per API call; we stay comfortably under.
BATCH_SIZE = 50

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# The free tier allows 100 embedding requests per minute, so a big
# document can exhaust it mid-ingestion. Waits must be long enough to
# let the per-minute window reset.
RETRY_WAITS = [20, 30, 60, 60, 60]


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts, retrying with backoff on rate limits."""
    client = _get_client()
    for attempt, wait in enumerate(RETRY_WAITS):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            message = str(e).lower()
            rate_limited = any(s in message for s in ("429", "rate", "quota", "exhausted"))
            if rate_limited and attempt < len(RETRY_WAITS) - 1:
                print(f"  Gemini rate limit — waiting {wait}s for the quota window to reset "
                      f"(attempt {attempt + 1}/{len(RETRY_WAITS)})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini rate limit persisted after retries.")


def embed_documents(chunks: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing, in batches."""
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        vectors.extend(_embed(chunks[i : i + BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a user question for searching."""
    return _embed([question], "RETRIEVAL_QUERY")[0]
