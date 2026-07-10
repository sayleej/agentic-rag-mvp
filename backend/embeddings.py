"""Embeddings — turn text into vectors ("fingerprints") using Gemini.

Two entry points, because Gemini tunes the vector differently depending
on which side of the search the text is on:
  - embed_documents(): for chunks going INTO the library (RETRIEVAL_DOCUMENT)
  - embed_query():     for a user question searching the library (RETRIEVAL_QUERY)
"""

from __future__ import annotations

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from backend.config import EMBEDDING_MODEL, GEMINI_API_KEY

# Gemini accepts up to 100 texts per API call; we stay comfortably under.
BATCH_SIZE = 50

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(s in message for s in ("429", "rate", "quota", "exhausted"))


# The free tier allows 100 embedding requests per minute, so a big
# document can exhaust it mid-ingestion. The wait is fixed at 30s —
# long enough for the per-minute quota window to actually reset, unlike
# naive exponential backoff, which keeps re-hitting the same closed window.
@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_fixed(30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts, retrying with a fixed wait on rate limits."""
    client = _get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in response.embeddings]


def embed_documents(chunks: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing, in batches."""
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        vectors.extend(_embed(chunks[i : i + BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a user question for searching."""
    return _embed([question], "RETRIEVAL_QUERY")[0]
