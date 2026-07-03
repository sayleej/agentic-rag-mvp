"""Central configuration — every other module gets its settings from here."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env file that sits at the project root, regardless of
# which folder the app is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- API keys & endpoints ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# --- Models ---
EMBEDDING_MODEL = "gemini-embedding-001"   # 3072-dim vectors, tuned for retrieval
EMBEDDING_DIM = 3072
GROQ_MODEL = "openai/gpt-oss-120b"         # fast, high-quality open model on Groq

# --- Vector store ---
QDRANT_COLLECTION = "rag_mvp"

# --- Ingestion ---
DATA_DIR = PROJECT_ROOT / "data"
CHUNK_SIZE = 1500          # max characters per chunk

# --- Retrieval ---
TOP_K = 5                  # how many chunks to hand the LLM per question
CANDIDATES = 15            # how many to fetch from Qdrant before reranking


def validate() -> list[str]:
    """Return a list of missing settings (empty list = all good)."""
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    return missing
