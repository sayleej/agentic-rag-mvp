"""Chunker — cut long text into paragraph-aware pieces.

We split on blank lines (paragraph breaks) and pack whole paragraphs
into chunks of up to CHUNK_SIZE characters. Keeping paragraphs intact
means each chunk stays a coherent thought, which makes retrieval and
answering more accurate than cutting at arbitrary character positions.
"""

from __future__ import annotations

from backend.config import CHUNK_SIZE


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # A single paragraph longer than the limit gets hard-split.
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(para), chunk_size):
                chunks.append(para[i : i + chunk_size].strip())
            continue

        # Would adding this paragraph overflow the current chunk?
        if len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            current = para + "\n\n"
        else:
            current += para + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks
