"""Ingestion script — load every document in data/ into Qdrant.

Run from the project root:
    .venv/bin/python -m backend.ingest

Re-running wipes and rebuilds the whole index, so the library always
matches exactly what's in the data/ folder.
"""

from __future__ import annotations

import sys

from backend import config
from backend.chunker import chunk_text
from backend.embeddings import embed_documents
from backend.loaders import load_file
from backend.vector_store import add_chunks, count_chunks, reset_collection


def main() -> None:
    missing = config.validate()
    if missing:
        print(f"Missing settings in .env: {', '.join(missing)}")
        sys.exit(1)

    files = sorted(p for p in config.DATA_DIR.iterdir() if p.is_file())
    if not files:
        print(f"No files found in {config.DATA_DIR}. Drop some documents there first.")
        sys.exit(1)

    print(f"Resetting collection '{config.QDRANT_COLLECTION}'...")
    reset_collection()

    total_chunks = 0
    for path in files:
        text = load_file(path)
        if text is None:
            print(f"  SKIP {path.name} (unsupported file type)")
            continue
        if not text.strip():
            print(f"  SKIP {path.name} (no text could be extracted)")
            continue

        chunks = chunk_text(text)
        print(f"  {path.name}: {len(text)} chars -> {len(chunks)} chunks, embedding...")
        vectors = embed_documents(chunks)
        add_chunks(chunks, vectors, source=path.name)
        total_chunks += len(chunks)

    print(f"\nDone. {total_chunks} chunks indexed; Qdrant now holds {count_chunks()}.")


if __name__ == "__main__":
    main()
