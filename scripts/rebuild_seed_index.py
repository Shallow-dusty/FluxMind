#!/usr/bin/env python3
"""Rebuild the local FAISS index from the bundled seed-paper library."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PAPERS_LIBRARY_DIR  # noqa: E402
from src.ingestion import discover_pdfs, rebuild_vector_store_from_pdfs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild FluxMind's local seed-paper FAISS index.")
    parser.add_argument("--require-count", type=int, default=52, help="Minimum required library PDF count.")
    args = parser.parse_args()

    paths = [path for path in discover_pdfs() if PAPERS_LIBRARY_DIR in path.parents]
    print(f"library_pdf_count={len(paths)}")
    if len(paths) < args.require_count:
        print(f"required_count={args.require_count}")
        return 1
    _store, chunk_count = rebuild_vector_store_from_pdfs(paths)
    print(f"rebuilt_chunks={chunk_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
