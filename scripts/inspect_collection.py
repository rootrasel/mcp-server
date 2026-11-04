#!/usr/bin/env python3
"""CLI utility to inspect collection statistics and indexed chunks in ChromaDB.

Usage:
    python scripts/inspect_collection.py
    python scripts/inspect_collection.py --limit 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from female_wellness.core.logging_config import setup_logging
from female_wellness.rag.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ChromaDB vector store statistics and sample records.")
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of sample chunks to display.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(log_level="WARNING")

    store = ChromaVectorStore()
    stats = store.get_collection_stats()

    print("=" * 65)
    print(" ChromaDB Collection Inspection")
    print("=" * 65)
    print(f" Collection Name:       {stats.collection_name}")
    print(f" Total Indexed Chunks:  {stats.total_chunks}")
    print(f" Unique Documents:      {stats.total_documents}")
    print(f" Available Life Stages: {', '.join(stats.available_life_stages) or 'None'}")
    print(f" Available Categories:  {', '.join(stats.available_categories) or 'None'}")
    print("=" * 65)

    if stats.total_chunks > 0 and args.limit > 0:
        print(f"\nSample Chunks (Showing up to {args.limit}):\n")
        sample_results = store._collection.get(
            limit=args.limit,
            include=["documents", "metadatas"],
        )
        ids = sample_results.get("ids", [])
        docs = sample_results.get("documents", [])
        metas = sample_results.get("metadatas", [])

        for idx, (cid, doc, meta) in enumerate(zip(ids, docs, metas, strict=False), 1):
            source = meta.get("source", "N/A")
            stage = meta.get("life_stage", "N/A")
            cat = meta.get("category", "N/A")
            page = meta.get("page_number", "N/A")
            snippet = doc[:160].replace("\n", " ").strip() + "..."
            print(f"[{idx}] ID: {cid} | Source: {source} (Page {page})")
            print(f"    Stage: {stage} | Category: {cat}")
            print(f"    Snippet: \"{snippet}\"\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

