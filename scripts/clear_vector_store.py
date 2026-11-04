#!/usr/bin/env python3
"""CLI utility to clear or reset the ChromaDB vector store collection.

Usage:
    python scripts/clear_vector_store.py
    python scripts/clear_vector_store.py --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from female_wellness.core.logging_config import setup_logging
from female_wellness.rag.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wipe and reset the ChromaDB vector collection.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute without interactive user confirmation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(log_level="WARNING")

    store = ChromaVectorStore()
    stats = store.get_collection_stats()

    print("=" * 60)
    print(" ChromaDB Vector Store Reset Utility")
    print(f" Target Collection: {stats.collection_name}")
    print(f" Current Chunks:    {stats.total_chunks}")
    print(f" Current Documents: {stats.total_documents}")
    print("=" * 60)

    if stats.total_chunks == 0:
        print("Vector store is already empty.")
        return 0

    if not args.force:
        confirm = input("\nAre you sure you want to delete ALL indexed vectors? (yes/no): ").strip().lower()
        if confirm not in ("yes", "y"):
            print("Operation aborted by user.")
            return 0

    try:
        store.clear_collection()
        print("\n✓ Collection wiped and successfully re-initialized.")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Failed to clear collection: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

