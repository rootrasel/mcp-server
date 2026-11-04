#!/usr/bin/env python3
"""CLI utility to ingest female wellness documents into ChromaDB.

Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --dry-run
    python scripts/ingest_documents.py --path knowledge_base/raw/
"""

import argparse
import sys
from pathlib import Path

# Add src to pythonpath if executing as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from female_wellness.core.logging_config import setup_logging
from female_wellness.rag.pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest female wellness knowledge base documents into ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=str,
        default="knowledge_base/raw",
        help="Root directory path containing raw documents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate loading and chunking without generating embeddings or modifying database.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(log_level=args.log_level)

    print("=" * 70)
    print(" Female Wellness MCP - Document Ingestion CLI")
    print(f" Target Directory: {args.path}")
    print(f" Dry Run Mode:     {'ENABLED (No vectors written)' if args.dry_run else 'DISABLED (Live indexing)'}")
    print("=" * 70)

    try:
        pipeline = RAGPipeline()
        summary = pipeline.ingest_directory(args.path, dry_run=args.dry_run)
    except Exception as exc:
        print(f"\n[FATAL ERROR] Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print("\n" + "-" * 70)
    print(" Ingestion Summary")
    print("-" * 70)
    print(f" Total files discovered:  {summary['scanned_files']}")
    print(f" Successfully processed:  {summary['successful_files']}")
    print(f" Failed files:            {summary['failed_files']}")
    print(f" Total chunks {'planned' if args.dry_run else 'indexed'}:     {summary['total_chunks']}")
    print("-" * 70)

    if summary["details"]:
        print("\nDocument Breakdown:")
        for item in summary["details"]:
            source = item.get("source", "Unknown")
            stage = item.get("life_stage", "Unknown")
            chunks = item.get("total_chunks") or item.get("chunks_planned", 0)
            status = item.get("status", "ok")
            print(f"  • {source:<45} | Stage: {stage:<22} | Chunks: {chunks:<3} | [{status}]")

    if summary["errors"]:
        print("\nIngestion Failures:")
        for err in summary["errors"]:
            print(f"  ✖ {err['file']}: {err['error']}")
        return 1

    print("\n✓ Ingestion process completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

