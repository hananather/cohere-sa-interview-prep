from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.retrieval.chroma_index import build_index
from defence_agent.retrieval.document_pages import load_document_pages


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or refresh the real Cohere Chroma index.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the whole collection.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable build metadata.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    pages = load_document_pages()
    result = build_index(force=args.force)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Parsed pages: {len(pages)}")
        print(f"Chroma collection: {result['collection']}")
        print(f"Embedding backend: {result['embedding_backend']}")
        print(f"Embedding dimension: {result['embedding_dimension']}")
        print(f"Chunk strategy: {result.get('chunk_strategy', 'page')}")
        print(f"Indexed chunks: {result['indexed_chunks']}")
        print(f"Retrieval chunks: {result.get('retrieval_chunks', result['indexed_chunks'])}")
        print(f"Embedded chunks this run: {result.get('embedded_chunks', result['embedded_pages'])}")
        print(f"Deleted orphan chunks: {result.get('orphaned_chunks_deleted', result['orphaned_pages_deleted'])}")
        print(f"Skipped: {result['skipped']}")


if __name__ == "__main__":
    main()
