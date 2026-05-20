#!/usr/bin/env python3
"""Export citation/source pairs for semantic citation precision review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.harness import (  # noqa: E402
    DEFAULT_REGISTRY,
    DEFAULT_TRANSCRIPTS_DIR,
    citation_review_queue,
)


DEFAULT_OUTPUT = ROOT / "defence_agent" / "data" / "evals" / "reports" / "citation_precision_review_queue.jsonl"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to demo/eval registry YAML.")
    parser.add_argument(
        "--transcript-dir",
        action="append",
        help="Transcript directory or JSON file to export. Can be repeated. Defaults to presentation readiness runs.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Export citation pairs from every transcript directory under defence_agent/data/transcripts.",
    )
    parser.add_argument(
        "--source-text",
        action="store_true",
        help="Include local corpus excerpts and deterministic lexical support hints.",
    )
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT), help="Output JSONL path.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    transcript_paths = args.transcript_dir
    if args.all_runs:
        transcript_paths = [str(path) for path in sorted(DEFAULT_TRANSCRIPTS_DIR.iterdir()) if path.is_dir()]
    rows = citation_review_queue(
        registry_path=args.registry,
        transcript_paths=transcript_paths,
        include_source_text=args.source_text,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} citation/source pairs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
