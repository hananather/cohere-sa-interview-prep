#!/usr/bin/env python3
"""Summarize layered offline eval metrics from saved Defence Agent transcripts."""

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
    markdown_report,
    summarize_transcript_runs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to demo/eval registry YAML.")
    parser.add_argument(
        "--transcript-dir",
        action="append",
        help="Transcript directory or JSON file to score. Can be repeated. Defaults to presentation readiness runs.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Score every transcript directory under defence_agent/data/transcripts.",
    )
    parser.add_argument("--output-json", help="Write the full scored report as JSON.")
    parser.add_argument("--output-md", help="Write the Markdown report.")
    parser.add_argument(
        "--citation-support",
        action="store_true",
        help="Reload corpus page text and compute deterministic lexical citation support precision.",
    )
    parser.add_argument(
        "--citation-labels",
        help="YAML file with human or calibrated-judge citation/source support labels.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    transcript_paths = args.transcript_dir
    if args.all_runs:
        transcript_paths = [str(path) for path in sorted(DEFAULT_TRANSCRIPTS_DIR.iterdir()) if path.is_dir()]
    report = summarize_transcript_runs(
        registry_path=args.registry,
        transcript_paths=transcript_paths,
        include_citation_support=args.citation_support,
        citation_label_path=args.citation_labels,
    )
    rendered = markdown_report(report)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
