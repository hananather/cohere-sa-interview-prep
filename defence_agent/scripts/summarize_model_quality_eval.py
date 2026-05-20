#!/usr/bin/env python3
"""Build the curated Defence Agent model-quality eval report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.model_quality_report import (  # noqa: E402
    DEFAULT_CITATION_LABELS,
    DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
    DEFAULT_MODEL_QUALITY_REGISTRY,
    DEFAULT_MODEL_QUALITY_REPORT_JSON,
    DEFAULT_MODEL_QUALITY_REPORT_MD,
    build_model_quality_report,
    write_model_quality_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_MODEL_QUALITY_REGISTRY), help="Curated eval bank YAML.")
    parser.add_argument(
        "--transcript-dir",
        default=str(DEFAULT_FULL_PILOT_TRANSCRIPT_DIR),
        help="Directory containing saved transcript JSON for the curated cases.",
    )
    parser.add_argument(
        "--citation-labels",
        default=str(DEFAULT_CITATION_LABELS),
        help="Citation support labels for semantic citation precision.",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_MODEL_QUALITY_REPORT_JSON), help="JSON report path.")
    parser.add_argument("--output-md", default=str(DEFAULT_MODEL_QUALITY_REPORT_MD), help="Markdown report path.")
    parser.add_argument(
        "--rerank-uplift",
        action="store_true",
        help="Run live retrieval diagnostics to compare BM25, embeddings, and rerank.",
    )
    parser.add_argument(
        "--reviewer-run-dir",
        help="Reviewer challenge run directory. Defaults to the latest saved reviewer run.",
    )
    parser.add_argument(
        "--no-citation-support",
        action="store_true",
        help="Skip deterministic lexical citation support scoring.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_model_quality_report(
        registry_path=Path(args.registry),
        transcript_dir=Path(args.transcript_dir),
        citation_label_path=Path(args.citation_labels) if args.citation_labels else None,
        include_citation_support=not args.no_citation_support,
        include_rerank_uplift=args.rerank_uplift,
        reviewer_run_dir=Path(args.reviewer_run_dir) if args.reviewer_run_dir else None,
    )
    write_model_quality_report(report, json_path=Path(args.output_json), md_path=Path(args.output_md))
    print(report["markdown"])
    print(f"\nWrote JSON report: {args.output_json}")
    print(f"Wrote Markdown report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
