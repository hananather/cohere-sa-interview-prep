#!/usr/bin/env python3
"""Build the client-presentable Defence Agent eval report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.defensible_eval_report import (  # noqa: E402
    DEFAULT_AMBIGUOUS_REGISTRY,
    DEFAULT_DEFENSIBLE_REGISTRY,
    DEFAULT_DEFENSIBLE_REPORT_JSON,
    DEFAULT_DEFENSIBLE_REPORT_MD,
    DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
    DEFAULT_REVIEWER_RUN_DIR,
    build_defensible_eval_report,
    write_defensible_eval_report,
)
from defence_agent.evals.model_quality_report import DEFAULT_CITATION_LABELS  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_DEFENSIBLE_REGISTRY), help="Balanced pilot eval bank YAML.")
    parser.add_argument(
        "--ambiguous-registry",
        default=str(DEFAULT_AMBIGUOUS_REGISTRY),
        help="Ambiguous clarification eval bank YAML.",
    )
    parser.add_argument(
        "--transcript-dir",
        default=str(DEFAULT_FULL_PILOT_TRANSCRIPT_DIR),
        help="Directory containing saved transcript JSON for the scored pilot bank.",
    )
    parser.add_argument(
        "--citation-labels",
        default=str(DEFAULT_CITATION_LABELS),
        help="Citation support labels for semantic citation precision.",
    )
    parser.add_argument(
        "--reviewer-run-dir",
        default=str(DEFAULT_REVIEWER_RUN_DIR),
        help="Reviewer challenge run directory with measured timestamps.",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_DEFENSIBLE_REPORT_JSON), help="JSON report path.")
    parser.add_argument("--output-md", default=str(DEFAULT_DEFENSIBLE_REPORT_MD), help="Markdown report path.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_defensible_eval_report(
        registry_path=Path(args.registry),
        ambiguous_registry_path=Path(args.ambiguous_registry),
        transcript_dir=Path(args.transcript_dir),
        citation_label_path=Path(args.citation_labels) if args.citation_labels else None,
        reviewer_run_dir=Path(args.reviewer_run_dir),
    )
    write_defensible_eval_report(report, json_path=Path(args.output_json), md_path=Path(args.output_md))
    print(report["markdown"])
    print(f"\nWrote JSON report: {args.output_json}")
    print(f"Wrote Markdown report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
