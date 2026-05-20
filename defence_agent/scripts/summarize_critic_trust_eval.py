#!/usr/bin/env python3
"""Run the Reviewer Agent citation trust eval and write report artifacts."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.critic_trust_report import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_REPORT_JSON,
    DEFAULT_REPORT_MD,
    build_critic_trust_report,
    write_critic_trust_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to critic trust YAML dataset.")
    parser.add_argument("--case-id", action="append", help="Run only one case id. Can be repeated.")
    parser.add_argument("--max-cases", type=int, help="Run the first N selected cases.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Reviewer trust threshold.")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Pause between live reviewer calls.")
    parser.add_argument("--output-json", default=str(DEFAULT_REPORT_JSON), help="Output JSON report path.")
    parser.add_argument("--output-md", default=str(DEFAULT_REPORT_MD), help="Output Markdown report path.")
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    report = await build_critic_trust_report(
        dataset_path=Path(args.dataset),
        case_ids=args.case_id,
        max_cases=args.max_cases,
        threshold=args.threshold,
        delay_seconds=args.delay_seconds,
    )
    write_critic_trust_report(report, json_path=Path(args.output_json), md_path=Path(args.output_md))
    metrics = report["metrics"]
    print(f"Wrote critic trust eval report: {args.output_md}")
    print(f"cases={report['case_count']} citation_labels={report['citation_label_count']}")
    print(f"invalid_citation_recall={metrics.get('invalid_citation_recall')}")
    print(f"trusted_citation_precision={metrics.get('trusted_citation_precision')}")
    print(f"release_gate_accuracy={metrics.get('release_gate_accuracy')}")
    print(f"unsafe_release_rate={metrics.get('unsafe_release_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
