#!/usr/bin/env python3
"""Build the agent-value route comparison report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.route_comparison_report import (
    DEFAULT_AGENTIC_DIR,
    DEFAULT_REGISTRY,
    DEFAULT_REPORT_JSON,
    DEFAULT_REPORT_MD,
    DEFAULT_REVIEWED_DIR,
    DEFAULT_SIMPLE_DIR,
    build_route_comparison_report,
    write_route_comparison_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--simple-dir", default=str(DEFAULT_SIMPLE_DIR))
    parser.add_argument("--agentic-dir", default=str(DEFAULT_AGENTIC_DIR))
    parser.add_argument("--reviewed-dir", default=str(DEFAULT_REVIEWED_DIR))
    parser.add_argument("--json-output", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-output", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--model-label", default="Command A")
    parser.add_argument("--model-id", default="command-a-03-2025")
    parser.add_argument("--input-usd-per-1m-tokens", type=float, default=2.5)
    parser.add_argument("--output-usd-per-1m-tokens", type=float, default=10.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_route_comparison_report(
        registry_path=Path(args.registry),
        simple_dir=Path(args.simple_dir),
        agentic_dir=Path(args.agentic_dir),
        reviewed_dir=Path(args.reviewed_dir),
        model_label=args.model_label,
        model_id=args.model_id,
        input_usd_per_1m_tokens=args.input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=args.output_usd_per_1m_tokens,
    )
    write_route_comparison_report(
        report,
        json_path=Path(args.json_output),
        md_path=Path(args.md_output),
    )
    print(report["markdown"])
    print(f"Wrote JSON: {args.json_output}")
    print(f"Wrote Markdown: {args.md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
