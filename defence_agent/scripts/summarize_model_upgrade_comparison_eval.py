#!/usr/bin/env python3
"""Build the Command A Plus model-upgrade comparison report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.model_upgrade_comparison_report import (
    DEFAULT_COMPARISON_JSON,
    DEFAULT_COMPARISON_MD,
    DEFAULT_CURRENT_REPORT_JSON,
    DEFAULT_PLUS_REPORT_JSON,
    build_model_upgrade_comparison_report,
    write_model_upgrade_comparison_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-report", default=str(DEFAULT_CURRENT_REPORT_JSON))
    parser.add_argument("--plus-report", default=str(DEFAULT_PLUS_REPORT_JSON))
    parser.add_argument("--json-output", default=str(DEFAULT_COMPARISON_JSON))
    parser.add_argument("--md-output", default=str(DEFAULT_COMPARISON_MD))
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_model_upgrade_comparison_report(
        current_report_path=Path(args.current_report),
        plus_report_path=Path(args.plus_report),
    )
    write_model_upgrade_comparison_report(
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
