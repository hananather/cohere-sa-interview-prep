#!/usr/bin/env python3
"""Evaluate saved report-agent artifacts without regenerating them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.reports.eval_harness import evaluate_report_payload
from defence_agent.reports.templates import get_report_task


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path, help="Directory containing report.md and artifact JSON files.")
    parser.add_argument("--task", default="weekly_northern_readiness_brief", help="Task id for success criteria.")
    parser.add_argument("--llm-judge", action="store_true", help="Run live Cohere LLM-as-judge evaluator.")
    parser.add_argument("--write", action="store_true", help="Overwrite report_eval.json in the artifact directory.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    task = get_report_task(args.task)
    markdown = (args.artifact_dir / "report.md").read_text(encoding="utf-8")
    evidence = json.loads((args.artifact_dir / "evidence_bundle.json").read_text(encoding="utf-8"))
    audit = json.loads((args.artifact_dir / "report_audit.json").read_text(encoding="utf-8"))
    result = evaluate_report_payload(
        task=task,
        markdown=markdown,
        sources=evidence.get("sources", []),
        findings=evidence.get("researcher_findings", []),
        citations=[{"text": item.get("claim", ""), "sources": item.get("sources", [])} for item in evidence.get("claim_map", [])],
        critic_report=audit.get("critic", {}),
        run_llm_judge=args.llm_judge,
    )
    if args.write:
        (args.artifact_dir / "report_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
