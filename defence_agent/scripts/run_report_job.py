#!/usr/bin/env python3
"""Run one live scheduled report-agent job and write report artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.reports.orchestrator import REPORT_ROOT, run_report_task
from defence_agent.reports.templates import list_report_tasks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="weekly_northern_readiness_brief",
        choices=[task.id for task in list_report_tasks()],
        help="Scheduled report task to run.",
    )
    parser.add_argument("--persona", help="Persona id. Defaults to the task persona.")
    parser.add_argument("--job-id", help="Stable job id. Defaults to task id plus UTC timestamp.")
    parser.add_argument("--out-root", type=Path, default=REPORT_ROOT, help="Root directory for generated reports.")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Also run the live Cohere LLM-as-judge evaluator.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable run summary JSON.")
    return parser


async def _main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = await run_report_task(
        args.task,
        persona_id=args.persona,
        out_root=args.out_root,
        job_id=args.job_id,
        run_llm_judge=args.llm_judge,
    )
    summary = {
        "job_id": result.job_id,
        "task_id": result.task.id,
        "artifact_dir": str(result.artifact_dir),
        "researcher_count": len(result.findings),
        "completed_researchers": sum(1 for finding in result.findings if finding.status == "completed"),
        "source_count": len(result.synthesis.sources),
        "citation_count": len(result.synthesis.citations),
        "reviewer_status": result.critic_report.get("status"),
        "reviewer_action": result.critic_report.get("release_gate"),
        "eval_overall_score": result.eval_summary.get("overall_score"),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print(f"Report job: {summary['job_id']}")
    print(f"Task: {summary['task_id']}")
    print(f"Artifacts: {summary['artifact_dir']}")
    print(
        "Researchers: "
        f"{summary['completed_researchers']}/{summary['researcher_count']} completed; "
        f"sources={summary['source_count']}; citations={summary['citation_count']}"
    )
    print(f"Reviewer: {summary['reviewer_status']} / {summary['reviewer_action']}")
    print(f"Eval score: {summary['eval_overall_score']}")


if __name__ == "__main__":
    asyncio.run(_main())
