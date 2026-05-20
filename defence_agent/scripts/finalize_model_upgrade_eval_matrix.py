#!/usr/bin/env python3
"""Finalize a same-timestamp Command A Plus model-upgrade eval matrix."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.harness import load_registry
from defence_agent.evals.model_upgrade_comparison_report import (
    build_model_upgrade_comparison_report,
    write_model_upgrade_comparison_report,
)
from defence_agent.evals.route_comparison_report import (
    build_route_comparison_report,
    write_route_comparison_report,
)


DEFAULT_REGISTRY = ROOT / "defence_agent" / "data" / "evals" / "retrieval_reviewer_live_24_model_upgrade.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "defence_agent" / "data" / "transcripts"
DEFAULT_REPORTS_DIR = ROOT / "defence_agent" / "data" / "evals" / "reports"
ROUTES = ("simple_rag", "agentic_rag", "reviewed_agent")
MODEL_CONFIGS = {
    "current": {
        "slug": "command_a",
        "label": "Command A full",
        "model_id": "command-a-03-2025",
        "route_report_json": "command_a_model_upgrade_route_report.json",
        "route_report_md": "command_a_model_upgrade_route_report.md",
    },
    "command_a_plus": {
        "slug": "command_a_plus",
        "label": "Command A Plus full",
        "model_id": "command-a-plus-05-2026",
        "route_report_json": "command_a_plus_model_upgrade_route_report.json",
        "route_report_md": "command_a_plus_model_upgrade_route_report.md",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True, help="Timestamp segment used by run_model_upgrade_eval_matrix.py.")
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--current-label", default=MODEL_CONFIGS["current"]["label"])
    parser.add_argument("--plus-label", default=MODEL_CONFIGS["command_a_plus"]["label"])
    parser.add_argument("--current-model", default=MODEL_CONFIGS["current"]["model_id"])
    parser.add_argument("--plus-model", default=MODEL_CONFIGS["command_a_plus"]["model_id"])
    parser.add_argument("--input-usd-per-1m-tokens", type=float, default=2.5)
    parser.add_argument("--output-usd-per-1m-tokens", type=float, default=10.0)
    parser.add_argument("--allow-partial", action="store_true", help="Write reports even when case coverage is incomplete.")
    parser.add_argument("--dry-run", action="store_true", help="Print coverage and output paths without writing reports.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    registry_path = Path(args.registry)
    output_root = Path(args.output_root)
    reports_dir = Path(args.reports_dir)
    registry = load_registry(registry_path)
    expected_case_ids = _registry_case_ids(registry)
    route_dirs = _route_dirs(output_root=output_root, timestamp=args.timestamp, phase=args.phase)
    coverage = _coverage(route_dirs=route_dirs, expected_case_ids=expected_case_ids)

    print(_coverage_markdown(coverage))
    print(_planned_outputs(reports_dir))
    if args.dry_run:
        return 0
    if coverage["status"] != "complete" and not args.allow_partial:
        print(
            "Refusing to write model-upgrade reports because case coverage is incomplete. "
            "Use --allow-partial only for explicitly internal partial reports.",
            file=sys.stderr,
        )
        return 2

    reports_dir.mkdir(parents=True, exist_ok=True)
    current_json = reports_dir / str(MODEL_CONFIGS["current"]["route_report_json"])
    current_md = reports_dir / str(MODEL_CONFIGS["current"]["route_report_md"])
    plus_json = reports_dir / str(MODEL_CONFIGS["command_a_plus"]["route_report_json"])
    plus_md = reports_dir / str(MODEL_CONFIGS["command_a_plus"]["route_report_md"])
    comparison_json = reports_dir / "command_a_plus_model_upgrade_comparison_report.json"
    comparison_md = reports_dir / "command_a_plus_model_upgrade_comparison_report.md"

    current_report = build_route_comparison_report(
        registry_path=registry_path,
        simple_dir=route_dirs["current"]["simple_rag"],
        agentic_dir=route_dirs["current"]["agentic_rag"],
        reviewed_dir=route_dirs["current"]["reviewed_agent"],
        model_label=args.current_label,
        model_id=args.current_model,
        input_usd_per_1m_tokens=args.input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=args.output_usd_per_1m_tokens,
    )
    write_route_comparison_report(current_report, json_path=current_json, md_path=current_md)

    plus_report = build_route_comparison_report(
        registry_path=registry_path,
        simple_dir=route_dirs["command_a_plus"]["simple_rag"],
        agentic_dir=route_dirs["command_a_plus"]["agentic_rag"],
        reviewed_dir=route_dirs["command_a_plus"]["reviewed_agent"],
        model_label=args.plus_label,
        model_id=args.plus_model,
        input_usd_per_1m_tokens=args.input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=args.output_usd_per_1m_tokens,
    )
    write_route_comparison_report(plus_report, json_path=plus_json, md_path=plus_md)

    comparison = build_model_upgrade_comparison_report(
        current_report_path=current_json,
        plus_report_path=plus_json,
    )
    write_model_upgrade_comparison_report(comparison, json_path=comparison_json, md_path=comparison_md)

    print(f"Wrote current route report: {current_json}")
    print(f"Wrote plus route report: {plus_json}")
    print(f"Wrote comparison report: {comparison_json}")
    print(f"Presentation gate: {comparison['presentation_gate']['recommendation']}")
    print(f"Case coverage: {comparison['case_count_audit']['status']}")
    return 0


def _registry_case_ids(registry: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(case.get("id", "")).strip()
        for case in registry.get("cases", []) or []
        if isinstance(case, dict) and str(case.get("id", "")).strip()
    )


def _route_dirs(*, output_root: Path, timestamp: str, phase: str) -> dict[str, dict[str, Path]]:
    dirs: dict[str, dict[str, Path]] = {}
    for model_key, config in MODEL_CONFIGS.items():
        slug = str(config["slug"])
        dirs[model_key] = {
            route: output_root / f"model_upgrade_{timestamp}_{phase}_{slug}_{route}"
            for route in ROUTES
        }
    return dirs


def _coverage(
    *,
    route_dirs: dict[str, dict[str, Path]],
    expected_case_ids: tuple[str, ...],
) -> dict[str, Any]:
    expected = set(expected_case_ids)
    rows = []
    for model_key, routes in route_dirs.items():
        for route, directory in routes.items():
            existing = {
                path.stem
                for path in directory.glob("*.json")
                if path.is_file()
            }
            matched = expected & existing
            missing = expected - existing
            unexpected = existing - expected
            rows.append(
                {
                    "model": model_key,
                    "route": route,
                    "directory": str(directory),
                    "existing_count": len(existing),
                    "matched_expected_count": len(matched),
                    "missing_expected_count": len(missing),
                    "unexpected_count": len(unexpected),
                    "missing_preview": sorted(missing)[:5],
                    "unexpected_preview": sorted(unexpected)[:5],
                }
            )
    status = "complete" if rows and all(row["missing_expected_count"] == 0 and row["unexpected_count"] == 0 for row in rows) else "incomplete"
    return {
        "status": status,
        "expected_case_count": len(expected_case_ids),
        "rows": rows,
    }


def _coverage_markdown(coverage: dict[str, Any]) -> str:
    lines = [
        f"Coverage status: {coverage['status']}",
        f"Expected cases per route/model: {coverage['expected_case_count']}",
        "model | route | matched | missing | unexpected | directory",
        "--- | --- | ---: | ---: | ---: | ---",
    ]
    for row in coverage["rows"]:
        lines.append(
            f"{row['model']} | {row['route']} | {row['matched_expected_count']} | "
            f"{row['missing_expected_count']} | {row['unexpected_count']} | {row['directory']}"
        )
        if row["missing_preview"]:
            lines.append(f"  missing: {', '.join(row['missing_preview'])}")
        if row["unexpected_preview"]:
            lines.append(f"  unexpected: {', '.join(row['unexpected_preview'])}")
    return "\n".join(lines)


def _planned_outputs(reports_dir: Path) -> str:
    outputs = [
        reports_dir / str(MODEL_CONFIGS["current"]["route_report_json"]),
        reports_dir / str(MODEL_CONFIGS["current"]["route_report_md"]),
        reports_dir / str(MODEL_CONFIGS["command_a_plus"]["route_report_json"]),
        reports_dir / str(MODEL_CONFIGS["command_a_plus"]["route_report_md"]),
        reports_dir / "command_a_plus_model_upgrade_comparison_report.json",
        reports_dir / "command_a_plus_model_upgrade_comparison_report.md",
    ]
    return "Planned outputs:\n" + "\n".join(f"- {path}" for path in outputs)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "case"


if __name__ == "__main__":
    raise SystemExit(main())
