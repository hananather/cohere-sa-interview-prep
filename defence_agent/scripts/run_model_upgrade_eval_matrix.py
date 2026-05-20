#!/usr/bin/env python3
"""Run the Command A Plus model-upgrade eval matrix."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.harness import load_registry

DEFAULT_REGISTRY = ROOT / "defence_agent" / "data" / "evals" / "retrieval_reviewer_live_24_model_upgrade.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "defence_agent" / "data" / "transcripts"
ROUTES = ("simple_rag", "agentic_rag", "reviewed_agent")
SMOKE_CASES = ("easy_ai_strategy_lines_of_effort", "model_upgrade_unanswerable_superseded_ai_directive")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--current-model", default="command-a-03-2025")
    parser.add_argument("--plus-model", default="command-a-plus-05-2026")
    parser.add_argument("--reviewer-model", default="command-a-reasoning-08-2025")
    parser.add_argument(
        "--smoke-models",
        choices=("plus", "current", "all"),
        default="plus",
        help="Which model configs to run during the smoke phase.",
    )
    parser.add_argument(
        "--models",
        choices=("all", "current", "plus"),
        help="Which model configs to run. Overrides --smoke-models when set.",
    )
    parser.add_argument(
        "--route",
        action="append",
        choices=ROUTES,
        help="Run only one route. Can be repeated. Defaults to all routes.",
    )
    parser.add_argument("--case-id", action="append", help="Run only one case id. Can be repeated.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-case JSON files in each output directory instead of rerunning them.",
    )
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--max-review-cycles", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true", help="Continue matrix and exit zero even if a case fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix commands without running live calls.")
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Print existing/missing transcript counts for the selected matrix without running live calls.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.phase == "full":
        print(
            "Comparability guard: this script reruns both models through the same current "
            "route implementation. Do not mix these outputs with older saved baselines."
        )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_configs = _selected_models(args)
    routes = tuple(args.route or ROUTES)
    case_ids = tuple(args.case_id or (SMOKE_CASES if args.phase == "smoke" else ()))
    planned_case_ids = case_ids or _registry_case_ids(Path(args.registry))
    if args.coverage_only:
        _print_coverage(
            output_root=output_root,
            timestamp=args.timestamp,
            phase=args.phase,
            model_configs=model_configs,
            routes=routes,
            case_ids=planned_case_ids,
        )
        return 0

    failures = 0
    for model_slug, model_label, model_id in model_configs:
        for route in routes:
            output_dir = output_root / f"model_upgrade_{args.timestamp}_{args.phase}_{model_slug}_{route}"
            command = [
                sys.executable,
                str(ROOT / "defence_agent" / "scripts" / "run_demo_query_registry.py"),
                "--registry",
                str(args.registry),
                "--run-mode",
                route,
                "--output-dir",
                str(output_dir),
                "--delay-seconds",
                str(args.delay_seconds),
                "--max-review-cycles",
                str(args.max_review_cycles),
            ]
            if args.no_fail:
                command.append("--no-fail")
            if args.skip_existing:
                command.append("--skip-existing")
            for case_id in case_ids:
                command.extend(["--case-id", case_id])
            env = _model_env(model_id=model_id, reviewer_model=args.reviewer_model)
            print(f"\n## {args.phase}: {model_label} / {route}")
            print(f"output_dir={output_dir}")
            if args.dry_run:
                print(" ".join(_quote(part) for part in command))
                continue
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode != 0:
                failures += 1
                if not args.no_fail:
                    return result.returncode
    return 0 if args.no_fail or failures == 0 else 1


def _registry_case_ids(registry_path: Path) -> tuple[str, ...]:
    registry = load_registry(registry_path)
    return tuple(
        str(case.get("id", "")).strip()
        for case in registry.get("cases", []) or []
        if isinstance(case, dict) and str(case.get("id", "")).strip()
    )


def _print_coverage(
    *,
    output_root: Path,
    timestamp: str,
    phase: str,
    model_configs: tuple[tuple[str, str, str], ...],
    routes: tuple[str, ...],
    case_ids: tuple[str, ...],
) -> None:
    print("model | route | existing | missing | output_dir")
    print("-" * 100)
    for model_slug, model_label, _model_id in model_configs:
        for route in routes:
            output_dir = output_root / f"model_upgrade_{timestamp}_{phase}_{model_slug}_{route}"
            existing = [
                case_id
                for case_id in case_ids
                if (output_dir / f"{_safe_filename(case_id)}.json").exists()
            ]
            existing_set = set(existing)
            missing = [case_id for case_id in case_ids if case_id not in existing_set]
            print(f"{model_label} | {route} | {len(existing)} | {len(missing)} | {output_dir}")
            if missing:
                preview = ", ".join(missing[:5])
                suffix = " ..." if len(missing) > 5 else ""
                print(f"  missing: {preview}{suffix}")


def _selected_models(args: argparse.Namespace) -> tuple[tuple[str, str, str], ...]:
    configs = {
        "current": ("command_a", "Command A", args.current_model),
        "plus": ("command_a_plus", "Command A Plus", args.plus_model),
    }
    selector = args.models
    if selector is None and args.phase == "smoke":
        selector = args.smoke_models
    if selector is None:
        selector = "all"
    if selector == "current":
        return (configs["current"],)
    if selector == "plus":
        return (configs["plus"],)
    return (configs["current"], configs["plus"])


def _quote(value: object) -> str:
    text = str(value)
    if not text or any(char.isspace() for char in text):
        return repr(text)
    return text


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "case"


def _model_env(*, model_id: str, reviewer_model: str) -> dict[str, str]:
    env = dict(os.environ)
    env["COHERE_CHAT_MODEL"] = model_id
    env["DEFTECH_ADK_MODEL"] = f"cohere/{model_id}"
    env["DEFTECH_ADK_CRITIC_MODEL"] = reviewer_model
    env["DEFTECH_ADK_REVIEWER_MODEL"] = reviewer_model
    env["DEFENCE_AGENT_CHUNK_STRATEGY"] = "page"
    env["COHERE_REQUESTS_PER_MINUTE"] = "0"
    return env


if __name__ == "__main__":
    raise SystemExit(main())
