#!/usr/bin/env python3
"""Build a balanced 60-case pilot eval bank from registry variants and supplements."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.harness import DEFAULT_REGISTRY, TARGET_SAMPLE_SIZES  # noqa: E402
from defence_agent.evals.harness import _case_slice  # noqa: E402


DEFAULT_SUPPLEMENTS = ROOT / "defence_agent" / "data" / "evals" / "pilot_eval_supplemental_cases.yaml"
DEFAULT_OUTPUT = ROOT / "defence_agent" / "data" / "evals" / "pilot_eval_bank.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Base registry YAML.")
    parser.add_argument("--supplements", default=str(DEFAULT_SUPPLEMENTS), help="Supplemental case YAML.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output pilot bank YAML.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    base_cases = _expanded_registry_cases(Path(args.registry))
    supplemental_cases = _load_cases(Path(args.supplements))
    candidates = base_cases + supplemental_cases
    selected = _select_balanced_cases(candidates)
    output = {
        "suite": "defence_agent_pilot_eval_bank",
        "description": "Balanced 60-case Cohere-aligned pilot evaluation bank.",
        "target_sample_sizes": dict(TARGET_SAMPLE_SIZES),
        "notes": [
            "Built from demo_query_registry query variants plus pilot_eval_supplemental_cases.",
            "Run through run_demo_query_registry.py with --registry when live Cohere cost is acceptable.",
        ],
        "cases": selected,
    }
    Path(args.output).write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(f"Wrote {len(selected)} cases to {args.output}")
    for slice_name, target in TARGET_SAMPLE_SIZES.items():
        count = sum(1 for case in selected if case.get("eval_slice") == slice_name)
        print(f"{slice_name}: {count}/{target}")
    return 0


def _load_cases(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [case for case in raw.get("cases", []) or [] if isinstance(case, dict)]


def _expanded_registry_cases(path: Path) -> list[dict[str, Any]]:
    cases = _load_cases(path)
    expanded: list[dict[str, Any]] = []
    for case in cases:
        base = copy.deepcopy(case)
        base.pop("query_variants", None)
        expanded.append(base)
        for index, variant in enumerate(case.get("query_variants", []) or [], start=1):
            variant_case = copy.deepcopy(base)
            variant_case["parent_id"] = str(case.get("id", ""))
            variant_case["id"] = f"{case.get('id')}__variant_{index:02d}"
            variant_case["query"] = str(variant.get("query", "")) if isinstance(variant, dict) else str(variant)
            if isinstance(variant, dict):
                variant_case.update({key: value for key, value in variant.items() if key != "query"})
            expanded.append(variant_case)
    return expanded


def _select_balanced_cases(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for slice_name, target in TARGET_SAMPLE_SIZES.items():
        slice_cases = [
            _case_with_slice(case, slice_name)
            for case in candidates
            if _case_slice(case) == slice_name and str(case.get("id", "")) not in seen_ids
        ]
        if len(slice_cases) < target:
            raise SystemExit(f"Not enough cases for {slice_name}: {len(slice_cases)}/{target}")
        chosen = slice_cases[:target]
        selected.extend(chosen)
        seen_ids.update(str(case.get("id", "")) for case in chosen)
    return selected


def _case_with_slice(case: dict[str, Any], slice_name: str) -> dict[str, Any]:
    value = copy.deepcopy(case)
    value["eval_slice"] = slice_name
    return value


if __name__ == "__main__":
    raise SystemExit(main())
