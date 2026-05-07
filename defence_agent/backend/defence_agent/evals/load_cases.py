from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from defence_agent.evals.schemas import EvalCase, EvalSuite


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = PROJECT_ROOT / "data" / "evals"

SUITE_FILES = {
    "canonical": "canonical_eval_set.yaml",
    "generated": "generated_eval_set.yaml",
    "heldout": "heldout_eval_set.yaml",
    "regression": "regression_eval_set.yaml",
    "adversarial": "adversarial_eval_set.yaml",
    "demo_candidates": "demo_candidates.yaml",
    "demo": "demo_queries.yaml",
}


def list_suites() -> dict[str, str]:
    return {name: str(EVAL_DIR / filename) for name, filename in SUITE_FILES.items() if (EVAL_DIR / filename).exists()}


def load_suite(name: str = "canonical") -> EvalSuite:
    if name not in SUITE_FILES:
        raise ValueError(f"Unknown eval suite: {name}")
    path = EVAL_DIR / SUITE_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Eval suite not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "cases" in payload:
        if name == "demo" and payload["cases"] and "query_id" not in payload["cases"][0]:
            payload = _convert_demo_queries(payload)
        return EvalSuite.model_validate(payload)
    raise ValueError(f"Invalid eval suite format: {path}")


def load_case(query_id: str, suite: str = "canonical") -> EvalCase:
    for case in load_suite(suite).cases:
        if case.query_id == query_id:
            return case
    raise KeyError(f"Case {query_id} not found in suite {suite}")


def validate_suites(names: Iterable[str] | None = None) -> dict[str, object]:
    names = list(names or [name for name in SUITE_FILES if name != "demo"])
    seen: set[str] = set()
    duplicates: list[str] = []
    counts: dict[str, int] = {}
    for name in names:
        suite = load_suite(name)
        counts[name] = len(suite.cases)
        for case in suite.cases:
            if case.query_id in seen:
                duplicates.append(case.query_id)
            seen.add(case.query_id)
    return {
        "ok": not duplicates,
        "suite_counts": counts,
        "duplicates": sorted(set(duplicates)),
        "total_cases": sum(counts.values()),
    }


def _convert_demo_queries(payload: dict) -> dict:
    cases = []
    for item in payload.get("cases", []):
        cases.append(
            {
                "query_id": item["query_id"],
                "user_query": item["user_query"],
                "task_type": "demo_candidate",
                "topic": item.get("evaluation_type", "demo"),
                "capability_tags": item.get("expected_tools", []),
                "complexity_level": "L3",
                "complexity_score": 5,
                "complexity_dimensions": ["citation_density_required"],
                "user_context": item.get("user_context", {"user_id": "alex_analyst", "role": "planning_analyst", "access_level": "public_internal", "language": "en"}),
                "expected_route": item["expected_route"],
                "expected_tools": item.get("expected_tools", []),
                "expected_sources": [{"doc_id": doc_id, "required": True} for doc_id in item.get("expected_sources", [])],
                "disallowed_sources": [],
                "expected_key_facts": [],
                "forbidden_claims": [],
                "expected_answer_behavior": item.get("expected_behavior", "answer_with_citations"),
                "expected_refusal": "deny" in str(item.get("expected_behavior", "")).lower(),
                "citation_required": bool(item.get("expected_sources")),
                "exact_expected_result": None,
                "grader_config": {"route": True, "retrieval": True, "filter": True, "answer_key_facts": False, "citation_validation": True, "llm_judge": False},
                "demo_notes": item.get("notes_for_presenter", ""),
            }
        )
    return {"suite": payload.get("suite", "demo"), "generated_by": payload.get("generated_by"), "cases": cases}
