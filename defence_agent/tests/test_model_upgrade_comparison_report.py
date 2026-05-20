from __future__ import annotations

import json
from pathlib import Path

from defence_agent.evals.defensible_eval_report import registry_sha256
from defence_agent.evals.harness import load_registry
from defence_agent.evals.model_upgrade_comparison_report import build_model_upgrade_comparison_report
from defence_agent.evals.route_comparison_report import _distribution


def test_model_upgrade_report_compares_routes_and_groups(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
cases:
  - id: canonical_case
    eval_slice: easy_direct
    query: Canonical?
  - id: challenge_case
    eval_slice: model_upgrade_bilingual
    model_upgrade_group: challenge_4
    query: Challenge?
""",
        encoding="utf-8",
    )
    registry = load_registry(registry_path)
    registry_hash = registry_sha256(registry)
    current_report = _route_report(
        registry_path=registry_path,
        registry_hash=registry_hash,
        model_label="Command A",
        model_id="command-a-03-2025",
        run_prefix="current",
        simple_passes=[True, False],
        reviewed_failed_released=1,
    )
    plus_report = _route_report(
        registry_path=registry_path,
        registry_hash=registry_hash,
        model_label="Command A Plus",
        model_id="command-a-plus-05-2026",
        run_prefix="plus",
        simple_passes=[True, True],
        reviewed_failed_released=0,
    )
    current_path = tmp_path / "current.json"
    plus_path = tmp_path / "plus.json"
    current_path.write_text(json.dumps(current_report), encoding="utf-8")
    plus_path.write_text(json.dumps(plus_report), encoding="utf-8")

    report = build_model_upgrade_comparison_report(
        current_report_path=current_path,
        plus_report_path=plus_path,
    )

    simple_combined = [
        item
        for item in report["deltas"]
        if item["group"] == "combined_24" and item["route"] == "simple_rag"
    ][0]
    assert report["sample_design"]["case_count"] == 2
    assert simple_combined["end_to_end_success"] == 0.5
    assert report["presentation_gate"]["recommendation"] == "headline_candidate"
    assert "Command A Plus Model-Upgrade Eval Report" in report["markdown"]


def test_latency_p95_uses_upper_tail_for_small_samples() -> None:
    dist = _distribution([3.0, 15.0])

    assert dist["p95"] == 15.0


def _route_report(
    *,
    registry_path: Path,
    registry_hash: str,
    model_label: str,
    model_id: str,
    run_prefix: str,
    simple_passes: list[bool],
    reviewed_failed_released: int,
) -> dict:
    route_dirs = {
        "simple_rag": f"/tmp/{run_prefix}_simple",
        "agentic_rag": f"/tmp/{run_prefix}_agentic",
        "reviewed_agent": f"/tmp/{run_prefix}_reviewed",
    }
    rows = []
    for route, path in route_dirs.items():
        run_name = Path(path).name
        for case_id, passed in zip(["canonical_case", "challenge_case"], simple_passes, strict=True):
            rows.append(
                {
                    "case_id": case_id,
                    "transcript_run": run_name,
                    "passed": passed,
                    "end_to_end_success": 1.0 if passed else 0.0,
                    "answerability_accuracy": 1.0,
                    "retrieval_recall": 1.0 if passed else 0.0,
                    "retrieval_precision": 0.5,
                    "citation_recall": 1.0 if passed else 0.0,
                    "citation_doc_recall": 1.0 if passed else 0.0,
                    "citation_doc_precision": 1.0,
                    "citation_source_precision_proxy": 1.0,
                    "citation_support_precision_lexical": 1.0,
                    "generation_fact_recall": None,
                    "forbidden_fact_absence": None,
                }
            )
    return {
        "schema_version": "route_comparison_report.v1",
        "model_label": model_label,
        "model_id": model_id,
        "registry_path": str(registry_path),
        "registry_sha256": registry_hash,
        "route_dirs": route_dirs,
        "orchestration_contract": {
            route: {
                "status": "complete",
                "harness_id": "defence_agent.run_turn.v1",
                "route_implementation_id": (
                    "direct_search_pages_plus_grounded_generation.v1"
                    if route == "simple_rag"
                    else "adk_litellm_planner_plus_grounded_generation.v1"
                ),
                "chunk_strategy": "page",
                "retrieval_mode": "hybrid",
            }
            for route in route_dirs
        },
        "base_harness_report": {"rows": rows},
        "operational_metrics": {
            route: {
                "n": 2,
                "latency_seconds": {"mean": 1.0, "median": 1.0, "p95": 1.0},
                "billed_input_tokens": 10,
                "billed_output_tokens": 5,
                "captured_generation_cost_mean_usd": 0.001,
                "captured_generation_cost_total_usd": 0.002,
            }
            for route in route_dirs
        },
        "reviewer_metrics": {
            "case_count": 2,
            "release_gate_rate": 0.5,
            "human_gate_rate": 0.5,
            "failed_cases_human_gated": 1,
            "failed_cases_released": reviewed_failed_released,
            "reviewer_estimated_citation_precision": 1.0,
            "reviewed_citation_coverage": 1.0,
        },
    }
