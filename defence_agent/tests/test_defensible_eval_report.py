from __future__ import annotations

from defence_agent.evals.defensible_eval_report import (
    DEFAULT_AMBIGUOUS_REGISTRY,
    DEFAULT_DEFENSIBLE_REGISTRY,
    DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
    build_defensible_eval_report,
    registry_sha256,
)
from defence_agent.evals.harness import load_registry


def test_defensible_eval_report_uses_balanced_60_case_bank() -> None:
    report = build_defensible_eval_report(
        registry_path=DEFAULT_DEFENSIBLE_REGISTRY,
        ambiguous_registry_path=DEFAULT_AMBIGUOUS_REGISTRY,
        transcript_dir=DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
    )

    design = report["sample_design"]
    assert design["scored_case_count"] == 60
    assert set(design["scored_slice_counts"].values()) == {6}
    assert design["answerable_case_count"] > 30
    assert design["unanswerable_case_count"] > 0
    assert design["hard_common_case_count"] >= 24
    assert design["ambiguous_clarification_case_count"] == 6
    assert "15-case" in report["markdown"]


def test_defensible_eval_report_tracks_operational_estimates() -> None:
    report = build_defensible_eval_report(
        registry_path=DEFAULT_DEFENSIBLE_REGISTRY,
        ambiguous_registry_path=DEFAULT_AMBIGUOUS_REGISTRY,
        transcript_dir=DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
    )

    telemetry = report["operational_metrics"]["telemetry_coverage"]
    cost = report["operational_metrics"]["overall"]["estimated_generation_cost_usd"]

    assert telemetry["estimated_token_rows"] == 60
    assert telemetry["measured_latency_rows"] == 0
    assert cost["mean"] is not None
    assert cost["total"] is not None


def test_registry_hash_is_stable_for_case_contract() -> None:
    registry = load_registry(DEFAULT_DEFENSIBLE_REGISTRY)

    assert registry_sha256(registry) == registry_sha256(load_registry(DEFAULT_DEFENSIBLE_REGISTRY))
