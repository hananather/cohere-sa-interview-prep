from __future__ import annotations

from defence_agent.ui.report_results import (
    load_report_artifacts,
    report_eval_rows,
    report_evidence_rows,
    report_status_rows,
    scheduled_template_rows,
)


def test_saved_report_artifact_loads_without_backend_or_api_call() -> None:
    artifacts = load_report_artifacts()

    assert artifacts
    artifact = artifacts[0]
    assert artifact.job_id == "demo_weekly_northern_readiness_brief"
    assert artifact.status == "completed"
    assert artifact.cadence == "weekly"
    assert artifact.source_count == 3
    assert artifact.claim_count == 3
    assert artifact.audit["model_calls_required_for_ui"] is False
    assert "Weekly Northern Readiness And Doctrine Brief" in artifact.markdown
    assert "da-report" in artifact.html


def test_report_rows_keep_schedule_evidence_and_eval_visible() -> None:
    artifact = load_report_artifacts()[0]

    status = report_status_rows(artifact)
    evidence = report_evidence_rows(artifact)
    eval_rows = report_eval_rows(artifact)
    templates = scheduled_template_rows(artifact)

    assert {"field": "cadence", "value": "weekly"} in status
    assert any(row["doc_id"] == "CA-DEF-POL-2024-EN" for row in evidence)
    assert any(row["check"] == "access_safety" and row["status"] == "pass" for row in eval_rows)
    assert any(row["cadence"] == "monthly" for row in templates)
