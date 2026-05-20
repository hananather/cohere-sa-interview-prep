"""Saved report artifacts for the Reports demo tab.

The Reports surface is intentionally artifact-first. It shows the scheduled
background-agent story without forcing a live Cohere call during the demo.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(__file__).resolve().parents[1] / "data" / "reports" / "generated"


@dataclass(frozen=True)
class ReportArtifact:
    job_id: str
    title: str
    status: str
    schedule_label: str
    cadence: str
    persona_id: str
    requested_by: str
    completed_at: str
    next_run: str
    artifact_dir: Path
    markdown: str
    html: str
    audit: dict[str, Any]
    evidence_bundle: dict[str, Any]
    eval_summary: dict[str, Any]

    @property
    def option_label(self) -> str:
        return f"{self.title} ({self.schedule_label})"

    @property
    def source_count(self) -> int:
        return len(self.evidence_bundle.get("sources", []) or [])

    @property
    def claim_count(self) -> int:
        return len(self.evidence_bundle.get("claim_map", []) or [])

    @property
    def overall_score(self) -> str:
        score = self.eval_summary.get("overall_score")
        if isinstance(score, (int, float)):
            return f"{score:.2f}"
        return str(score or "n/a")


def load_report_artifacts(root: Path = REPORT_ROOT) -> list[ReportArtifact]:
    """Load completed demo report artifacts from disk."""

    if not root.exists():
        return []
    artifacts: list[ReportArtifact] = []
    for artifact_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            artifacts.append(load_report_artifact(artifact_dir))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return artifacts


def load_report_artifact(artifact_dir: Path) -> ReportArtifact:
    audit = _read_json(artifact_dir / "report_audit.json")
    evidence_bundle = _read_json(artifact_dir / "evidence_bundle.json")
    eval_summary = _read_json(artifact_dir / "report_eval.json")
    return ReportArtifact(
        job_id=str(audit["job_id"]),
        title=str(audit["title"]),
        status=str(audit.get("status", "completed")),
        schedule_label=str(audit.get("schedule_label", "")),
        cadence=str(audit.get("cadence", "")),
        persona_id=str(audit.get("persona_id", "")),
        requested_by=str(audit.get("requested_by", "")),
        completed_at=str(audit.get("completed_at", "")),
        next_run=str(audit.get("next_run", "")),
        artifact_dir=artifact_dir,
        markdown=(artifact_dir / "report.md").read_text(encoding="utf-8"),
        html=(artifact_dir / "report.html").read_text(encoding="utf-8"),
        audit=audit,
        evidence_bundle=evidence_bundle,
        eval_summary=eval_summary,
    )


def report_status_rows(artifact: ReportArtifact) -> list[dict[str, object]]:
    return [
        {"field": "job_id", "value": artifact.job_id},
        {"field": "status", "value": artifact.status},
        {"field": "cadence", "value": artifact.cadence},
        {"field": "schedule", "value": artifact.schedule_label},
        {"field": "persona", "value": artifact.persona_id},
        {"field": "requested_by", "value": artifact.requested_by},
        {"field": "completed_at", "value": artifact.completed_at},
        {"field": "next_run", "value": artifact.next_run},
    ]


def report_evidence_rows(artifact: ReportArtifact) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in artifact.evidence_bundle.get("sources", []) or []:
        if not isinstance(source, dict):
            continue
        rows.append(
            {
                "source_id": source.get("source_id", ""),
                "doc_id": source.get("doc_id", ""),
                "title": source.get("title", ""),
                "page": source.get("page", ""),
                "access": source.get("access_level", ""),
                "why_used": source.get("why_used", ""),
            }
        )
    return rows


def report_eval_rows(artifact: ReportArtifact) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    checks = artifact.eval_summary.get("checks", {})
    if not isinstance(checks, dict):
        return rows
    for name, payload in checks.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "check": name,
                "status": payload.get("status", ""),
                "score": payload.get("score", ""),
                "note": payload.get("note", ""),
            }
        )
    return rows


def scheduled_template_rows(artifact: ReportArtifact) -> list[dict[str, object]]:
    templates = artifact.audit.get("scheduled_templates", [])
    rows: list[dict[str, object]] = []
    for template in templates if isinstance(templates, list) else []:
        if not isinstance(template, dict):
            continue
        rows.append(
            {
                "template": template.get("name", ""),
                "cadence": template.get("cadence", ""),
                "owner": template.get("owner", ""),
                "purpose": template.get("purpose", ""),
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload
