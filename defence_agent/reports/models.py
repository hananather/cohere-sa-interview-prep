"""Data models for background report-generation agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchSectionTask:
    """One bounded research assignment delegated to a researcher agent."""

    id: str
    title: str
    objective: str
    query: str
    output_guidance: str
    expected_doc_ids: tuple[str, ...] = ()
    top_k: int = 6
    language: str = "en"


@dataclass(frozen=True)
class ReportTask:
    """Scheduled report task definition."""

    id: str
    title: str
    cadence: str
    schedule_label: str
    objective: str
    audience: str
    requested_by: str
    persona_id: str
    sections: tuple[ResearchSectionTask, ...]
    required_terms: tuple[str, ...] = ()
    expected_doc_ids: tuple[str, ...] = ()
    next_run: str = ""


@dataclass(frozen=True)
class ResearchFinding:
    """Output from one researcher agent."""

    section_id: str
    title: str
    query: str
    answer: str
    raw_answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    search_audit: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    status: str = "completed"
    error: str = ""


@dataclass(frozen=True)
class ReportSynthesis:
    """Final report generated from researcher findings."""

    markdown: str
    raw_markdown: str
    citations: list[dict[str, Any]]
    citation_validation: dict[str, Any]
    sources: list[dict[str, Any]]
    model: str
    content_blocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ReportRunResult:
    """Completed report run and generated artifacts."""

    job_id: str
    task: ReportTask
    persona_id: str
    status: str
    started_at: str
    completed_at: str
    findings: list[ResearchFinding]
    synthesis: ReportSynthesis
    critic_report: dict[str, Any]
    eval_summary: dict[str, Any]
    artifact_dir: Path
    artifact_paths: dict[str, Path]
    run_trace: dict[str, Any]
