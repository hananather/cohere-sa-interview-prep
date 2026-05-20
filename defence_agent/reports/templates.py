"""Scheduled report tasks for the Reports demo tier."""

from __future__ import annotations

from defence_agent.auth.context import DEFAULT_PERSONA_ID
from defence_agent.reports.models import ReportTask, ResearchSectionTask


_TASKS: dict[str, ReportTask] = {
    "weekly_northern_readiness_brief": ReportTask(
        id="weekly_northern_readiness_brief",
        title="Weekly Northern Readiness And Doctrine Brief",
        cadence="weekly",
        schedule_label="Mondays 06:00",
        objective=(
            "Produce a cited weekly staff brief that combines defence policy, allied doctrine, "
            "and AI-readiness guidance for central planning staff."
        ),
        audience="Chief of Staff and central planning staff",
        requested_by="chief_of_staff_demo",
        persona_id=DEFAULT_PERSONA_ID,
        next_run="2026-05-25T06:00:00-04:00",
        required_terms=("north", "NATO", "AI", "readiness"),
        expected_doc_ids=(
            "CA-DEF-POL-2024-EN",
            "NATO-STRAT-CONCEPT-2022-EN",
            "CA-AI-STRAT-2024-EN",
        ),
        sections=(
            ResearchSectionTask(
                id="northern_posture",
                title="Northern posture and sovereignty",
                objective="Find the strongest approved evidence about Canada's northern defence posture.",
                query=(
                    "For a weekly central planning brief, summarize Canada's approved northern defence "
                    "posture, Arctic sovereignty priorities, and readiness implications."
                ),
                output_guidance="Focus on concrete posture, investment, readiness, and planning implications.",
                expected_doc_ids=("CA-DEF-POL-2024-EN",),
            ),
            ResearchSectionTask(
                id="allied_alignment",
                title="Allied and NATO alignment",
                objective="Find NATO alignment points that matter for a Canadian defence planning brief.",
                query=(
                    "For a weekly central planning brief, identify NATO strategic concept themes that align "
                    "with Canadian northern defence and collective defence planning."
                ),
                output_guidance="Focus on allied planning themes, deterrence, resilience, and interoperability.",
                expected_doc_ids=("NATO-STRAT-CONCEPT-2022-EN", "CA-DEF-POL-2024-EN"),
            ),
            ResearchSectionTask(
                id="ai_readiness",
                title="AI-enabled readiness and governance",
                objective="Find AI strategy evidence relevant to operational readiness and responsible adoption.",
                query=(
                    "For a weekly central planning brief, summarize DND CAF AI strategy guidance relevant "
                    "to readiness, responsible AI adoption, data, and governance."
                ),
                output_guidance="Focus on actionable implications for staff workflows and adoption risk.",
                expected_doc_ids=("CA-AI-STRAT-2024-EN",),
            ),
        ),
    ),
    "monthly_doctrine_gap_review": ReportTask(
        id="monthly_doctrine_gap_review",
        title="Monthly Doctrine Evidence Gap Review",
        cadence="monthly",
        schedule_label="First Monday 07:00",
        objective=(
            "Produce a cited monthly review of where the approved corpus supports planning questions "
            "and where the assistant should abstain or request more evidence."
        ),
        audience="Knowledge management lead and central planning staff",
        requested_by="knowledge_manager_demo",
        persona_id=DEFAULT_PERSONA_ID,
        next_run="2026-06-01T07:00:00-04:00",
        required_terms=("evidence", "gap", "approved", "sources"),
        expected_doc_ids=(
            "CA-DEF-POL-2024-EN",
            "CA-AI-STRAT-2024-EN",
            "UK-MOD-ASOEM-2023-EN",
        ),
        sections=(
            ResearchSectionTask(
                id="policy_coverage",
                title="Policy coverage",
                objective="Identify where approved policy sources support recurring planning questions.",
                query=(
                    "For a monthly evidence-gap review, identify the strongest approved policy sources "
                    "for central defence planning questions and what they support."
                ),
                output_guidance="Focus on what the agent can answer from approved sources.",
                expected_doc_ids=("CA-DEF-POL-2024-EN", "CA-AI-STRAT-2024-EN"),
            ),
            ResearchSectionTask(
                id="format_coverage",
                title="Format and normalization coverage",
                objective="Show that normalized page evidence can support DOCX-origin or scanned-manual workflows.",
                query=(
                    "For a monthly evidence-gap review, summarize what the approved corpus shows about "
                    "DOCX-origin or scanned manual content represented in the normalized evidence pipeline."
                ),
                output_guidance="Focus on normalized retrieval evidence, not native DOCX parsing claims.",
                expected_doc_ids=("UK-MOD-ASOEM-2023-EN", "US-ARMY-FM30-16-1972-SCAN"),
            ),
        ),
    ),
}


def list_report_tasks() -> list[ReportTask]:
    """Return report tasks in stable presentation order."""

    return [_TASKS[key] for key in sorted(_TASKS)]


def get_report_task(task_id: str) -> ReportTask:
    """Return a scheduled report task by id."""

    try:
        return _TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown report task: {task_id}") from exc
