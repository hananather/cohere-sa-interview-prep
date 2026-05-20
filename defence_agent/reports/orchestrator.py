"""Live background report-generation pipeline.

The Streamlit Reports tab stays artifact-first. This module is the real backend
that a scheduler or CLI can invoke to create those artifacts with Cohere model
calls, parallel researcher sections, a synthesis pass, and a reviewer score.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from defence_agent.config import get_settings
from defence_agent.critic import review_answer
from defence_agent.grounding import (
    _cohere_documents,
    _insert_inline_markers,
    _message_content_blocks,
    _message_text,
    _normalize_citations,
    _response_citations,
    _validate_native_citations,
    finalize_answer,
)
from defence_agent.index import search_pages
from defence_agent.reports.eval_harness import evaluate_report_payload
from defence_agent.reports.models import (
    ReportRunResult,
    ReportSynthesis,
    ReportTask,
    ResearchFinding,
    ResearchSectionTask,
)
from defence_agent.reports.templates import get_report_task, list_report_tasks
from defence_agent.tool_state import record_search_documents_state


REPORT_ROOT = Path(__file__).resolve().parents[1] / "data" / "reports" / "generated"
REPORT_SYNTHESIS_MAX_SOURCES = 20
REPORT_SECTION_MAX_SOURCES = 6
REPORT_SYNTHESIS_MAX_TOKENS = 2600
REPORT_SYNTHESIS_REPAIR_ATTEMPTS = 1


async def run_report_task(
    task_id: str,
    *,
    persona_id: str | None = None,
    out_root: Path | str = REPORT_ROOT,
    job_id: str | None = None,
    run_llm_judge: bool = False,
) -> ReportRunResult:
    """Run one scheduled report task and write presentation-ready artifacts."""

    task = get_report_task(task_id)
    resolved_persona_id = persona_id or task.persona_id
    started_at = _utc_now()
    resolved_job_id = job_id or _job_id(task)

    findings = await _run_researchers(task, persona_id=resolved_persona_id)
    synthesis = await asyncio.to_thread(_synthesize_report, task, findings, resolved_persona_id)
    answer_audit = _report_answer_audit(
        task=task,
        job_id=resolved_job_id,
        persona_id=resolved_persona_id,
        findings=findings,
        synthesis=synthesis,
    )
    critic_report = await review_answer(
        query=_synthesis_query(task),
        answer=synthesis.markdown,
        citations=synthesis.citations,
        answer_audit=answer_audit,
        sources_sent_to_answer=synthesis.sources,
    )
    completed_at = _utc_now()
    preliminary = _preliminary_result(
        task=task,
        persona_id=resolved_persona_id,
        job_id=resolved_job_id,
        started_at=started_at,
        completed_at=completed_at,
        findings=findings,
        synthesis=synthesis,
        critic_report=critic_report,
        out_root=Path(out_root),
    )
    eval_summary = evaluate_report_payload(
        task=task,
        markdown=synthesis.markdown,
        sources=synthesis.sources,
        findings=[finding.__dict__ for finding in findings],
        citations=synthesis.citations,
        critic_report=critic_report,
        run_llm_judge=run_llm_judge,
    )
    result = ReportRunResult(
        **{
            **preliminary.__dict__,
            "eval_summary": eval_summary,
            "run_trace": _run_trace(
                task=task,
                job_id=resolved_job_id,
                persona_id=resolved_persona_id,
                started_at=started_at,
                completed_at=completed_at,
                findings=findings,
                synthesis=synthesis,
                critic_report=critic_report,
                eval_summary=eval_summary,
            ),
        }
    )
    artifact_paths = _write_artifacts(result)
    return ReportRunResult(**{**result.__dict__, "artifact_paths": artifact_paths})


async def _run_researchers(task: ReportTask, *, persona_id: str) -> list[ResearchFinding]:
    """Run section researcher agents concurrently."""

    return list(
        await asyncio.gather(
            *[
                asyncio.to_thread(_run_researcher, task, section, persona_id)
                for section in task.sections
            ]
        )
    )


def _run_researcher(task: ReportTask, section: ResearchSectionTask, persona_id: str) -> ResearchFinding:
    try:
        search_result = search_pages(
            query=section.query,
            persona_id=persona_id,
            top_k=section.top_k,
            status_filter="approved",
            language=section.language,
        )
        search_audit = record_search_documents_state(
            tool_context=None,
            query=section.query,
            persona_id=persona_id,
            result=search_result,
        )
        sources = [
            source
            for source in search_result.get("authorized_sources", []) or []
            if isinstance(source, dict)
        ][:REPORT_SECTION_MAX_SOURCES]
        grounded = finalize_answer(
            query=_section_grounding_query(task, section),
            sources=sources,
            prior_answer="",
            fallback_answer="I do not have enough authorized evidence to answer.",
            target_answer_language=section.language,
        )
        return ResearchFinding(
            section_id=section.id,
            title=section.title,
            query=section.query,
            answer=grounded.answer,
            raw_answer=grounded.raw_answer,
            sources=sources,
            citations=grounded.citations,
            search_audit=search_audit,
            model=grounded.model,
        )
    except Exception as exc:
        return ResearchFinding(
            section_id=section.id,
            title=section.title,
            query=section.query,
            answer="",
            raw_answer="",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _synthesize_report(task: ReportTask, findings: list[ResearchFinding], persona_id: str) -> ReportSynthesis:
    sources = _dedupe_sources(
        source
        for finding in findings
        for source in finding.sources
        if finding.status == "completed"
    )[:REPORT_SYNTHESIS_MAX_SOURCES]
    if not sources:
        fallback = "# " + task.title + "\n\nEvidence is insufficient for this scheduled report."
        return ReportSynthesis(
            markdown=fallback,
            raw_markdown=fallback,
            citations=[],
            citation_validation={"passed": False, "errors": ["no_authorized_sources"]},
            sources=[],
            model="none",
        )

    documents, evidence_by_label = _cohere_documents(sources)
    raw_markdown = ""
    markdown = ""
    citations: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    repair_feedback = ""
    response = None
    for _ in range(REPORT_SYNTHESIS_REPAIR_ATTEMPTS + 1):
        response = _chat_report_synthesis(
            task=task,
            findings=findings,
            persona_id=persona_id,
            documents=documents,
            repair_feedback=repair_feedback,
        )
        raw_markdown = _clean_report_markdown(_message_text(response))
        citations = _normalize_citations(_response_citations(response), evidence_by_label)
        validation = _validate_native_citations(citations, evidence_by_label, answer=raw_markdown, query=task.objective)
        markdown = _insert_inline_markers(_ensure_markdown_title(task, raw_markdown), citations)
        format_errors = _report_format_errors(markdown)
        if not format_errors:
            break
        repair_feedback = (
            "The previous draft failed report formatting checks: "
            + "; ".join(format_errors)
            + ". Rewrite the complete report from scratch. Do not emit XML, HTML, <co> tags, or unfinished text."
        )
    return ReportSynthesis(
        markdown=markdown,
        raw_markdown=raw_markdown,
        citations=citations,
        citation_validation=validation,
        sources=sources,
        model=get_settings().cohere_chat_model,
        content_blocks=_message_content_blocks(response),
    )


def _chat_report_synthesis(
    *,
    task: ReportTask,
    findings: list[ResearchFinding],
    persona_id: str,
    documents: list[dict[str, Any]],
    repair_feedback: str = "",
) -> Any:
    from defence_agent.cohere_gateway import cohere_gateway

    return cohere_gateway.chat(
        model=get_settings().cohere_chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Defence Agent Report Reviewer. Generate a staff-ready Markdown report "
                    "from only the provided authorized documents and researcher findings. Use the findings "
                    "as planning notes, not as evidence. The documents are the evidence. Every factual "
                    "paragraph must be supported by native citations. Do not reveal or infer restricted "
                    "source text. Synthesize instead of copying long source passages. Never emit XML, HTML, "
                    "<co> tags, footnotes, or custom citation markup. Return complete Markdown only."
                ),
            },
            {
                "role": "user",
                "content": _synthesis_prompt(task, findings, persona_id, repair_feedback=repair_feedback),
            },
        ],
        documents=documents,
        temperature=0.1,
        max_tokens=REPORT_SYNTHESIS_MAX_TOKENS,
    )


def _section_grounding_query(task: ReportTask, section: ResearchSectionTask) -> str:
    return (
        f"Scheduled report: {task.title}\n"
        f"Report objective: {task.objective}\n"
        f"Section: {section.title}\n"
        f"Section objective: {section.objective}\n"
        f"Output guidance: {section.output_guidance}\n\n"
        "Write the researcher finding for this section. Use only the authorized evidence."
    )


def _synthesis_prompt(
    task: ReportTask,
    findings: list[ResearchFinding],
    persona_id: str,
    *,
    repair_feedback: str = "",
) -> str:
    repair = f"Repair feedback: {repair_feedback}" if repair_feedback else ""
    return "\n\n".join(
        [part for part in [
            f"Report title: {task.title}",
            f"Cadence: {task.cadence} ({task.schedule_label})",
            f"Audience: {task.audience}",
            f"Persona/access context: {persona_id}",
            f"Objective: {task.objective}",
            repair,
            "Required structure, exactly these headings:\n"
            f"# {task.title}\n"
            "## Executive Summary\n"
            "## Key Findings\n"
            "## Evidence Gaps\n"
            "## Planning Implications",
            "Length and style: 550 to 850 words. Prefer synthesis over copied source wording. "
            "Use compact paragraphs and at most 8 bullets total.",
            "Researcher findings:\n" + "\n\n".join(_finding_prompt_block(finding) for finding in findings),
            "Return Markdown only. Do not include JSON.",
        ] if part]
    )


def _finding_prompt_block(finding: ResearchFinding) -> str:
    if finding.status != "completed":
        return f"Section {finding.title}: failed with {finding.error}"
    return (
        f"Section {finding.title}\n"
        f"Research query: {finding.query}\n"
        f"Researcher answer:\n{finding.answer[:2200]}\n"
        f"Top source ids: {', '.join(_source_id(source) for source in finding.sources[:6])}"
    )


def _synthesis_query(task: ReportTask) -> str:
    return f"Generate the scheduled report: {task.title}. Objective: {task.objective}"


def _report_answer_audit(
    *,
    task: ReportTask,
    job_id: str,
    persona_id: str,
    findings: list[ResearchFinding],
    synthesis: ReportSynthesis,
) -> dict[str, Any]:
    return {
        "query": _synthesis_query(task),
        "job_id": job_id,
        "persona_id": persona_id,
        "tool_calls": ["research_section" for _ in findings] + ["synthesize_report"],
        "tool_responses": [finding.status for finding in findings] + ["completed"],
        "retrieval": {
            "search_count": len(findings),
            "search_queries": [finding.query for finding in findings],
            "sources_sent_to_answer": [_source_summary(source) for source in synthesis.sources],
            "excluded_sources": _excluded_sources(findings),
        },
        "generation": {
            "model": synthesis.model,
            "citation_mode": "cohere_native_report",
            "citation_resolution": synthesis.citation_validation,
            "document_count": len(synthesis.sources),
            "cohere_document_ids": [_source_id(source) for source in synthesis.sources],
        },
        "citations": synthesis.citations,
    }


def _preliminary_result(
    *,
    task: ReportTask,
    persona_id: str,
    job_id: str,
    started_at: str,
    completed_at: str,
    findings: list[ResearchFinding],
    synthesis: ReportSynthesis,
    critic_report: dict[str, Any],
    out_root: Path,
) -> ReportRunResult:
    artifact_dir = out_root / job_id
    return ReportRunResult(
        job_id=job_id,
        task=task,
        persona_id=persona_id,
        status="completed",
        started_at=started_at,
        completed_at=completed_at,
        findings=findings,
        synthesis=synthesis,
        critic_report=critic_report,
        eval_summary={},
        artifact_dir=artifact_dir,
        artifact_paths={},
        run_trace={},
    )


def _run_trace(
    *,
    task: ReportTask,
    job_id: str,
    persona_id: str,
    started_at: str,
    completed_at: str,
    findings: list[ResearchFinding],
    synthesis: ReportSynthesis,
    critic_report: dict[str, Any],
    eval_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "report_run_trace.v1",
        "job_id": job_id,
        "task_id": task.id,
        "persona_id": persona_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "architecture": {
            "pattern": "orchestrator_parallel_researchers_reviewer",
            "researcher_agent_count": len(findings),
            "parallel_execution": True,
            "reviewer_agent": "report_reviewer",
            "citation_reviewer_agent": "defence_agent_reviewer",
        },
        "researchers": [
            {
                "section_id": finding.section_id,
                "title": finding.title,
                "status": finding.status,
                "query": finding.query,
                "model": finding.model,
                "source_count": len(finding.sources),
                "citation_count": len(finding.citations),
                "error": finding.error,
            }
            for finding in findings
        ],
        "synthesis": {
            "model": synthesis.model,
            "source_count": len(synthesis.sources),
            "citation_count": len(synthesis.citations),
            "citation_validation": synthesis.citation_validation,
        },
        "critic": critic_report,
        "eval": eval_summary,
    }


def _write_artifacts(result: ReportRunResult) -> dict[str, Path]:
    result.artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_md": result.artifact_dir / "report.md",
        "report_html": result.artifact_dir / "report.html",
        "report_audit": result.artifact_dir / "report_audit.json",
        "evidence_bundle": result.artifact_dir / "evidence_bundle.json",
        "report_eval": result.artifact_dir / "report_eval.json",
        "run_trace": result.artifact_dir / "run_trace.json",
    }
    paths["report_md"].write_text(result.synthesis.markdown + "\n", encoding="utf-8")
    paths["report_html"].write_text(_render_html(result), encoding="utf-8")
    paths["report_audit"].write_text(json.dumps(_report_audit_artifact(result), indent=2), encoding="utf-8")
    paths["evidence_bundle"].write_text(json.dumps(_evidence_bundle(result), indent=2), encoding="utf-8")
    paths["report_eval"].write_text(json.dumps(result.eval_summary, indent=2), encoding="utf-8")
    paths["run_trace"].write_text(json.dumps(result.run_trace, indent=2), encoding="utf-8")
    return paths


def _report_audit_artifact(result: ReportRunResult) -> dict[str, Any]:
    return {
        "schema_version": "report_audit.v1",
        "job_id": result.job_id,
        "task_id": result.task.id,
        "title": result.task.title,
        "status": result.status,
        "schedule_label": result.task.schedule_label,
        "cadence": result.task.cadence,
        "persona_id": result.persona_id,
        "requested_by": result.task.requested_by,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "next_run": result.task.next_run,
        "model_calls_required_for_ui": False,
        "live_model_calls_used_to_generate": True,
        "scheduled_templates": [
            {
                "name": task.title,
                "cadence": task.cadence,
                "owner": task.requested_by,
                "purpose": task.objective,
            }
            for task in list_report_tasks()
        ],
        "architecture": result.run_trace.get("architecture", {}),
        "critic": result.critic_report,
    }


def _evidence_bundle(result: ReportRunResult) -> dict[str, Any]:
    sources = [
        {
            "source_id": _source_id(source),
            "doc_id": source.get("doc_id", ""),
            "title": source.get("title", ""),
            "page": source.get("page", ""),
            "access_level": source.get("access_level", ""),
            "status": source.get("status", ""),
            "language": source.get("language", ""),
            "rerank_score": source.get("rerank_score"),
            "why_used": _why_used(source, result.findings),
        }
        for source in result.synthesis.sources
    ]
    claim_map = [
        {
            "claim": citation.get("text", ""),
            "sources": [
                {
                    "source_id": source.get("source_id", ""),
                    "doc_id": source.get("doc_id", ""),
                    "page": source.get("page", ""),
                }
                for source in citation.get("sources", []) or []
                if isinstance(source, dict)
            ],
        }
        for citation in result.synthesis.citations
    ]
    return {
        "schema_version": "report_evidence_bundle.v1",
        "job_id": result.job_id,
        "sources": sources,
        "claim_map": claim_map,
        "researcher_findings": [
            {
                "section_id": finding.section_id,
                "title": finding.title,
                "query": finding.query,
                "status": finding.status,
                "answer": finding.answer,
                "source_ids": [_source_id(source) for source in finding.sources],
            }
            for finding in result.findings
        ],
    }


def _render_html(result: ReportRunResult) -> str:
    body = _markdown_to_basic_html(result.synthesis.markdown)
    return (
        '<section class="da-report">'
        f"<h1>{html.escape(result.task.title)}</h1>"
        f'<p class="meta">{html.escape(result.task.cadence.title())} report | {html.escape(result.completed_at)}</p>'
        f"{body}"
        "</section>"
    )


def _markdown_to_basic_html(markdown: str) -> str:
    lines = []
    in_list = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                lines.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h2>{html.escape(line[2:].strip())}</h2>")
        elif line.startswith("## "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h3>{html.escape(line[3:].strip())}</h3>")
        elif line.startswith("- "):
            if not in_list:
                lines.append("<ul>")
                in_list = True
            lines.append(f"<li>{html.escape(line[2:].strip())}</li>")
        else:
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        lines.append("</ul>")
    return "\n".join(lines)


def _ensure_markdown_title(task: ReportTask, markdown: str) -> str:
    stripped = markdown.strip()
    if stripped.startswith("#"):
        return stripped
    return f"# {task.title}\n\n{stripped}"


def _clean_report_markdown(markdown: str) -> str:
    cleaned = re.sub(r"<co\b[^>]*>", "", markdown, flags=re.IGNORECASE)
    cleaned = re.sub(r"</co\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?co>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _report_format_errors(markdown: str) -> list[str]:
    errors: list[str] = []
    lowered = markdown.lower()
    for heading in (
        "## executive summary",
        "## key findings",
        "## evidence gaps",
        "## planning implications",
    ):
        if heading not in lowered:
            errors.append(f"missing_heading:{heading}")
    if re.search(r"</?co>|<[^>\n]{1,40}>", markdown, flags=re.IGNORECASE):
        errors.append("raw_markup")
    trimmed = markdown.rstrip()
    if not trimmed or trimmed[-1] not in ".!)]":
        errors.append("appears_truncated")
    return errors


def _dedupe_sources(sources: Any) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        key = _source_id(source)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _source_id(source: dict[str, Any]) -> str:
    return str(source.get("chunk_id") or source.get("source_id") or source.get("doc_id") or "")


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": source.get("chunk_id", ""),
        "doc_id": source.get("doc_id", ""),
        "title": source.get("title", ""),
        "page": source.get("page", ""),
        "status": source.get("status", ""),
        "access_level": source.get("access_level", ""),
        "language": source.get("language", ""),
        "rerank_score": source.get("rerank_score"),
    }


def _excluded_sources(findings: list[ResearchFinding]) -> list[dict[str, Any]]:
    return _dedupe_sources(
        source
        for finding in findings
        for source in finding.search_audit.get("excluded_sources", []) or []
        if isinstance(source, dict)
    )


def _why_used(source: dict[str, Any], findings: list[ResearchFinding]) -> str:
    source_id = _source_id(source)
    sections = [
        finding.title
        for finding in findings
        if any(_source_id(candidate) == source_id for candidate in finding.sources)
    ]
    return "Used by " + ", ".join(sections) if sections else "Used by report synthesis"


def _job_id(task: ReportTask) -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return f"{task.id}_{timestamp}"


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
