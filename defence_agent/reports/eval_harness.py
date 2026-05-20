"""Eval harness for background report-generation agents."""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any

from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.reports.models import ReportRunResult, ReportTask


REPORT_JUDGE_MODEL = "command-a-03-2025"


def evaluate_report_run(
    result: ReportRunResult,
    *,
    run_llm_judge: bool = False,
) -> dict[str, Any]:
    """Run deterministic and optional LLM-as-judge evaluators for a report run."""

    checks = _deterministic_checks(result)
    if run_llm_judge:
        checks["llm_judge"] = _llm_judge_check(result)
    else:
        checks["llm_judge"] = {
            "status": "skipped",
            "score": None,
            "note": "LLM-as-judge evaluator was not requested for this run.",
        }
    scores = [
        float(check["score"])
        for check in checks.values()
        if isinstance(check, dict) and isinstance(check.get("score"), int | float)
    ]
    return {
        "schema_version": "report_eval.v1",
        "task_id": result.task.id,
        "job_id": result.job_id,
        "overall_score": round(mean(scores), 4) if scores else None,
        "evaluator_types": ["code", "llm_judge"],
        "checks": checks,
        "summary": _eval_summary(checks),
        "anthropic_eval_mapping": {
            "task": "scheduled report job with defined audience, cadence, and success criteria",
            "trial": result.job_id,
            "transcript": "run_trace.json records section queries, model calls, and reviewer output",
            "outcome": "report.md plus evidence_bundle.json and report_eval.json",
            "graders": "deterministic code evaluators plus optional Cohere LLM-as-judge",
        },
    }


def evaluate_report_payload(
    *,
    task: ReportTask,
    markdown: str,
    sources: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    critic_report: dict[str, Any] | None = None,
    run_llm_judge: bool = False,
) -> dict[str, Any]:
    """Evaluate a report-like payload without requiring a full run object."""

    checks = {
        "artifact_shape": _artifact_shape_check(markdown),
        "format_safety": _format_safety_check(markdown),
        "section_coverage": _section_coverage_check(task, markdown),
        "source_coverage": _source_coverage_check(task, sources),
        "citation_integrity": _citation_integrity_check(markdown, citations),
        "access_safety": _access_safety_check(sources),
        "parallel_research": _parallel_research_check(findings),
        "reviewer_score": _critic_gate_check(critic_report or {}),
    }
    if run_llm_judge:
        checks["llm_judge"] = _llm_judge_payload_check(task=task, markdown=markdown, sources=sources)
    else:
        checks["llm_judge"] = {
            "status": "skipped",
            "score": None,
            "note": "LLM-as-judge evaluator was not requested for this payload.",
        }
    scores = [
        float(check["score"])
        for check in checks.values()
        if isinstance(check, dict) and isinstance(check.get("score"), int | float)
    ]
    return {
        "schema_version": "report_eval.v1",
        "task_id": task.id,
        "overall_score": round(mean(scores), 4) if scores else None,
        "evaluator_types": ["code", "llm_judge"],
        "checks": checks,
        "summary": _eval_summary(checks),
    }


def _deterministic_checks(result: ReportRunResult) -> dict[str, dict[str, Any]]:
    return {
        "artifact_shape": _artifact_shape_check(result.synthesis.markdown),
        "format_safety": _format_safety_check(result.synthesis.markdown),
        "section_coverage": _section_coverage_check(result.task, result.synthesis.markdown),
        "source_coverage": _source_coverage_check(result.task, result.synthesis.sources),
        "citation_integrity": _citation_integrity_check(result.synthesis.markdown, result.synthesis.citations),
        "access_safety": _access_safety_check(result.synthesis.sources),
        "parallel_research": _parallel_research_check([finding.__dict__ for finding in result.findings]),
        "reviewer_score": _critic_gate_check(result.critic_report),
    }


def _artifact_shape_check(markdown: str) -> dict[str, Any]:
    has_title = markdown.lstrip().startswith("#")
    has_sections = markdown.count("\n## ") >= 2
    length_ok = len(markdown.split()) >= 250
    score = sum([has_title, has_sections, length_ok]) / 3
    return {
        "status": "pass" if score >= 1.0 else "warn" if score >= 0.67 else "fail",
        "score": round(score, 4),
        "note": "Report has title, multiple sections, and enough depth for a staff brief.",
    }


def _format_safety_check(markdown: str) -> dict[str, Any]:
    required_headings = (
        "## Executive Summary",
        "## Key Findings",
        "## Evidence Gaps",
        "## Planning Implications",
    )
    missing = [heading for heading in required_headings if heading.lower() not in markdown.lower()]
    raw_tags = re.findall(r"</?co>|<[^>\n]{1,40}>", markdown, flags=re.IGNORECASE)
    trimmed = markdown.rstrip()
    appears_complete = bool(trimmed) and trimmed[-1] in ".!)]"
    failures = []
    if missing:
        failures.append("missing_headings:" + ",".join(missing))
    if raw_tags:
        failures.append("raw_markup:" + ",".join(sorted(set(raw_tags))[:5]))
    if not appears_complete:
        failures.append("report_appears_truncated")
    score = 1.0 if not failures else max(0.0, 1.0 - (0.34 * len(failures)))
    return {
        "status": "pass" if not failures else "fail" if score < 0.5 else "warn",
        "score": round(score, 4),
        "note": (
            "Report format is complete and free of raw model markup."
            if not failures
            else "Format issues: " + "; ".join(failures)
        ),
    }


def _section_coverage_check(task: ReportTask, markdown: str) -> dict[str, Any]:
    text = markdown.lower()
    hits = [
        section.id
        for section in task.sections
        if section.title.lower() in text or _tokens_present(section.title, text)
    ]
    required_hits = [term for term in task.required_terms if term.lower() in text]
    section_score = len(hits) / len(task.sections) if task.sections else 1.0
    term_score = len(required_hits) / len(task.required_terms) if task.required_terms else 1.0
    score = round((section_score + term_score) / 2, 4)
    return {
        "status": "pass" if score >= 0.8 else "warn" if score >= 0.5 else "fail",
        "score": score,
        "note": f"Covered {len(hits)}/{len(task.sections)} sections and {len(required_hits)}/{len(task.required_terms)} required terms.",
    }


def _source_coverage_check(task: ReportTask, sources: list[dict[str, Any]]) -> dict[str, Any]:
    observed = {str(source.get("doc_id", "")) for source in sources if source.get("doc_id")}
    expected = set(task.expected_doc_ids)
    score = len(observed & expected) / len(expected) if expected else 1.0
    return {
        "status": "pass" if score >= 0.67 else "warn" if score > 0 else "fail",
        "score": round(score, 4),
        "note": "Observed expected source families: " + ", ".join(sorted(observed & expected)) or "No expected sources observed.",
    }


def _citation_integrity_check(markdown: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    marker_count = len(re.findall(r"\[C\d+(?:,\s*C\d+)*\]", markdown))
    citation_count = len(citations)
    has_native_citations = citation_count > 0
    has_markers = marker_count > 0
    score = 1.0 if has_native_citations and has_markers else 0.5 if has_native_citations or has_markers else 0.0
    return {
        "status": "pass" if score == 1.0 else "warn" if score == 0.5 else "fail",
        "score": score,
        "note": f"Native citations: {citation_count}; rendered citation markers: {marker_count}.",
    }


def _access_safety_check(sources: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe = [
        str(source.get("doc_id", ""))
        for source in sources
        if str(source.get("access_level", "")).lower() not in {"", "unclassified"}
    ]
    score = 0.0 if unsafe else 1.0
    return {
        "status": "pass" if not unsafe else "fail",
        "score": score,
        "note": "No restricted sources were used." if not unsafe else "Restricted sources used: " + ", ".join(unsafe),
    }


def _parallel_research_check(findings: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [finding for finding in findings if finding.get("status") == "completed"]
    score = 1.0 if len(completed) >= 2 else 0.5 if completed else 0.0
    return {
        "status": "pass" if score == 1.0 else "warn" if score == 0.5 else "fail",
        "score": score,
        "note": f"{len(completed)} researcher section(s) completed. The harness runs them with asyncio gather.",
    }


def _critic_gate_check(critic_report: dict[str, Any]) -> dict[str, Any]:
    status = str(critic_report.get("status", "") or "")
    release_gate = str(critic_report.get("release_gate", "") or "")
    if status == "approved" and release_gate == "release":
        score = 1.0
    elif status:
        score = 0.5
    else:
        score = 0.0
    return {
        "status": "pass" if score == 1.0 else "warn" if score == 0.5 else "fail",
        "score": score,
        "note": f"Reviewer status: {status or 'missing'}; action: {release_gate or 'missing'}.",
    }


def _llm_judge_check(result: ReportRunResult) -> dict[str, Any]:
    return _llm_judge_payload_check(
        task=result.task,
        markdown=result.synthesis.markdown,
        sources=result.synthesis.sources,
    )


def _llm_judge_payload_check(
    *,
    task: ReportTask,
    markdown: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = {
        "task": {
            "id": task.id,
            "title": task.title,
            "objective": task.objective,
            "audience": task.audience,
            "required_terms": list(task.required_terms),
            "expected_doc_ids": list(task.expected_doc_ids),
        },
        "report_markdown": markdown[:10000],
        "source_summaries": [_source_summary(source) for source in sources[:20]],
    }
    try:
        response = cohere_gateway.chat(
            model=REPORT_JUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a calibrated evaluator for a defence research report agent. "
                        "Grade only the supplied report and source summaries. Return JSON only with "
                        "scores from 0.0 to 1.0 for factual_grounding, completeness, usefulness, "
                        "citation_discipline, and access_safety, plus pass boolean and short rationale."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        payload = _parse_json(_response_text(response))
        scores = [
            float(payload[key])
            for key in ("factual_grounding", "completeness", "usefulness", "citation_discipline", "access_safety")
            if isinstance(payload.get(key), int | float)
        ]
        score = round(mean(scores), 4) if scores else 0.0
        return {
            "status": "pass" if bool(payload.get("pass")) and score >= 0.75 else "warn" if score >= 0.5 else "fail",
            "score": score,
            "note": str(payload.get("rationale", "") or "LLM judge returned a structured score."),
            "raw": payload,
        }
    except Exception as exc:
        return {
            "status": "warn",
            "score": None,
            "note": f"LLM judge failed: {type(exc).__name__}: {exc}",
        }


def _tokens_present(title: str, text: str) -> bool:
    tokens = [token for token in re.findall(r"[a-z0-9]+", title.lower()) if len(token) > 4]
    if not tokens:
        return False
    return sum(1 for token in tokens if token in text) >= max(1, len(tokens) - 1)


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": source.get("doc_id", ""),
        "title": source.get("title", ""),
        "page": source.get("page", ""),
        "access_level": source.get("access_level", ""),
        "rerank_score": source.get("rerank_score"),
    }


def _response_text(response: Any) -> str:
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message else None
    if isinstance(content, list):
        return "".join(str(getattr(part, "text", "") or "") for part in content).strip()
    return str(content or "").strip()


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM judge returned JSON that was not an object")
    return parsed


def _eval_summary(checks: dict[str, dict[str, Any]]) -> str:
    failed = [name for name, check in checks.items() if check.get("status") == "fail"]
    warned = [name for name, check in checks.items() if check.get("status") == "warn"]
    if failed:
        return "Failing checks: " + ", ".join(failed)
    if warned:
        return "Warnings: " + ", ".join(warned)
    return "All report-agent eval checks passed."
