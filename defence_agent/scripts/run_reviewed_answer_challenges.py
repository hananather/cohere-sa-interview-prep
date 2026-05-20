#!/usr/bin/env python3
"""Run hard reviewed-answer challenges through the live Defence Agent backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.retrieval.chroma_index import build_index  # noqa: E402
from defence_agent.session import AgentTurnResult, create_session_service, run_turn  # noqa: E402


DEFAULT_CHALLENGES = ROOT / "defence_agent" / "data" / "evals" / "reviewer_challenge_set.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "defence_agent" / "data" / "evals" / "reviewer_challenge_runs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", default=str(DEFAULT_CHALLENGES), help="Path to challenge YAML.")
    parser.add_argument("--case-id", action="append", help="Run only one case id. Can be repeated.")
    parser.add_argument("--max-cases", type=int, help="Run the first N selected cases.")
    parser.add_argument("--output-dir", help="Directory for JSON outputs. Defaults to timestamped run dir.")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Pause between live cases.")
    parser.add_argument("--skip-index-check", action="store_true", help="Do not call build_index(force=False).")
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    cases = _load_cases(Path(args.challenges))
    selected = set(args.case_id or [])
    if selected:
        cases = [case for case in cases if str(case.get("id", "")) in selected]
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("No challenge cases selected.")

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_result = {} if args.skip_index_check else build_index(force=False)
    rows: list[dict[str, Any]] = []
    print("case_id | difficulty | persona | searches | docs | citations | score | action | value")
    print("-" * 120)
    for case in cases:
        outcome = await _run_case(case)
        rows.append(outcome)
        _write_json(output_dir / f"{_safe_filename(outcome['case_id'])}.json", outcome)
        print(_format_row(outcome))
        if args.delay_seconds:
            await asyncio.sleep(args.delay_seconds)

    summary = _summary(
        challenge_path=Path(args.challenges),
        output_dir=output_dir,
        index_result=index_result,
        rows=rows,
    )
    _write_json(output_dir / "run_summary.json", summary)
    (output_dir / "run_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    print(f"\nWrote reviewed-answer challenge run: {output_dir}")
    return 0


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [case for case in data.get("cases", []) or [] if isinstance(case, dict)]


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("reviewed_answer_challenges_%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / stamp


async def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    service = create_session_service(persistent=False)
    persona_id = str(case.get("persona_id", "clearance_unclassified"))
    target_answer_language = str(case.get("target_answer_language", "auto"))
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = await run_turn(
            str(case["query"]),
            persona_id=persona_id,
            session_service=service,
            target_answer_language=target_answer_language,
        )
        outcome = _outcome(case, result, started_at=started_at)
    except Exception as exc:  # pragma: no cover - live diagnostics path
        outcome = _error_outcome(case, exc, started_at=started_at)
    await _close_service(service)
    return outcome


def _outcome(case: dict[str, Any], result: AgentTurnResult, *, started_at: str) -> dict[str, Any]:
    audit = result.answer_audit
    retrieval = audit.get("retrieval", {}) if isinstance(audit, dict) else {}
    generation = audit.get("generation", {}) if isinstance(audit, dict) else {}
    critic = audit.get("critic", {}) if isinstance(audit, dict) else {}
    doc_ids = _doc_ids(audit)
    excluded_doc_ids = _excluded_doc_ids(audit)
    validations = _validate_expectations(case, result, doc_ids=doc_ids, excluded_doc_ids=excluded_doc_ids)
    reviewer_added_value = _reviewer_value(case, result, critic, validations)
    return {
        "case_id": str(case.get("id", "")),
        "difficulty": int(case.get("difficulty", 0) or 0),
        "persona_id": str(case.get("persona_id", "")),
        "query": str(case.get("query", "")),
        "expected_behavior": str(case.get("expected_behavior", "")),
        "expected_reviewer_value": str(case.get("expected_reviewer_value", "")),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "real_backend_evidence": {
            "tool_calls": list(result.tool_calls),
            "tool_responses": list(result.tool_responses),
            "grounded_model": result.grounded_model,
            "critic_model": critic.get("model", ""),
            "documents_sent_to_model": result.documents_sent_to_model,
            "search_count": int(retrieval.get("search_count", 0) or 0),
            "cohere_document_ids": generation.get("cohere_document_ids", []),
        },
        "answer": result.answer,
        "citation_count": len(result.citations),
        "citation_validation": result.citation_validation,
        "doc_ids": sorted(doc_ids),
        "excluded_doc_ids": sorted(excluded_doc_ids),
        "reviewer": {
            "status": critic.get("status", ""),
            "credibility_score": critic.get("credibility_score"),
            "threshold": critic.get("threshold"),
            "verified_citation_count": critic.get("verified_citation_count"),
            "unverified_citation_count": critic.get("unverified_citation_count"),
            "total_citation_count": critic.get("total_citation_count"),
            "answer_citation_count": critic.get("answer_citation_count"),
            "unreviewed_citation_count": critic.get("unreviewed_citation_count"),
            "release_gate": critic.get("release_gate", ""),
            "requires_human_decision": critic.get("requires_human_decision"),
            "critic_output_validation": critic.get("critic_output_validation", {}),
            "summary": critic.get("summary", ""),
            "overall_reason": critic.get("overall_reason", ""),
            "citation_reviews": critic.get("citation_reviews", []),
        },
        "expectation_failures": validations,
        "reviewer_added_value": reviewer_added_value,
        "final_answer_audit": audit,
    }


def _error_outcome(case: dict[str, Any], exc: Exception, *, started_at: str) -> dict[str, Any]:
    return {
        "case_id": str(case.get("id", "")),
        "difficulty": int(case.get("difficulty", 0) or 0),
        "persona_id": str(case.get("persona_id", "")),
        "query": str(case.get("query", "")),
        "expected_behavior": str(case.get("expected_behavior", "")),
        "expected_reviewer_value": str(case.get("expected_reviewer_value", "")),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "real_backend_evidence": {"error": f"{type(exc).__name__}: {exc}"},
        "answer": "",
        "citation_count": 0,
        "citation_validation": {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"]},
        "doc_ids": [],
        "excluded_doc_ids": [],
        "reviewer": {
            "status": "error",
            "credibility_score": 0.0,
            "release_gate": "error",
            "requires_human_decision": True,
            "summary": "Run failed before reviewer analysis.",
        },
        "expectation_failures": [f"{type(exc).__name__}: {exc}"],
        "reviewer_added_value": "run_failed",
        "final_answer_audit": {},
    }


def _validate_expectations(
    case: dict[str, Any],
    result: AgentTurnResult,
    *,
    doc_ids: set[str],
    excluded_doc_ids: set[str],
) -> list[str]:
    failures: list[str] = []
    expected_doc_ids = set(case.get("expected_doc_ids", []) or [])
    allowed_alt_doc_ids = set(case.get("allowed_alt_doc_ids", []) or [])
    acceptable_doc_ids = expected_doc_ids | allowed_alt_doc_ids
    if expected_doc_ids and not (expected_doc_ids <= doc_ids or acceptable_doc_ids & doc_ids):
        failures.append("missing_expected_doc")
    for doc_id in case.get("expected_excluded_doc_ids", []) or []:
        if str(doc_id) not in excluded_doc_ids:
            failures.append(f"missing_excluded_doc:{doc_id}")
    if int(case.get("min_citations", 0) or 0) > len(result.citations):
        failures.append(f"citations<{case.get('min_citations')}")
    expected_documents_sent = case.get("expected_documents_sent_to_model")
    if expected_documents_sent is not None and result.documents_sent_to_model != int(expected_documents_sent):
        failures.append(f"documents_sent!={expected_documents_sent}")
    if case.get("expected_refusal") and not _is_refusal(result.answer):
        failures.append("expected_refusal_missing")
    for fact in case.get("expected_answer_facts", []) or []:
        if str(fact) not in result.answer:
            failures.append(f"missing_answer_fact:{fact}")
    allow_negated_forbidden = str(case.get("expected_behavior", "")).startswith("insufficiency")
    for fact in case.get("forbidden_answer_facts", []) or []:
        if _contains_forbidden_fact(result.answer, str(fact), allow_negated=allow_negated_forbidden):
            failures.append(f"forbidden_answer_fact:{fact}")
    expected_behavior = str(case.get("expected_behavior", ""))
    allows_insufficiency = expected_behavior.startswith("insufficiency") and _is_refusal(result.answer)
    if not case.get("expected_refusal") and not allows_insufficiency and not result.citation_validation.get("passed", False):
        failures.append("citation_validation_failed")
    return failures


def _reviewer_value(
    case: dict[str, Any],
    result: AgentTurnResult,
    critic: dict[str, Any],
    expectation_failures: list[str],
) -> str:
    if critic.get("release_gate") == "human_continue_or_stop_required":
        if not expectation_failures:
            return "quality_gate_added_review_pressure"
        return "quality_gate_caught_case_failure"
    if expectation_failures:
        return "reviewer_missed_case_failure"
    if case.get("expected_refusal") and _is_refusal(result.answer):
        return "confirmed_safe_refusal"
    return "confirmed_release"


def _doc_ids(audit: dict[str, Any]) -> set[str]:
    retrieval = audit.get("retrieval", {}) if isinstance(audit, dict) else {}
    sources = list(retrieval.get("sources_sent_to_answer", []) or [])
    sources += list(retrieval.get("authorized_sources", []) or [])
    citations = audit.get("citations", []) or []
    for citation in citations:
        if isinstance(citation, dict):
            sources += [source for source in citation.get("sources", []) or [] if isinstance(source, dict)]
    return {str(source.get("doc_id", "") or "") for source in sources if source.get("doc_id")}


def _excluded_doc_ids(audit: dict[str, Any]) -> set[str]:
    retrieval = audit.get("retrieval", {}) if isinstance(audit, dict) else {}
    return {
        str(source.get("doc_id", "") or "")
        for source in retrieval.get("excluded_sources", []) or []
        if isinstance(source, dict) and source.get("doc_id")
    }


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in (
            "do not have enough",
            "insufficient evidence",
            "evidence is insufficient",
            "no authorized evidence",
            "cannot answer",
        )
    )


def _contains_forbidden_fact(answer: str, fact: str, *, allow_negated: bool = False) -> bool:
    lowered = answer.lower()
    needle = fact.lower()
    for match in re.finditer(re.escape(needle), lowered):
        if allow_negated and _is_negated_reference(lowered[max(0, match.start() - 100) : match.start()]):
            continue
        return True
    return False


def _is_negated_reference(prefix: str) -> bool:
    return any(
        marker in prefix
        for marker in (
            "do not contain",
            "does not contain",
            "do not include",
            "does not include",
            "not contain",
            "not include",
            "no evidence of",
            "no cited evidence for",
            "evidence is insufficient for",
        )
    )


def _summary(
    *,
    challenge_path: Path,
    output_dir: Path,
    index_result: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    value_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("reviewer_added_value", ""))
        value_counts[key] = value_counts.get(key, 0) + 1
    scores = [
        float(row.get("reviewer", {}).get("credibility_score"))
        for row in rows
        if isinstance(row.get("reviewer", {}).get("credibility_score"), (int, float))
    ]
    return {
        "challenge_path": str(challenge_path),
        "output_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_result": {
            "embedding_backend": index_result.get("embedding_backend"),
            "parsed_pages": index_result.get("parsed_pages"),
            "embedded_pages": index_result.get("embedded_pages"),
            "skipped": index_result.get("skipped"),
        },
        "case_count": len(rows),
        "value_counts": value_counts,
        "average_credibility_score": round(sum(scores) / len(scores), 4) if scores else None,
        "rows": [
            {
                "case_id": row["case_id"],
                "difficulty": row["difficulty"],
                "reviewer_added_value": row["reviewer_added_value"],
                "credibility_score": row["reviewer"].get("credibility_score"),
                "release_gate": row["reviewer"].get("release_gate"),
                "expectation_failures": row["expectation_failures"],
            }
            for row in rows
        ],
    }


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Reviewed Answer Challenge Run",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Average credibility score: {summary['average_credibility_score']}",
        f"- Output dir: `{summary['output_dir']}`",
        "",
        "| Case | Difficulty | Score | Action | Value | Failures |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in summary["rows"]:
        failures = ", ".join(row["expectation_failures"]) or ""
        lines.append(
            f"| {row['case_id']} | {row['difficulty']} | {row['credibility_score']} | "
            f"{row['release_gate']} | {row['reviewer_added_value']} | {failures} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_row(outcome: dict[str, Any]) -> str:
    evidence = outcome.get("real_backend_evidence", {})
    reviewer = outcome.get("reviewer", {})
    return (
        f"{outcome['case_id']} | {outcome['difficulty']} | {outcome['persona_id']} | "
        f"{evidence.get('search_count', 0)} | {evidence.get('documents_sent_to_model', 0)} | "
        f"{outcome['citation_count']} | {reviewer.get('credibility_score')} | "
        f"{reviewer.get('release_gate')} | {outcome['reviewer_added_value']}"
    )


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "case"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


async def _close_service(service: Any) -> None:
    close = getattr(service, "close", None)
    if callable(close):
        result = close()
        if asyncio.iscoroutine(result):
            await result


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
