#!/usr/bin/env python3
"""Run the presentation demo query registry through the ADK harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.evals.harness import load_registry
from defence_agent.config import get_settings
from defence_agent.session import AgentTurnResult, create_session_service, run_turn


DEFAULT_REGISTRY = ROOT / "defence_agent" / "data" / "evals" / "demo_query_registry.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to the demo query registry YAML.")
    parser.add_argument("--case-id", action="append", help="Run only one case id. Can be repeated.")
    parser.add_argument("--include-variants", action="store_true", help="Also run query_variants for selected cases.")
    parser.add_argument("--show-audit", action="store_true", help="Print full answer_audit JSON for each turn.")
    parser.add_argument("--output-dir", help="Write one JSON transcript per case to this directory.")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Pause between live Cohere cases.")
    parser.add_argument("--run-mode", default=os.getenv("DEFENCE_AGENT_RUN_MODE", "reviewed_agent"))
    parser.add_argument("--accuracy-priority", type=int, default=int(os.getenv("DEFENCE_AGENT_ACCURACY_PRIORITY", "4")))
    parser.add_argument("--latency-priority", type=int, default=int(os.getenv("DEFENCE_AGENT_LATENCY_PRIORITY", "2")))
    parser.add_argument("--max-review-cycles", type=int, default=int(os.getenv("DEFTECH_ADK_MAX_REVIEW_CYCLES", "2")))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="When --output-dir is set, reuse existing per-case JSON transcripts instead of rerunning them.",
    )
    parser.add_argument("--no-fail", action="store_true", help="Print failures but exit zero.")
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    registry = _load_registry(Path(args.registry))
    cases = [case for case in registry.get("cases", []) if isinstance(case, dict)]
    cases = _expand_cases(cases, include_variants=args.include_variants)
    selected_ids = set(args.case_id or [])
    if selected_ids:
        cases = [
            case
            for case in cases
            if str(case.get("id", "")) in selected_ids or str(case.get("parent_id", "")) in selected_ids
        ]
    if selected_ids and not cases:
        known_ids = {str(case.get("id", "")) for case in _expand_cases(registry.get("cases", []), include_variants=True)}
        missing = sorted(selected_ids - known_ids)
        raise SystemExit(f"Unknown case id(s): {', '.join(missing)}")

    failures = 0
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    print("case_id | status | persona | route | searches | docs | excluded | citations | mode | reviewer | notes")
    print("-" * 126)
    for case in cases:
        existing_path = _existing_outcome_path(output_dir, case)
        if args.skip_existing and existing_path and existing_path.exists():
            outcome = _load_existing_outcome(existing_path)
            outcome["_skipped_existing"] = True
        else:
            try:
                outcome = await _run_case(
                    case,
                    show_audit=args.show_audit,
                    run_mode=args.run_mode,
                    accuracy_priority=args.accuracy_priority,
                    latency_priority=args.latency_priority,
                    max_review_cycles=args.max_review_cycles,
                )
            except Exception as exc:  # pragma: no cover - exercised during live readiness runs
                outcome = _error_outcome(case, exc)
        failures += 0 if outcome["passed"] else 1
        print(_format_row(outcome))
        if output_dir and not outcome.get("_skipped_existing"):
            _write_outcome(output_dir, outcome)
        if args.delay_seconds > 0:
            await asyncio.sleep(args.delay_seconds)
    return 0 if args.no_fail or failures == 0 else 1


def _load_registry(path: Path) -> dict[str, Any]:
    return load_registry(path)


def _expand_cases(cases: list[dict[str, Any]], *, include_variants: bool) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for case in cases:
        expanded.append(case)
        if not include_variants:
            continue
        for index, variant in enumerate(case.get("query_variants", []) or [], start=1):
            variant_case = dict(case)
            variant_case.pop("query_variants", None)
            variant_case.pop("follow_up", None)
            variant_case["parent_id"] = str(case.get("id", ""))
            variant_case["id"] = f"{case.get('id')}__variant_{index:02d}"
            variant_case["query"] = str(variant.get("query", "")) if isinstance(variant, dict) else str(variant)
            if isinstance(variant, dict):
                variant_case.update({key: value for key, value in variant.items() if key != "query"})
            variant_case["demo_point"] = f"Variant of {case.get('id')}: {case.get('demo_point', '')}"
            expanded.append(variant_case)
    return expanded


async def _run_case(
    case: dict[str, Any],
    *,
    show_audit: bool,
    run_mode: str,
    accuracy_priority: int,
    latency_priority: int,
    max_review_cycles: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    service = create_session_service(persistent=False)
    persona_id = str(case.get("persona_id", "clearance_unclassified"))
    target_answer_language = str(case.get("target_answer_language", "auto"))
    first = await run_turn(
        str(case["query"]),
        persona_id=persona_id,
        session_service=service,
        target_answer_language=target_answer_language,
        run_mode=run_mode,
        accuracy_priority=accuracy_priority,
        latency_priority=latency_priority,
        max_review_cycles=max_review_cycles,
    )
    final = first
    if case.get("follow_up"):
        final = await run_turn(
            str(case["follow_up"]),
            persona_id=persona_id,
            session_id=first.session_id,
            session_service=service,
            target_answer_language=target_answer_language,
            run_mode=run_mode,
            accuracy_priority=accuracy_priority,
            latency_priority=latency_priority,
            max_review_cycles=max_review_cycles,
        )
    if show_audit:
        print(f"\n## {case.get('id')} / turn 1 audit")
        print(json.dumps(first.answer_audit, indent=2, sort_keys=True, ensure_ascii=False))
        if final is not first:
            print(f"\n## {case.get('id')} / follow-up audit")
            print(json.dumps(final.answer_audit, indent=2, sort_keys=True, ensure_ascii=False))
    checks = _validate_case(case, first, final)
    await _close_service(service)
    latency_seconds = round(time.perf_counter() - started, 4)
    return {
        "case_id": str(case.get("id", "")),
        "parent_id": str(case.get("parent_id", "")),
        "persona_id": persona_id,
        "target_answer_language": target_answer_language,
        "passed": not checks,
        "failures": checks,
        "latency_seconds": latency_seconds,
        "latency_ms": int(latency_seconds * 1000),
        "search_count": _search_count(first),
        "doc_ids": sorted(_doc_ids(final) or _doc_ids(first)),
        "excluded_doc_ids": sorted(_excluded_doc_ids(final) or _excluded_doc_ids(first)),
        "citation_count": len(final.citations),
        "mode": final.citation_mode,
        "critic": _critic_summary(final),
        "routing": final.answer_audit.get("routing", {}),
        "retrieval_modes": final.answer_audit.get("retrieval", {}).get("retrieval_modes", []),
        "chunk_strategies": final.answer_audit.get("retrieval", {}).get("chunk_strategies", []),
        "retrieval_metrics": final.answer_audit.get("retrieval", {}).get("retrieval_metrics", []),
        "citation_quality": final.answer_audit.get("generation", {}).get("citation_quality", {}),
        "operational_metrics": {
            "latency_seconds": latency_seconds,
            "latency_ms": int(latency_seconds * 1000),
            "measured_latency": True,
            "token_usage": final.answer_audit.get("generation", {}).get("usage", {}),
            "billed_units": final.answer_audit.get("generation", {}).get("billed_units", {}),
        },
        "eval_metadata": _eval_metadata(run_mode),
        "demo_point": str(case.get("demo_point", "")),
        "query": str(case.get("query", "")),
        "follow_up": str(case.get("follow_up", "")),
        "answer": final.answer,
        "citation_validation": final.citation_validation,
        "first_answer_audit": first.answer_audit,
        "final_answer_audit": final.answer_audit,
    }


def _error_outcome(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    message = f"{type(exc).__name__}: {exc}"
    return {
        "case_id": str(case.get("id", "")),
        "parent_id": str(case.get("parent_id", "")),
        "persona_id": str(case.get("persona_id", "")),
        "passed": False,
        "failures": [message],
        "search_count": 0,
        "doc_ids": [],
        "excluded_doc_ids": [],
        "citation_count": 0,
        "mode": "error",
        "critic": {"status": "error", "score": 0.0, "gate": "error"},
        "routing": {},
        "citation_quality": {},
        "eval_metadata": _eval_metadata(str(os.getenv("DEFENCE_AGENT_RUN_MODE", ""))),
        "demo_point": str(case.get("demo_point", "")),
        "query": str(case.get("query", "")),
        "follow_up": str(case.get("follow_up", "")),
        "answer": "",
        "citation_validation": {"passed": False, "errors": [message]},
        "first_answer_audit": {},
        "final_answer_audit": {},
    }


def _eval_metadata(run_mode: str) -> dict[str, Any]:
    settings = get_settings()
    return {
        "schema_version": "eval_run_metadata.v1",
        "harness_id": "defence_agent.run_turn.v1",
        "route": run_mode,
        "route_implementation_id": _route_implementation_id(run_mode),
        "cohere_chat_model": settings.cohere_chat_model,
        "deftech_adk_model": os.getenv("DEFTECH_ADK_MODEL", "cohere/command-a-03-2025"),
        "deftech_adk_critic_model": os.getenv("DEFTECH_ADK_CRITIC_MODEL", "command-a-reasoning-08-2025"),
        "deftech_adk_reviewer_model": os.getenv("DEFTECH_ADK_REVIEWER_MODEL", ""),
        "chunk_strategy": settings.chunk_strategy,
        "retrieval_mode": settings.retrieval_mode,
        "cohere_embed_model": settings.cohere_embed_model,
        "cohere_rerank_model": settings.cohere_rerank_model,
        "local_pacing_requests_per_minute": settings.cohere_requests_per_minute,
    }


def _route_implementation_id(run_mode: str) -> str:
    if run_mode == "simple_rag":
        return "direct_search_pages_plus_grounded_generation.v1"
    if run_mode in {"agentic_rag", "reviewed_agent"}:
        return "adk_litellm_planner_plus_grounded_generation.v1"
    return "unknown"


def _write_outcome(output_dir: Path, outcome: dict[str, Any]) -> None:
    path = output_dir / f"{_safe_filename(outcome['case_id'])}.json"
    path.write_text(json.dumps(outcome, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _existing_outcome_path(output_dir: Path | None, case: dict[str, Any]) -> Path | None:
    if output_dir is None:
        return None
    return output_dir / f"{_safe_filename(str(case.get('id', '')))}.json"


def _load_existing_outcome(path: Path) -> dict[str, Any]:
    outcome = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(outcome, dict):
        raise ValueError(f"Existing transcript is not a JSON object: {path}")
    return outcome


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "case"


def _validate_case(case: dict[str, Any], first: AgentTurnResult, final: AgentTurnResult) -> list[str]:
    failures: list[str] = []
    expected_doc_ids = set(case.get("expected_doc_ids", []) or [])
    allowed_alt_doc_ids = set(case.get("allowed_alt_doc_ids", []) or [])
    expected_excluded_doc_ids = set(case.get("expected_excluded_doc_ids", []) or [])
    expected_source_types = set(case.get("expected_source_types", []) or [])
    min_search_calls = int(case.get("min_search_calls", 1))
    min_citations = int(case.get("min_citations", 0))
    first_doc_ids = _doc_ids(first)
    final_doc_ids = _doc_ids(final)
    all_doc_ids = first_doc_ids | final_doc_ids
    excluded_doc_ids = _excluded_doc_ids(first) | _excluded_doc_ids(final)

    if case.get("expected_clarification"):
        if not _answer_requests_clarification(final.answer):
            failures.append("expected_clarification_missing")
        if final.citations:
            failures.append("clarification_should_not_have_citations")
        return failures

    if allowed_alt_doc_ids:
        acceptable_doc_ids = expected_doc_ids | allowed_alt_doc_ids
        if acceptable_doc_ids and not (acceptable_doc_ids & all_doc_ids):
            failures.append("missing_expected_or_alt_doc")
    else:
        for doc_id in sorted(expected_doc_ids - all_doc_ids):
            failures.append(f"missing_doc:{doc_id}")
    for doc_id in sorted(expected_excluded_doc_ids):
        if doc_id not in excluded_doc_ids:
            failures.append(f"missing_excluded_doc:{doc_id}")
    for source_type in sorted(expected_source_types):
        if source_type not in _source_types(first) | _source_types(final):
            failures.append(f"missing_source_type:{source_type}")
    if _search_count(first) < min_search_calls:
        failures.append(f"search_count<{min_search_calls}")
    if len(_search_queries(first)) < min_search_calls:
        failures.append(f"search_queries<{min_search_calls}")
    if len(final.citations) < min_citations:
        failures.append(f"citations<{min_citations}")
    expected_documents_sent = case.get("expected_documents_sent_to_model")
    if expected_documents_sent is not None and final.documents_sent_to_model != int(expected_documents_sent):
        failures.append(f"documents_sent!={expected_documents_sent}")
    if case.get("expected_refusal") and not _is_refusal_answer(final.answer):
        failures.append("expected_refusal_answer_missing")
    if (
        not case.get("expected_refusal")
        and not case.get("allow_citation_validation_failure")
        and not final.citation_validation.get("passed", False)
    ):
        failures.append("citation_validation_failed")
    expected_critic_gate = case.get("expected_reviewer_action", case.get("expected_critic_gate"))
    if expected_critic_gate and _critic_summary(final).get("gate") != str(expected_critic_gate):
        failures.append(f"reviewer_action:{_critic_summary(final).get('gate')}")
    for fact in case.get("expected_answer_facts", []) or []:
        if str(fact) not in final.answer:
            failures.append(f"missing_answer_fact:{fact}")
    allow_negated_forbidden = str(case.get("expected_behavior", "")).startswith("insufficiency")
    for fact in case.get("forbidden_answer_facts", []) or []:
        if _contains_forbidden_fact(final.answer, str(fact), allow_negated=allow_negated_forbidden):
            failures.append(f"forbidden_answer_fact:{fact}")
    expected_follow_up_mode = case.get("expected_follow_up_mode")
    if expected_follow_up_mode and final.citation_mode != expected_follow_up_mode:
        failures.append(f"follow_up_mode:{final.citation_mode}")
    failures.extend(_validate_citation_source_resolution(final))
    failures.extend(_validate_facets(case, first, final))
    return failures


def _answer_requests_clarification(answer: str) -> bool:
    text = str(answer or "").lower()
    clarification_terms = {
        "clarify",
        "which",
        "specify",
        "what document",
        "which document",
        "which policy",
        "which manual",
        "which source",
        "more detail",
        "quel",
        "quelle",
        "preciser",
        "préciser",
    }
    question_like = "?" in text or any(term in text for term in clarification_terms)
    unsupported_answer = "not have enough" in text or "evidence is insufficient" in text
    return question_like and not unsupported_answer


def _search_count(result: AgentTurnResult) -> int:
    return int(result.answer_audit.get("retrieval", {}).get("search_count", 0) or 0)


def _is_refusal_answer(answer: str) -> bool:
    text = answer.lower()
    return any(
        phrase in text
        for phrase in [
            "do not have enough",
            "not enough",
            "evidence is insufficient",
            "insufficient evidence",
            "no authorized evidence",
            "cannot answer",
        ]
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


def _doc_ids(result: AgentTurnResult) -> set[str]:
    retrieval = result.answer_audit.get("retrieval", {})
    source_groups = [
        retrieval.get("sources_sent_to_answer", []),
        retrieval.get("authorized_sources", []),
    ]
    return {
        str(source.get("doc_id", ""))
        for group in source_groups
        for source in group or []
        if isinstance(source, dict) and source.get("doc_id")
    }


def _source_types(result: AgentTurnResult) -> set[str]:
    return {
        str(source.get("source_type", ""))
        for source in _retrieval_sources(result)
        if source.get("source_type")
    }


def _retrieval_sources(result: AgentTurnResult) -> list[dict[str, Any]]:
    retrieval = result.answer_audit.get("retrieval", {})
    source_groups = [
        retrieval.get("sources_sent_to_answer", []),
        retrieval.get("authorized_sources", []),
    ]
    return [
        source
        for group in source_groups
        for source in group or []
        if isinstance(source, dict)
    ]


def _citation_sources(result: AgentTurnResult) -> list[dict[str, Any]]:
    return [
        source
        for citation in result.answer_audit.get("citations", []) or []
        if isinstance(citation, dict)
        for source in citation.get("sources", []) or []
        if isinstance(source, dict)
    ]


def _validate_citation_source_resolution(result: AgentTurnResult) -> list[str]:
    failures: list[str] = []
    citations = [citation for citation in result.answer_audit.get("citations", []) or [] if isinstance(citation, dict)]
    if not citations:
        return failures

    sent_keys = _source_keys_for_resolution(
        result.answer_audit.get("retrieval", {}).get("sources_sent_to_answer", []),
        include_doc_id=False,
    )
    excluded_keys = _source_keys_for_resolution(
        result.answer_audit.get("retrieval", {}).get("excluded_sources", []),
        include_doc_id=True,
    )
    cohere_document_ids = {
        str(value)
        for value in result.answer_audit.get("generation", {}).get("cohere_document_ids", []) or []
        if str(value)
    }
    allowed_keys = sent_keys | cohere_document_ids

    for citation in citations:
        for source in citation.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            keys = _source_keys_for_resolution([source], include_doc_id=False)
            exclusion_keys = _source_keys_for_resolution([source], include_doc_id=True)
            if not keys:
                failures.append("citation_source_missing_id")
                continue
            if not keys & allowed_keys:
                failures.append(f"citation_source_not_sent:{_source_label(source)}")
            if exclusion_keys & excluded_keys:
                failures.append(f"citation_source_is_excluded:{_source_label(source)}")
    return failures


def _source_keys_for_resolution(sources: Any, *, include_doc_id: bool) -> set[str]:
    keys: set[str] = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        for field in ("source_id", "chunk_id", "cohere_document_id", "citation_id"):
            value = str(source.get(field, "")).strip()
            if value:
                keys.add(value)
        doc_id = str(source.get("doc_id", "")).strip()
        page = str(source.get("page", "")).strip()
        if doc_id and include_doc_id:
            keys.add(doc_id)
        if doc_id and page:
            keys.add(f"{doc_id}:page:{page}")
    return keys


def _source_label(source: dict[str, Any]) -> str:
    return str(
        source.get("source_id")
        or source.get("chunk_id")
        or source.get("cohere_document_id")
        or source.get("doc_id")
        or "unknown"
    )


def _search_queries(result: AgentTurnResult) -> list[str]:
    return [
        str(query)
        for query in result.answer_audit.get("retrieval", {}).get("search_queries", []) or []
        if str(query).strip()
    ]


def _validate_facets(case: dict[str, Any], first: AgentTurnResult, final: AgentTurnResult) -> list[str]:
    failures: list[str] = []
    facets = [facet for facet in case.get("expected_facets", []) or [] if isinstance(facet, dict)]
    if not facets:
        return failures

    retrieval_sources = _retrieval_sources(first) + _retrieval_sources(final)
    citation_sources = _citation_sources(final)
    search_text = " ".join(_search_queries(first) + _search_queries(final)).lower()

    for facet in facets:
        facet_id = str(facet.get("id", "facet"))
        expected_doc_ids = set(facet.get("expected_doc_ids", []) or [])
        allowed_alt_doc_ids = set(facet.get("allowed_alt_doc_ids", []) or [])
        acceptable_doc_ids = expected_doc_ids | allowed_alt_doc_ids
        if expected_doc_ids:
            if allowed_alt_doc_ids:
                if not any(source.get("doc_id") in acceptable_doc_ids for source in retrieval_sources):
                    failures.append(f"facet:{facet_id}:missing_expected_or_alt_doc")
            else:
                found = {str(source.get("doc_id", "")) for source in retrieval_sources}
                for doc_id in sorted(expected_doc_ids - found):
                    failures.append(f"facet:{facet_id}:missing_doc:{doc_id}")

        min_sources = int(facet.get("min_sources", 0) or 0)
        if min_sources and acceptable_doc_ids:
            source_count = sum(1 for source in retrieval_sources if source.get("doc_id") in acceptable_doc_ids)
            if source_count < min_sources:
                failures.append(f"facet:{facet_id}:sources<{min_sources}")

        min_citations = int(facet.get("min_citations", 0) or 0)
        if min_citations and acceptable_doc_ids:
            citation_count = sum(1 for source in citation_sources if source.get("doc_id") in acceptable_doc_ids)
            if citation_count < min_citations:
                failures.append(f"facet:{facet_id}:citations<{min_citations}")

        for expected_page in facet.get("expected_pages", []) or []:
            if not isinstance(expected_page, dict):
                continue
            doc_id = str(expected_page.get("doc_id", ""))
            pages = {int(page) for page in expected_page.get("pages", []) or []}
            found_pages = {
                int(source.get("page"))
                for source in retrieval_sources + citation_sources
                if str(source.get("doc_id", "")) == doc_id and str(source.get("page", "")).isdigit()
            }
            if pages and not (pages & found_pages):
                failures.append(f"facet:{facet_id}:missing_page:{doc_id}:{sorted(pages)}")

        terms_any = [str(term).lower() for term in facet.get("search_terms_any", []) or []]
        if terms_any and not any(term in search_text for term in terms_any):
            failures.append(f"facet:{facet_id}:missing_search_term")

    return failures


def _excluded_doc_ids(result: AgentTurnResult) -> set[str]:
    return {
        str(source.get("doc_id", ""))
        for source in result.answer_audit.get("retrieval", {}).get("excluded_sources", []) or []
        if isinstance(source, dict) and source.get("doc_id")
    }


def _format_row(outcome: dict[str, Any]) -> str:
    status = "SKIP" if outcome.get("_skipped_existing") else ("PASS" if outcome["passed"] else "FAIL")
    notes = outcome["demo_point"] if outcome["passed"] else ";".join(outcome["failures"])
    critic = outcome.get("critic", {}) if isinstance(outcome.get("critic"), dict) else {}
    routing = outcome.get("routing", {}) if isinstance(outcome.get("routing"), dict) else {}
    critic_label = f"{critic.get('status', '-')}/{critic.get('score', '-')}/{critic.get('gate', '-')}"
    route_label = str(routing.get("selected_mode") or routing.get("label") or "-")
    return (
        f"{outcome['case_id']} | {status} | {outcome['persona_id']} | {route_label} | {outcome['search_count']} | "
        f"{','.join(outcome['doc_ids']) or '-'} | {','.join(outcome['excluded_doc_ids']) or '-'} | "
        f"{outcome['citation_count']} | {outcome['mode']} | {critic_label} | {notes}"
    )


def _critic_summary(result: AgentTurnResult) -> dict[str, Any]:
    critic = result.answer_audit.get("critic", {}) if isinstance(result.answer_audit, dict) else {}
    if not isinstance(critic, dict):
        return {}
    return {
        "status": critic.get("status", ""),
        "score": critic.get("credibility_score"),
        "gate": critic.get("release_gate", ""),
        "verified": critic.get("verified_citation_count"),
        "unverified": critic.get("unverified_citation_count"),
        "summary": critic.get("summary", ""),
    }


async def _close_service(service: Any) -> None:
    close = getattr(service, "close", None)
    if close is not None:
        value = close()
        if hasattr(value, "__await__"):
            await value


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
