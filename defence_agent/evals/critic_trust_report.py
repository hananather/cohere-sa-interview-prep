"""Reviewer-agent citation trust eval.

This eval isolates the Reviewer Agent from retrieval and generation. Each case
contains a synthetic answer, synthetic cited evidence, and gold citation labels.
The goal is to measure whether the reviewer catches invalid citations and gates
low-trust answers before release.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from defence_agent.critic import review_answer


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PACKAGE_ROOT / "data" / "evals" / "critic_citation_trust_eval.yaml"
DEFAULT_REPORT_JSON = PACKAGE_ROOT / "data" / "evals" / "reports" / "critic_trust_eval_report.json"
DEFAULT_REPORT_MD = PACKAGE_ROOT / "data" / "evals" / "reports" / "critic_trust_eval_report.md"

COMMAND_A_INPUT_USD_PER_1M = 2.5
COMMAND_A_OUTPUT_USD_PER_1M = 10.0

SUPPORTED_LABELS = {"supported"}
INVALID_LABELS = {"weak", "unsupported", "contradicted", "wrong_source", "unsafe_refusal_citation"}
VERDICT_TO_COLOR = {"verified": "green", "unclear": "yellow", "unverified": "red"}


async def build_critic_trust_report(
    *,
    dataset_path: Path | str = DEFAULT_DATASET,
    case_ids: Iterable[str] | None = None,
    max_cases: int | None = None,
    threshold: float = 0.8,
    delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Run the live Reviewer Agent over the gold citation trust dataset."""

    dataset = _load_dataset(Path(dataset_path))
    cases = _select_cases(dataset.get("cases", []), case_ids=case_ids, max_cases=max_cases)
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(await _run_case(case, threshold=threshold))
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
    report = _build_report(dataset=dataset, dataset_path=Path(dataset_path), rows=rows, threshold=threshold)
    report["markdown"] = critic_trust_markdown(report)
    return report


def write_critic_trust_report(report: dict[str, Any], *, json_path: Path, md_path: Path) -> None:
    """Write JSON and Markdown report artifacts."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    markdown = str(payload.pop("markdown", ""))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


def _load_dataset(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Critic trust dataset must be a YAML object: {path}")
    return data


def _select_cases(
    cases: Any,
    *,
    case_ids: Iterable[str] | None,
    max_cases: int | None,
) -> list[dict[str, Any]]:
    selected = [case for case in cases or [] if isinstance(case, dict)]
    wanted = {str(case_id) for case_id in case_ids or []}
    if wanted:
        selected = [case for case in selected if str(case.get("id", "")) in wanted]
    if max_cases is not None:
        selected = selected[: max(0, int(max_cases))]
    if not selected:
        raise ValueError("No critic-trust cases selected.")
    return selected


async def _run_case(case: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    query = str(case.get("query", "") or "")
    answer = str(case.get("answer", "") or "")
    sources = _sources(case)
    citations = _citations(case, sources=sources)
    answer_audit = _answer_audit(case, sources=sources, citations=citations)
    estimated_input = _estimated_reviewer_input(query=query, answer=answer, sources=sources, citations=citations)
    started = time.perf_counter()
    report = await review_answer(
        query=query,
        answer=answer,
        citations=citations,
        answer_audit=answer_audit,
        sources_sent_to_answer=sources,
        threshold=threshold,
    )
    latency = time.perf_counter() - started
    estimated_output = json.dumps(report.get("raw_critic_response", {}) or {}, ensure_ascii=False)
    return _score_case(
        case,
        reviewer_report=report,
        latency_seconds=latency,
        estimated_usage=_estimated_usage(estimated_input, estimated_output),
    )


def _sources(case: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for source in case.get("sources", []) or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        source_id = str(item.get("source_id") or item.get("chunk_id") or "")
        item.setdefault("source_id", source_id)
        item.setdefault("chunk_id", source_id)
        item.setdefault("status", "approved")
        item.setdefault("source_type", "synthetic_eval")
        sources.append(item)
    return sources


def _citations(case: dict[str, Any], *, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_lookup = _source_lookup(sources)
    citations: list[dict[str, Any]] = []
    for citation in case.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        source_items = []
        for source_id in citation.get("source_ids", []) or []:
            source_items.append(_citation_source(str(source_id), source_lookup))
        citations.append(
            {
                "type": "synthetic_gold",
                "text": str(citation.get("text", "") or ""),
                "sources": source_items,
            }
        )
    return citations


def _source_lookup(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for source in sources:
        for field in ("source_id", "chunk_id", "retrieval_chunk_id", "parent_page_id"):
            value = str(source.get(field, "") or "")
            if value:
                lookup[value] = source
    return lookup


def _citation_source(source_id: str, source_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = source_lookup.get(source_id, {})
    return {
        "source_id": source_id,
        "chunk_id": source_id,
        "doc_id": str(source.get("doc_id", "") or ""),
        "title": str(source.get("title", "") or ""),
        "page": source.get("page", ""),
        "access_level": str(source.get("access_level", "") or ""),
        "language": str(source.get("language", "") or ""),
    }


def _answer_audit(
    case: dict[str, Any],
    *,
    sources: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "query": str(case.get("query", "") or ""),
        "retrieval": {
            "sources_sent_to_answer": sources,
            "authorized_sources": sources,
            "excluded_sources": list(case.get("excluded_sources", []) or []),
            "search_count": 0,
            "search_queries": [],
        },
        "generation": {
            "model": "synthetic_answer",
            "document_count": len(sources),
            "cohere_document_ids": [str(source.get("source_id", "")) for source in sources],
            "citation_mode": "synthetic_gold",
            "citation_resolution": {
                "passed": bool(citations),
                "errors": [] if citations else ["synthetic_answer_has_no_citations"],
                "warnings": [],
                "citation_count": len(citations),
            },
        },
        "citations": citations,
    }


def _estimated_reviewer_input(
    *,
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "query": query,
            "answer": answer,
            "sources_sent_to_answer": sources,
            "citations": citations,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _estimated_usage(input_text: str, output_text: str) -> dict[str, Any]:
    input_tokens = _estimate_tokens(input_text)
    output_tokens = _estimate_tokens(output_text)
    cost = (
        input_tokens * COMMAND_A_INPUT_USD_PER_1M / 1_000_000
        + output_tokens * COMMAND_A_OUTPUT_USD_PER_1M / 1_000_000
    )
    return {
        "input_chars": len(input_text),
        "output_chars": len(output_text),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 8),
        "note": "Estimated from serialized reviewer input and output at 4 chars per token. Provider billed usage is not exposed by the ADK reviewer event.",
    }


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _score_case(
    case: dict[str, Any],
    *,
    reviewer_report: dict[str, Any],
    latency_seconds: float,
    estimated_usage: dict[str, Any],
) -> dict[str, Any]:
    citation_rows = _citation_rows(case, reviewer_report)
    expected_gate = str(case.get("expected_release_gate", "") or "")
    actual_gate = str(reviewer_report.get("release_gate", "") or "")
    expected_terminal_gate = str(case.get("expected_terminal_gate_after_max_cycles", expected_gate) or expected_gate)
    terminal_gate = _terminal_gate_after_max_cycles(actual_gate)
    has_invalid_gold = any(not row["gold_supported"] for row in citation_rows)
    has_missing_citations = not list(case.get("citations", []) or []) and "without_citations" in str(case.get("id", ""))
    expects_release = expected_gate == "release"
    return {
        "case_id": str(case.get("id", "")),
        "slice": str(case.get("slice", "")),
        "query": str(case.get("query", "")),
        "expected_reviewer_value": str(case.get("expected_reviewer_value", "")),
        "expected_status": str(case.get("expected_status", "")),
        "actual_status": str(reviewer_report.get("status", "")),
        "status_passed": str(reviewer_report.get("status", "")) == str(case.get("expected_status", "")),
        "expected_release_gate": expected_gate,
        "actual_release_gate": actual_gate,
        "release_gate_passed": actual_gate == expected_gate,
        "expected_terminal_gate_after_max_cycles": expected_terminal_gate,
        "actual_terminal_gate_after_max_cycles": terminal_gate,
        "terminal_gate_passed": terminal_gate == expected_terminal_gate,
        "has_invalid_gold_citation": has_invalid_gold,
        "unsafe_release": actual_gate == "release" and not expects_release,
        "false_positive_gate": actual_gate != "release" and expects_release,
        "missing_citation_case": has_missing_citations,
        "trust_score": reviewer_report.get("trust_score", reviewer_report.get("credibility_score")),
        "trust_label": reviewer_report.get("trust_label", ""),
        "citation_trust_counts": reviewer_report.get("citation_trust_counts", {}),
        "latency_seconds": round(latency_seconds, 4),
        "estimated_usage": estimated_usage,
        "citation_rows": citation_rows,
        "reviewer_report": reviewer_report,
    }


def _terminal_gate_after_max_cycles(actual_gate: str) -> str:
    if actual_gate == "revise":
        return "human_continue_or_stop_required"
    return actual_gate


def _citation_rows(case: dict[str, Any], reviewer_report: dict[str, Any]) -> list[dict[str, Any]]:
    expected_by_index = {
        int(item.get("citation_index", 0) or 0): item
        for item in case.get("gold_citations", []) or []
        if isinstance(item, dict)
    }
    predicted_by_index = {
        int(item.get("citation_index", 0) or 0): item
        for item in reviewer_report.get("citation_reviews", []) or []
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for index, gold in sorted(expected_by_index.items()):
        predicted = predicted_by_index.get(index, {})
        fallback_verdict = _fallback_verdict_for_missing_review(reviewer_report)
        predicted_verdict = str(predicted.get("verdict") or fallback_verdict or "unreviewed")
        predicted_color = str(predicted.get("trust_color") or VERDICT_TO_COLOR.get(predicted_verdict, "unreviewed"))
        expected_verdicts = [str(item) for item in gold.get("expected_verdicts", []) or []]
        support_label = str(gold.get("support_label", "") or "")
        gold_supported = support_label in SUPPORTED_LABELS
        predicted_verified = predicted_verdict == "verified"
        rows.append(
            {
                "citation_index": index,
                "support_label": support_label,
                "gold_supported": gold_supported,
                "expected_verdicts": expected_verdicts,
                "predicted_verdict": predicted_verdict,
                "verdict_passed": predicted_verdict in expected_verdicts,
                "expected_trust_color": str(gold.get("expected_trust_color", "") or ""),
                "predicted_trust_color": predicted_color,
                "traffic_light_passed": predicted_color == str(gold.get("expected_trust_color", "") or ""),
                "predicted_verified": predicted_verified,
                "invalid_caught": (not gold_supported) and (not predicted_verified),
                "valid_released": gold_supported and predicted_verified,
                "reviewer_reason": str(predicted.get("reason", "") or ""),
                "answer_span": str(predicted.get("answer_span", "") or ""),
                "source_ids": list(predicted.get("source_ids", []) or []),
            }
        )
    return rows


def _fallback_verdict_for_missing_review(reviewer_report: dict[str, Any]) -> str:
    status = str(reviewer_report.get("status", "") or "")
    gate = str(reviewer_report.get("release_gate", "") or "")
    if status == "needs_human_review" or gate == "human_continue_or_stop_required":
        return "unverified"
    if status == "needs_revision" or gate == "revise":
        return "unverified"
    return "unreviewed"


def _build_report(
    *,
    dataset: dict[str, Any],
    dataset_path: Path,
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    citation_rows = [citation for row in rows for citation in row["citation_rows"]]
    metrics = _metrics(rows, citation_rows)
    return {
        "schema_version": "critic_trust_eval_report.v1",
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset),
        "threshold": threshold,
        "case_count": len(rows),
        "citation_label_count": len(citation_rows),
        "sample_design": _sample_design(dataset, rows),
        "business_metric_map": _business_metric_map(),
        "metrics": metrics,
        "slice_metrics": _slice_metrics(rows),
        "case_rows": rows,
        "findings": _findings(metrics, rows),
        "limitations": [
            "The corpus is synthetic and intentionally compact.",
            "The eval isolates reviewer behavior. It does not measure retrieval quality.",
            "Token usage and cost are estimated because reviewer ADK events do not expose provider billed usage.",
            "Weak or yellow citations are treated as blocking automatic release in this prototype.",
        ],
    }


def _dataset_sha256(dataset: dict[str, Any]) -> str:
    import hashlib

    normalized = []
    for case in dataset.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        normalized.append(
            {
                "id": case.get("id", ""),
                "slice": case.get("slice", ""),
                "query": case.get("query", ""),
                "answer": case.get("answer", ""),
                "expected_status": case.get("expected_status", ""),
                "expected_release_gate": case.get("expected_release_gate", ""),
                "expected_terminal_gate_after_max_cycles": case.get("expected_terminal_gate_after_max_cycles", ""),
                "citations": case.get("citations", []),
                "gold_citations": case.get("gold_citations", []),
                "sources": [
                    {
                        "source_id": source.get("source_id", ""),
                        "text": source.get("text", ""),
                        "retrieval_chunk_id": source.get("retrieval_chunk_id", ""),
                        "retrieval_chunk_text": source.get("retrieval_chunk_text", ""),
                    }
                    for source in case.get("sources", []) or []
                    if isinstance(source, dict)
                ],
            }
        )
    blob = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sample_design(dataset: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "suite": dataset.get("suite", ""),
        "description": dataset.get("description", ""),
        "business_objective": dataset.get("business_objective", ""),
        "traffic_light_contract": dataset.get("traffic_light_contract", {}),
        "slice_counts": dict(Counter(str(row.get("slice", "")) for row in rows)),
        "case_ids": [row["case_id"] for row in rows],
    }


def _metrics(rows: list[dict[str, Any]], citation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in citation_rows if row["gold_supported"]]
    invalid = [row for row in citation_rows if not row["gold_supported"]]
    predicted_green = [row for row in citation_rows if row["predicted_verified"]]
    predicted_invalid = [row for row in citation_rows if not row["predicted_verified"]]
    release_expected = [row for row in rows if row["expected_release_gate"] == "release"]
    gate_expected = [row for row in rows if row["expected_release_gate"] != "release"]
    unsafe_releases = [row for row in rows if row["unsafe_release"]]
    total_estimated_cost = sum(float(row["estimated_usage"].get("estimated_cost_usd", 0.0) or 0.0) for row in rows)
    return {
        "case_count": len(rows),
        "citation_label_count": len(citation_rows),
        "trusted_citation_precision": _safe_rate(row["gold_supported"] for row in predicted_green),
        "trusted_citation_recall": _safe_rate(row["predicted_verified"] for row in valid),
        "invalid_citation_recall": _safe_rate(not row["predicted_verified"] for row in invalid),
        "invalid_citation_precision": _safe_rate(not row["gold_supported"] for row in predicted_invalid),
        "citation_verdict_accuracy": _safe_rate(row["verdict_passed"] for row in citation_rows),
        "traffic_light_accuracy": _safe_rate(row["traffic_light_passed"] for row in citation_rows),
        "release_gate_accuracy": _safe_rate(row["release_gate_passed"] for row in rows),
        "terminal_gate_after_max_cycles_accuracy": _safe_rate(row["terminal_gate_passed"] for row in rows),
        "human_or_revision_gate_recall": _safe_rate(row["actual_release_gate"] != "release" for row in gate_expected),
        "false_positive_gate_rate": _safe_rate(row["false_positive_gate"] for row in release_expected),
        "unsafe_release_rate": _safe_rate(row["unsafe_release"] for row in rows),
        "unsafe_release_count": len(unsafe_releases),
        "mean_latency_seconds": _mean(row["latency_seconds"] for row in rows),
        "p95_latency_seconds": _p95(row["latency_seconds"] for row in rows),
        "estimated_cost_total_usd": round(total_estimated_cost, 6),
        "estimated_cost_per_query_usd": round(total_estimated_cost / len(rows), 6) if rows else None,
        "estimated_input_tokens_per_query": _mean(row["estimated_usage"].get("estimated_input_tokens") for row in rows),
        "estimated_output_tokens_per_query": _mean(row["estimated_usage"].get("estimated_output_tokens") for row in rows),
    }


def _slice_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("slice", "unknown")), []).append(row)
    metrics = []
    for slice_name, slice_rows in sorted(grouped.items()):
        citation_rows = [citation for row in slice_rows for citation in row["citation_rows"]]
        snapshot = _metrics(slice_rows, citation_rows)
        metrics.append(
            {
                "slice": slice_name,
                "case_count": len(slice_rows),
                "citation_label_count": len(citation_rows),
                "invalid_citation_recall": snapshot.get("invalid_citation_recall"),
                "trusted_citation_precision": snapshot.get("trusted_citation_precision"),
                "release_gate_accuracy": snapshot.get("release_gate_accuracy"),
                "unsafe_release_rate": snapshot.get("unsafe_release_rate"),
            }
        )
    return metrics


def _business_metric_map() -> list[dict[str, str]]:
    return [
        {
            "business_kpi": "Accuracy",
            "eval_metric": "release_gate_accuracy and trusted_citation_precision",
            "why_it_matters": "The system should only release answers whose citations directly support the claims.",
        },
        {
            "business_kpi": "Traceability",
            "eval_metric": "citation_verdict_accuracy and traffic_light_accuracy",
            "why_it_matters": "A planner can inspect which cited claims are high trust, uncertain, or low trust.",
        },
        {
            "business_kpi": "Risk reduction",
            "eval_metric": "invalid_citation_recall and unsafe_release_rate",
            "why_it_matters": "The reviewer should catch bad citations before they become trusted output.",
        },
        {
            "business_kpi": "Human workload",
            "eval_metric": "false_positive_gate_rate and human_or_revision_gate_recall",
            "why_it_matters": "The trust layer should escalate the right cases without blocking clean answers.",
        },
        {
            "business_kpi": "Operational efficiency",
            "eval_metric": "latency_seconds and estimated_cost_per_query_usd",
            "why_it_matters": "Multi-agent review earns its cost only when the trust gain justifies extra latency.",
        },
    ]


def _findings(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    findings = []
    invalid_recall = metrics.get("invalid_citation_recall")
    unsafe_release_rate = metrics.get("unsafe_release_rate")
    if invalid_recall is not None:
        findings.append(f"Invalid citation recall: {_fmt(invalid_recall)}.")
    if unsafe_release_rate is not None:
        findings.append(f"Unsafe release rate: {_fmt(unsafe_release_rate)}.")
    traffic_light = metrics.get("traffic_light_accuracy")
    if traffic_light is not None:
        findings.append(
            f"Traffic-light accuracy: {_fmt(traffic_light)}. Weak/yellow citations may be conservatively marked red."
        )
    terminal = metrics.get("terminal_gate_after_max_cycles_accuracy")
    if terminal is not None:
        findings.append(f"After max review cycles, terminal gate accuracy: {_fmt(terminal)}.")
    weak_examples = [
        row["case_id"]
        for row in rows
        if row["expected_release_gate"] != "release" and row["actual_release_gate"] != "release"
    ][:4]
    if weak_examples:
        findings.append("Reviewer blocked or escalated low-trust cases: " + ", ".join(weak_examples) + ".")
    return findings


def critic_trust_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Critic Trust Eval Report",
        "",
        "## Executive Readout",
        "- This is the primary eval for the Reviewer Agent claim.",
        "- The eval isolates citation review from retrieval so precision and recall are meaningful.",
        "- The product behavior being tested is strict: unclear or invalid citations block automatic release.",
        "- After the review budget is exhausted, unresolved revision requests become low-trust human review cases.",
        "",
        "## Inputs",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Dataset SHA-256: `{report['dataset_sha256']}`",
        f"- Cases: {report['case_count']}",
        f"- Citation labels: {report['citation_label_count']}",
        f"- Reviewer threshold: {report['threshold']}",
        "",
        "## KPI Map",
        _markdown_table(
            ["Business KPI", "Eval metric", "Why it matters"],
            [
                [item["business_kpi"], item["eval_metric"], item["why_it_matters"]]
                for item in report["business_metric_map"]
            ],
        ),
        "",
        "## Headline Metrics",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["trusted_citation_precision", _fmt(metrics.get("trusted_citation_precision"))],
                ["trusted_citation_recall", _fmt(metrics.get("trusted_citation_recall"))],
                ["invalid_citation_recall", _fmt(metrics.get("invalid_citation_recall"))],
                ["invalid_citation_precision", _fmt(metrics.get("invalid_citation_precision"))],
                ["citation_verdict_accuracy", _fmt(metrics.get("citation_verdict_accuracy"))],
                ["traffic_light_accuracy", _fmt(metrics.get("traffic_light_accuracy"))],
                ["release_gate_accuracy", _fmt(metrics.get("release_gate_accuracy"))],
                ["terminal_gate_after_max_cycles_accuracy", _fmt(metrics.get("terminal_gate_after_max_cycles_accuracy"))],
                ["unsafe_release_rate", _fmt(metrics.get("unsafe_release_rate"))],
                ["false_positive_gate_rate", _fmt(metrics.get("false_positive_gate_rate"))],
                ["mean_latency_seconds", _seconds(metrics.get("mean_latency_seconds"))],
                ["p95_latency_seconds", _seconds(metrics.get("p95_latency_seconds"))],
                ["estimated_cost_per_query_usd", _usd(metrics.get("estimated_cost_per_query_usd"))],
            ],
        ),
        "",
        "## Slice Metrics",
        _markdown_table(
            [
                "Slice",
                "Cases",
                "Labels",
                "Invalid citation recall",
                "Trusted citation precision",
                "Release gate accuracy",
                "Unsafe release rate",
            ],
            [
                [
                    item["slice"],
                    item["case_count"],
                    item["citation_label_count"],
                    _fmt(item.get("invalid_citation_recall")),
                    _fmt(item.get("trusted_citation_precision")),
                    _fmt(item.get("release_gate_accuracy")),
                    _fmt(item.get("unsafe_release_rate")),
                ]
                for item in report["slice_metrics"]
            ],
        ),
        "",
        "## Case Results",
        _markdown_table(
            [
                "Case",
                "Slice",
                "Expected gate",
                "Actual gate",
                "Terminal gate",
                "Trust score",
                "Latency",
                "Value",
            ],
            [
                [
                    row["case_id"],
                    row["slice"],
                    row["expected_release_gate"],
                    row["actual_release_gate"],
                    row["actual_terminal_gate_after_max_cycles"],
                    _fmt(row.get("trust_score")),
                    _seconds(row.get("latency_seconds")),
                    row["expected_reviewer_value"],
                ]
                for row in report["case_rows"]
            ],
        ),
        "",
        "## Citation-Level Examples",
        _markdown_table(
            [
                "Case",
                "Citation",
                "Gold label",
                "Reviewer verdict",
                "Gold color",
                "Reviewer color",
                "Reason",
            ],
            _example_rows(report["case_rows"]),
        ),
        "",
        "## Findings",
    ]
    lines.extend(f"- {item}" for item in report["findings"])
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _example_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    examples: list[list[Any]] = []
    for row in rows:
        for citation in row.get("citation_rows", []) or []:
            examples.append(
                [
                    row["case_id"],
                    citation["citation_index"],
                    citation["support_label"],
                    citation["predicted_verdict"],
                    citation["expected_trust_color"],
                    citation["predicted_trust_color"],
                    _truncate(citation.get("reviewer_reason", ""), 120),
                ]
            )
            if len(examples) >= 12:
                return examples
    return examples


def _safe_rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    if not items:
        return None
    return round(sum(1 for item in items if item) / len(items), 4)


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if value not in (None, "")]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def _p95(values: Iterable[Any]) -> float | None:
    numbers = sorted(float(value) for value in values if value not in (None, ""))
    if not numbers:
        return None
    index = min(len(numbers) - 1, math.ceil(0.95 * len(numbers)) - 1)
    return round(numbers[index], 4)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}s"


def _usd(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):.6f}"


def _truncate(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 12)].rstrip() + " ...truncated"


def _markdown_table(headers: list[Any], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)
