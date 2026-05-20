"""Route-comparison eval report for agent value.

This module scores the same query distribution across Simple RAG, the Research
Agent, and Research + Reviewer. It is intentionally explicit about metric
boundaries: raw task success, retrieval/citation recall, reviewer gates, and
operational telemetry are separate claims.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from defence_agent.evals.defensible_eval_report import registry_sha256
from defence_agent.evals.harness import load_registry, summarize_transcript_runs


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PACKAGE_ROOT / "data" / "evals" / "retrieval_reviewer_live_20.yaml"
DEFAULT_SIMPLE_DIR = PACKAGE_ROOT / "data" / "transcripts" / "route_comparison_live_20260520_074144_simple"
DEFAULT_AGENTIC_DIR = PACKAGE_ROOT / "data" / "transcripts" / "route_comparison_live_20260520_074144_agentic"
DEFAULT_REVIEWED_DIR = PACKAGE_ROOT / "data" / "transcripts" / "route_comparison_live_20260520_074144_reviewed_final"
DEFAULT_REPORT_JSON = PACKAGE_ROOT / "data" / "evals" / "reports" / "agent_value_route_comparison_report.json"
DEFAULT_REPORT_MD = PACKAGE_ROOT / "data" / "evals" / "reports" / "agent_value_route_comparison_report.md"

COMMAND_A_INPUT_USD_PER_1M = 2.5
COMMAND_A_OUTPUT_USD_PER_1M = 10.0

ROUTE_ORDER = ("simple_rag", "agentic_rag", "reviewed_agent")
ROUTE_LABELS = {
    "simple_rag": "Simple RAG",
    "agentic_rag": "Research Agent",
    "reviewed_agent": "Research + Reviewer",
}

QUALITY_METRICS = (
    "end_to_end_success",
    "answerability_accuracy",
    "retrieval_recall",
    "retrieval_precision",
    "citation_recall",
    "citation_doc_recall",
    "citation_doc_precision",
    "citation_source_precision_proxy",
    "citation_support_precision_lexical",
    "generation_fact_recall",
    "forbidden_fact_absence",
)


def build_route_comparison_report(
    *,
    registry_path: Path | str = DEFAULT_REGISTRY,
    simple_dir: Path | str = DEFAULT_SIMPLE_DIR,
    agentic_dir: Path | str = DEFAULT_AGENTIC_DIR,
    reviewed_dir: Path | str = DEFAULT_REVIEWED_DIR,
    model_label: str = "Command A",
    model_id: str = "command-a-03-2025",
    input_usd_per_1m_tokens: float = COMMAND_A_INPUT_USD_PER_1M,
    output_usd_per_1m_tokens: float = COMMAND_A_OUTPUT_USD_PER_1M,
) -> dict[str, Any]:
    """Build a report comparing value added by the agentic routes."""

    route_dirs = {
        "simple_rag": Path(simple_dir),
        "agentic_rag": Path(agentic_dir),
        "reviewed_agent": Path(reviewed_dir),
    }
    registry = load_registry(registry_path)
    scored = summarize_transcript_runs(
        registry_path=registry_path,
        transcript_paths=[route_dirs[route] for route in ROUTE_ORDER],
        include_citation_support=True,
    )
    _attach_declared_slices(scored.get("rows", []), _case_slices(registry))
    route_by_run = {path.name: route for route, path in route_dirs.items()}
    rows_by_route = _rows_by_route(scored.get("rows", []), route_by_run)
    scorecards = {
        route: _quality_scorecard(rows_by_route.get(route, []))
        for route in ROUTE_ORDER
    }
    report = {
        "schema_version": "route_comparison_report.v1",
        "model_label": model_label,
        "model_id": model_id,
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha256(registry),
        "route_dirs": {route: str(path) for route, path in route_dirs.items()},
        "sample_design": _sample_design(registry),
        "scored_case_counts": {route: scorecards[route]["n"] for route in ROUTE_ORDER},
        "scorecards": scorecards,
        "lifts": _route_lifts(scorecards),
        "by_slice": _by_slice(rows_by_route),
        "case_matrix": _case_matrix(rows_by_route),
        "operational_metrics": _operational_metrics(
            route_dirs,
            model_id=model_id,
            input_usd_per_1m_tokens=input_usd_per_1m_tokens,
            output_usd_per_1m_tokens=output_usd_per_1m_tokens,
        ),
        "orchestration_contract": _orchestration_contract(route_dirs),
        "reviewer_metrics": _reviewer_metrics(route_dirs["reviewed_agent"]),
        "citation_granularity_diagnostics": _citation_granularity_diagnostics(route_dirs),
        "findings": _findings(scorecards, route_dirs),
        "limitations": _limitations(
            registry_case_count=len(registry.get("cases", []) or []),
            scored_case_count=max((scorecards[route]["n"] for route in ROUTE_ORDER), default=0),
        ),
        "base_harness_report": scored,
    }
    report["markdown"] = route_comparison_markdown(report)
    return report


def write_route_comparison_report(report: dict[str, Any], *, json_path: Path, md_path: Path) -> None:
    """Write JSON and Markdown report artifacts."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    markdown = str(payload.pop("markdown", ""))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


def route_comparison_markdown(report: dict[str, Any]) -> str:
    """Render a presentation-ready Markdown report."""

    scorecards = report["scorecards"]
    ops = report["operational_metrics"]
    reviewer = report["reviewer_metrics"]
    sample_count = report["sample_design"]["case_count"]
    scored_counts = report.get("scored_case_counts", {})
    scored_count_label = ", ".join(
        f"{ROUTE_LABELS[route]}={scored_counts.get(route, 0)}"
        for route in ROUTE_ORDER
    )
    model_label = report.get("model_label", "Command A")
    model_id = report.get("model_id", "")
    lines = [
        f"# Agent Value Route Comparison Report: {model_label}",
        "",
        "## Executive Readout",
        f"- This report scores `{model_id}` against transcripts from a {sample_count}-case route-comparison registry.",
        f"- Scored transcript outcomes by route: {scored_count_label}.",
        "- The Research Agent measures evidence-acquisition lift over Simple RAG.",
        "- The Reviewer Agent measures release-control behavior and citation-support gating.",
        "- The reviewer caught or gated some bad outputs, which is traceability value, not the same as retrieval recall.",
        "- Treat this as one live run per route, not a statistically powered benchmark.",
        "",
        "## Inputs",
        f"- Model: `{model_label}` (`{model_id}`)",
        "- Orchestration contract:",
    ]
    lines.extend(
        f"  - {ROUTE_LABELS[route]}: `{item['route_implementation_id']}`"
        for route, item in report["orchestration_contract"].items()
    )
    lines.extend(
        [
        f"- Registry: `{report['registry_path']}`",
        f"- Registry SHA-256: `{report['registry_sha256']}`",
        "- Route transcript dirs:",
        ]
    )
    lines.extend(f"  - {ROUTE_LABELS[route]}: `{path}`" for route, path in report["route_dirs"].items())
    lines.extend(
        [
            "",
            "## Sample Design",
            _markdown_table(
                ["Slice", "Cases"],
                [[item["slice"], item["count"]] for item in report["sample_design"]["slice_counts"]],
            ),
            "",
            "## Sample Groups",
            _markdown_table(
                ["Group", "Cases"],
                [[item["group"], item["count"]] for item in report["sample_design"]["group_counts"]],
            ),
            "",
            "## Route Scorecard",
            _markdown_table(
                [
                    "Route",
                    "N",
                    "Task success",
                    "Answerability",
                    "Retrieval R",
                    "Retrieval P",
                    "Citation R",
                    "Citation doc R",
                    "Citation doc P",
                    "Citation source P",
                    "Lexical support P",
                ],
                [
                    [
                        ROUTE_LABELS[route],
                        scorecards[route]["n"],
                        _fmt(scorecards[route].get("end_to_end_success")),
                        _fmt(scorecards[route].get("answerability_accuracy")),
                        _fmt(scorecards[route].get("retrieval_recall")),
                        _fmt(scorecards[route].get("retrieval_precision")),
                        _fmt(scorecards[route].get("citation_recall")),
                        _fmt(scorecards[route].get("citation_doc_recall")),
                        _fmt(scorecards[route].get("citation_doc_precision")),
                        _fmt(scorecards[route].get("citation_source_precision_proxy")),
                        _fmt(scorecards[route].get("citation_support_precision_lexical")),
                    ]
                    for route in ROUTE_ORDER
                ],
            ),
            "",
            "## Lift",
            _markdown_table(
                ["Comparison", "Task success", "Answerability", "Retrieval R", "Citation R", "Citation doc R", "Citation doc P"],
                [
                    [
                        item["comparison"],
                        _fmt_signed(item.get("end_to_end_success")),
                        _fmt_signed(item.get("answerability_accuracy")),
                        _fmt_signed(item.get("retrieval_recall")),
                        _fmt_signed(item.get("citation_recall")),
                        _fmt_signed(item.get("citation_doc_recall")),
                        _fmt_signed(item.get("citation_doc_precision")),
                    ]
                    for item in report["lifts"]
                ],
            ),
            "",
            "## Operational Metrics",
            _markdown_table(
                ["Route", "Mean latency", "Median latency", "P95 latency", "Mean searches", "Mean cited spans", "Captured generation cost/query"],
                [
                    [
                        ROUTE_LABELS[route],
                        _seconds(ops[route]["latency_seconds"].get("mean")),
                        _seconds(ops[route]["latency_seconds"].get("median")),
                        _seconds(ops[route]["latency_seconds"].get("p95")),
                        _fmt(ops[route].get("mean_search_count")),
                        _fmt(ops[route].get("mean_citation_count")),
                        _usd(ops[route].get("captured_generation_cost_mean_usd")),
                    ]
                    for route in ROUTE_ORDER
                ],
            ),
            "",
            "Cost note: captured generation cost uses recorded billed tokens for the final grounded generation call only. It excludes Embed, Rerank, ADK planner calls, and reviewer-call billed usage because those are not fully recorded in these transcripts.",
            "",
            "## Reviewer Metrics",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["reviewed_cases", reviewer["case_count"]],
                    ["release_gate_rate", _fmt(reviewer.get("release_gate_rate"))],
                    ["human_gate_rate", _fmt(reviewer.get("human_gate_rate"))],
                    ["reviewer_estimated_citation_precision", _fmt(reviewer.get("reviewer_estimated_citation_precision"))],
                    ["reviewed_citation_coverage", _fmt(reviewer.get("reviewed_citation_coverage"))],
                    ["bad_or_failed_cases_gated", reviewer.get("failed_cases_human_gated")],
                    ["failed_cases_released", reviewer.get("failed_cases_released")],
                ],
            ),
            "",
            "## Citation Granularity Diagnostics",
            _markdown_table(
                [
                    "Route",
                    "Retrieval strategies",
                    "Sources sent",
                    "Child chunks promoted",
                    "Citation source refs",
                    "Citation child refs",
                    "Reviewer granularity risk",
                ],
                [
                    [
                        ROUTE_LABELS[route],
                        _dict_counts(diag["chunk_strategy_counts"]),
                        diag["sources_sent_to_answer_count"],
                        diag["child_chunk_promoted_source_count"],
                        diag["citation_source_ref_count"],
                        diag["citation_child_chunk_ref_count"],
                        diag["reviewer_granularity_risk"],
                    ]
                    for route, diag in report["citation_granularity_diagnostics"].items()
                ],
            ),
            "",
            "Current interpretation: this saved route run is page-level throughout. The mismatch risk is low here, but it becomes important when `DEFENCE_AGENT_CHUNK_STRATEGY=windowed` because retrieval may rank child chunks while generation and citations are promoted back to parent pages.",
            "",
            "## Case Movement",
            _markdown_table(
                ["Case", "Slice", "Simple", "Research Agent", "Research + Reviewer"],
                [
                    [
                        item["case_id"],
                        item["slice"],
                        _pass_label(item["simple_rag"]),
                        _pass_label(item["agentic_rag"]),
                        _pass_label(item["reviewed_agent"]),
                    ]
                    for item in report["case_matrix"]
                ],
            ),
            "",
            "## Findings",
        ]
    )
    lines.extend(f"- {finding}" for finding in report["findings"])
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _sample_design(registry: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in registry.get("cases", []) or [] if isinstance(case, dict)]
    counts = Counter(str(case.get("eval_slice", "unknown")) for case in cases)
    group_counts = Counter(str(case.get("model_upgrade_group", "canonical_20")) for case in cases)
    return {
        "case_count": len(cases),
        "slice_counts": [{"slice": key, "count": counts[key]} for key in sorted(counts)],
        "group_counts": [{"group": key, "count": group_counts[key]} for key in sorted(group_counts)],
    }


def _case_slices(registry: dict[str, Any]) -> dict[str, str]:
    return {
        str(case.get("id", "")): str(case.get("eval_slice", "unknown"))
        for case in registry.get("cases", []) or []
        if isinstance(case, dict)
    }


def _attach_declared_slices(rows: list[dict[str, Any]], case_slices: dict[str, str]) -> None:
    for row in rows:
        case_id = str(row.get("case_id", ""))
        row["declared_slice"] = case_slices.get(case_id) or str(row.get("slice", "unknown"))


def _rows_by_route(rows: list[dict[str, Any]], route_by_run: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        route = route_by_run.get(str(row.get("transcript_run", "")))
        if route:
            grouped[route].append(row)
    return grouped


def _quality_scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    item: dict[str, Any] = {"n": len(rows)}
    for metric in QUALITY_METRICS:
        item[metric] = _mean(row.get(metric) for row in rows)
    return item


def _route_lifts(scorecards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _delta("Research Agent - Simple RAG", scorecards["agentic_rag"], scorecards["simple_rag"]),
        _delta("Research + Reviewer - Research Agent", scorecards["reviewed_agent"], scorecards["agentic_rag"]),
        _delta("Research + Reviewer - Simple RAG", scorecards["reviewed_agent"], scorecards["simple_rag"]),
    ]


def _delta(label: str, after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    item = {"comparison": label}
    for metric in QUALITY_METRICS:
        item[metric] = _sub(after.get(metric), before.get(metric))
    return item


def _by_slice(rows_by_route: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for route in ROUTE_ORDER:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_by_route.get(route, []):
            grouped[str(row.get("declared_slice") or row.get("slice", "unknown"))].append(row)
        for slice_name, slice_rows in sorted(grouped.items()):
            score = _quality_scorecard(slice_rows)
            rows.append({"route": route, "slice": slice_name, **score})
    return rows


def _case_matrix(rows_by_route: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for route, rows in rows_by_route.items():
        for row in rows:
            case_id = str(row.get("case_id", ""))
            item = cases.setdefault(
                case_id,
                {"case_id": case_id, "slice": str(row.get("declared_slice") or row.get("slice", ""))},
            )
            item[route] = {
                "passed": bool(row.get("passed")),
                "failures": str(row.get("failures", "")),
            }
    return [cases[key] for key in sorted(cases)]


def _operational_metrics(
    route_dirs: dict[str, Path],
    *,
    model_id: str,
    input_usd_per_1m_tokens: float,
    output_usd_per_1m_tokens: float,
) -> dict[str, Any]:
    return {
        route: _operational_route(
            path,
            model_id=model_id,
            input_usd_per_1m_tokens=input_usd_per_1m_tokens,
            output_usd_per_1m_tokens=output_usd_per_1m_tokens,
        )
        for route, path in route_dirs.items()
    }


def _orchestration_contract(route_dirs: dict[str, Path]) -> dict[str, Any]:
    return {route: _orchestration_route(path, route=route) for route, path in route_dirs.items()}


def _orchestration_route(path: Path, *, route: str) -> dict[str, Any]:
    outcomes = _load_outcomes(path)
    metadatas = [
        outcome.get("eval_metadata")
        for outcome in outcomes
        if isinstance(outcome.get("eval_metadata"), dict)
    ]
    if not metadatas:
        return {
            "status": "missing",
            "route": route,
            "harness_id": "unknown",
            "route_implementation_id": "unknown",
            "case_count": len(outcomes),
            "metadata_rows": 0,
        }
    harness_ids = sorted({str(item.get("harness_id", "")) for item in metadatas})
    implementation_ids = sorted({str(item.get("route_implementation_id", "")) for item in metadatas})
    chunk_strategies = sorted({str(item.get("chunk_strategy", "")) for item in metadatas})
    retrieval_modes = sorted({str(item.get("retrieval_mode", "")) for item in metadatas})
    return {
        "status": "complete" if len(metadatas) == len(outcomes) else "partial",
        "route": route,
        "harness_id": harness_ids[0] if len(harness_ids) == 1 else "mixed",
        "route_implementation_id": implementation_ids[0] if len(implementation_ids) == 1 else "mixed",
        "chunk_strategy": chunk_strategies[0] if len(chunk_strategies) == 1 else "mixed",
        "retrieval_mode": retrieval_modes[0] if len(retrieval_modes) == 1 else "mixed",
        "case_count": len(outcomes),
        "metadata_rows": len(metadatas),
    }


def _citation_granularity_diagnostics(route_dirs: dict[str, Path]) -> dict[str, Any]:
    return {route: _citation_granularity_route(path) for route, path in route_dirs.items()}


def _citation_granularity_route(path: Path) -> dict[str, Any]:
    outcomes = _load_outcomes(path)
    strategy_counts: Counter[str] = Counter()
    sources_sent = 0
    promoted_sources = 0
    citation_refs = 0
    citation_child_refs = 0
    source_aliases: set[str] = set()
    child_aliases: set[str] = set()
    for outcome in outcomes:
        final = outcome.get("final_answer_audit", {}) if isinstance(outcome.get("final_answer_audit"), dict) else {}
        retrieval = final.get("retrieval", {}) if isinstance(final.get("retrieval"), dict) else {}
        for source in retrieval.get("sources_sent_to_answer", []) or []:
            if not isinstance(source, dict):
                continue
            sources_sent += 1
            strategy = str(source.get("chunk_strategy", "") or "unknown")
            strategy_counts[strategy] += 1
            chunk_id = str(source.get("chunk_id", "") or "")
            retrieval_chunk_id = str(source.get("retrieval_chunk_id", "") or "")
            if retrieval_chunk_id and retrieval_chunk_id != chunk_id:
                promoted_sources += 1
                child_aliases.add(retrieval_chunk_id)
            source_aliases.update(alias for alias in _source_aliases(source) if alias)
        for citation in final.get("citations", []) or []:
            if not isinstance(citation, dict):
                continue
            for source in citation.get("sources", []) or []:
                if not isinstance(source, dict):
                    continue
                citation_refs += 1
                source_id = str(source.get("source_id") or source.get("chunk_id") or "")
                if source_id in child_aliases or "_chunk_" in source_id:
                    citation_child_refs += 1
    if promoted_sources and not citation_child_refs:
        risk = "medium: retrieval child chunks are promoted to page-level citations"
    elif citation_child_refs:
        risk = "medium: citations include child-chunk refs; reviewer must resolve child evidence"
    else:
        risk = "low: page-level retrieval and page-level citations"
    return {
        "chunk_strategy_counts": dict(sorted(strategy_counts.items())),
        "sources_sent_to_answer_count": sources_sent,
        "child_chunk_promoted_source_count": promoted_sources,
        "citation_source_ref_count": citation_refs,
        "citation_child_chunk_ref_count": citation_child_refs,
        "source_alias_count": len(source_aliases),
        "reviewer_granularity_risk": risk,
    }


def _source_aliases(source: dict[str, Any]) -> set[str]:
    aliases = {
        str(source.get("source_id", "") or ""),
        str(source.get("chunk_id", "") or ""),
        str(source.get("parent_page_id", "") or ""),
        str(source.get("retrieval_chunk_id", "") or ""),
        str(source.get("citation_id", "") or ""),
    }
    doc_id = str(source.get("doc_id", "") or "")
    page = str(source.get("page", "") or "")
    if doc_id and page:
        try:
            page = f"{int(page):03d}"
        except ValueError:
            pass
        aliases.add(f"{doc_id}_page_{page}")
    return {alias for alias in aliases if alias}


def _operational_route(
    path: Path,
    *,
    model_id: str,
    input_usd_per_1m_tokens: float,
    output_usd_per_1m_tokens: float,
) -> dict[str, Any]:
    outcomes = _load_outcomes(path)
    costs = [
        _captured_generation_cost_usd(
            outcome,
            input_usd_per_1m_tokens=input_usd_per_1m_tokens,
            output_usd_per_1m_tokens=output_usd_per_1m_tokens,
        )
        for outcome in outcomes
    ]
    recorded_costs = [cost for cost in costs if cost is not None]
    total_cost = round(sum(recorded_costs), 6)
    billed = [_billed_units(outcome) for outcome in outcomes]
    return {
        "n": len(outcomes),
        "latency_seconds": _distribution(outcome.get("latency_seconds") for outcome in outcomes),
        "mean_search_count": _mean(outcome.get("search_count") for outcome in outcomes),
        "mean_citation_count": _mean(outcome.get("citation_count") for outcome in outcomes),
        "captured_generation_cost_total_usd": total_cost,
        "captured_generation_cost_mean_usd": round(total_cost / len(outcomes), 6) if outcomes else None,
        "captured_generation_cost_recorded_rows": len(recorded_costs),
        "billed_input_tokens": sum(int(item.get("input_tokens") or 0) for item in billed),
        "billed_output_tokens": sum(int(item.get("output_tokens") or 0) for item in billed),
        "pricing_assumption": {
            "model_id": model_id,
            "input_usd_per_1m_tokens": input_usd_per_1m_tokens,
            "output_usd_per_1m_tokens": output_usd_per_1m_tokens,
            "source": "https://docs.cohere.com/docs/command-a",
            "note": "Use as an estimate when public Command A Plus pricing is not separately listed.",
        },
    }


def _reviewer_metrics(path: Path) -> dict[str, Any]:
    outcomes = _load_outcomes(path)
    rows = []
    for outcome in outcomes:
        reviewer = _reviewer_report(outcome)
        rows.append(
            {
                "case_id": outcome.get("case_id", ""),
                "passed": bool(outcome.get("passed")),
                "status": reviewer.get("status"),
                "release_gate": reviewer.get("release_gate"),
                "credibility_score": reviewer.get("credibility_score"),
                "verified_citation_count": int(reviewer.get("verified_citation_count") or 0),
                "total_citation_count": int(reviewer.get("total_citation_count") or 0),
                "answer_citation_count": int(reviewer.get("answer_citation_count") or 0),
                "unreviewed_citation_count": int(reviewer.get("unreviewed_citation_count") or 0),
                "failures": outcome.get("failures", []),
            }
        )
    total_reviewed = sum(row["total_citation_count"] for row in rows)
    total_verified = sum(row["verified_citation_count"] for row in rows)
    total_answer_citations = sum(row["answer_citation_count"] for row in rows)
    human_gated = [row for row in rows if row["release_gate"] == "human_continue_or_stop_required"]
    released_failures = [row for row in rows if not row["passed"] and row["release_gate"] == "release"]
    return {
        "case_count": len(rows),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "release_gate_counts": dict(Counter(str(row["release_gate"]) for row in rows)),
        "release_gate_rate": _rate(row["release_gate"] == "release" for row in rows),
        "human_gate_rate": _rate(row["release_gate"] == "human_continue_or_stop_required" for row in rows),
        "reviewer_estimated_citation_precision": (
            round(total_verified / total_reviewed, 4) if total_reviewed else None
        ),
        "reviewed_citation_coverage": (
            round(total_reviewed / total_answer_citations, 4) if total_answer_citations else None
        ),
        "answer_citation_count": total_answer_citations,
        "reviewed_citation_count": total_reviewed,
        "unreviewed_citation_count": sum(row["unreviewed_citation_count"] for row in rows),
        "failed_cases_human_gated": len([row for row in human_gated if not row["passed"]]),
        "passed_cases_human_gated": len([row for row in human_gated if row["passed"]]),
        "failed_cases_released": len(released_failures),
        "human_gated_cases": [row["case_id"] for row in human_gated],
        "released_failure_cases": [row["case_id"] for row in released_failures],
        "rows": rows,
    }


def _findings(scorecards: dict[str, dict[str, Any]], route_dirs: dict[str, Path]) -> list[str]:
    agentic_lift = _sub(
        scorecards["agentic_rag"].get("end_to_end_success"),
        scorecards["simple_rag"].get("end_to_end_success"),
    )
    citation_doc_lift = _sub(
        scorecards["agentic_rag"].get("citation_doc_recall"),
        scorecards["simple_rag"].get("citation_doc_recall"),
    )
    reviewed_lift = _sub(
        scorecards["reviewed_agent"].get("end_to_end_success"),
        scorecards["agentic_rag"].get("end_to_end_success"),
    )
    reviewer = _reviewer_metrics(route_dirs["reviewed_agent"])
    granularity = _citation_granularity_diagnostics(route_dirs)
    granularity_risks = {
        route: item["reviewer_granularity_risk"]
        for route, item in granularity.items()
    }
    return [
        f"Research Agent raw task success lift is {_fmt_signed(agentic_lift)} over Simple RAG.",
        f"Research Agent citation document recall lift is {_fmt_signed(citation_doc_lift)} over Simple RAG.",
        f"Research + Reviewer raw task success lift versus Research Agent is {_fmt_signed(reviewed_lift)} on this single live run.",
        (
            "Reviewer Agent gated "
            f"{reviewer['failed_cases_human_gated']} failed cases and {reviewer['passed_cases_human_gated']} passing case, "
            "which is a release-control benefit rather than a retrieval-recall benefit."
        ),
        (
            "The reviewer still released "
            f"{reviewer['failed_cases_released']} failed cases, so it is not yet a complete safety net."
        ),
        "Citation granularity for this saved run: "
        + "; ".join(f"{ROUTE_LABELS[route]}={risk}" for route, risk in granularity_risks.items())
        + ".",
        "Citation precision remains unlabeled semantic precision in this route run; lexical support is only a deterministic proxy.",
    ]


def _limitations(*, registry_case_count: int, scored_case_count: int) -> list[str]:
    return [
        f"This is one live run per route over {scored_case_count} scored cases from a {registry_case_count}-case registry, so route variance is visible and should not be oversold.",
        "The dataset is synthetic and prototype-sized, but it now includes hard, impossible, bilingual, mixed-format, and reviewer-catch cases.",
        "Citation precision is not a full human-labeled semantic metric in this report; citation source precision and lexical support are proxies.",
        "Reviewer precision is estimated over reviewed citation spans, not all answer citations, because the reviewer has a citation cap.",
        "Captured generation cost is not full route cost because planner, reviewer, Embed, and Rerank billing telemetry is incomplete.",
        "This route comparison does not isolate rerank uplift because all three routes use the same live hybrid retrieval stack.",
    ]


def _load_outcomes(path: Path) -> list[dict[str, Any]]:
    outcomes = []
    for file_path in sorted(path.glob("*.json")):
        outcomes.append(json.loads(file_path.read_text(encoding="utf-8")))
    return outcomes


def _reviewer_report(outcome: dict[str, Any]) -> dict[str, Any]:
    final = outcome.get("final_answer_audit", {}) if isinstance(outcome.get("final_answer_audit"), dict) else {}
    reviewer = final.get("reviewer") if isinstance(final.get("reviewer"), dict) else None
    critic = outcome.get("critic") if isinstance(outcome.get("critic"), dict) else None
    return reviewer or critic or {}


def _captured_generation_cost_usd(
    outcome: dict[str, Any],
    *,
    input_usd_per_1m_tokens: float,
    output_usd_per_1m_tokens: float,
) -> float | None:
    billed = _billed_units(outcome)
    if not billed:
        return None
    input_tokens = float(billed.get("input_tokens") or 0)
    output_tokens = float(billed.get("output_tokens") or 0)
    return round(
        (input_tokens * input_usd_per_1m_tokens / 1_000_000)
        + (output_tokens * output_usd_per_1m_tokens / 1_000_000),
        8,
    )


def _billed_units(outcome: dict[str, Any]) -> dict[str, Any]:
    operational = outcome.get("operational_metrics", {}) if isinstance(outcome.get("operational_metrics"), dict) else {}
    usage = operational.get("token_usage") if isinstance(operational.get("token_usage"), dict) else {}
    billed = usage.get("billed_units") if isinstance(usage.get("billed_units"), dict) else {}
    if billed:
        return billed
    final = outcome.get("final_answer_audit", {}) if isinstance(outcome.get("final_answer_audit"), dict) else {}
    generation = final.get("generation", {}) if isinstance(final.get("generation"), dict) else {}
    usage = generation.get("usage", {}) if isinstance(generation.get("usage"), dict) else {}
    return usage.get("billed_units", {}) if isinstance(usage.get("billed_units"), dict) else {}


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    numeric = sorted(float(value) for value in values if isinstance(value, int | float))
    if not numeric:
        return {"mean": None, "median": None, "p95": None, "max": None}
    p95_index = max(0, ceil(len(numeric) * 0.95) - 1)
    return {
        "mean": round(mean(numeric), 4),
        "median": round(median(numeric), 4),
        "p95": round(numeric[p95_index], 4),
        "max": round(max(numeric), 4),
    }


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return round(mean(numeric), 4) if numeric else None


def _rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    return round(sum(1 for item in items if item) / len(items), 4) if items else None


def _sub(after: Any, before: Any) -> float | None:
    if not isinstance(after, int | float) or not isinstance(before, int | float):
        return None
    return round(float(after) - float(before), 4)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_signed(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int | float):
        sign = "+" if value > 0 else ""
        return f"{sign}{float(value):.3f}"
    return str(value)


def _seconds(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}s"


def _usd(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):.4f}"


def _dict_counts(value: dict[str, Any]) -> str:
    if not value:
        return "n/a"
    return ", ".join(f"{key}:{count}" for key, count in sorted(value.items()))


def _pass_label(value: dict[str, Any] | None) -> str:
    if not value:
        return "n/a"
    if value.get("passed"):
        return "PASS"
    failures = str(value.get("failures", "") or "FAIL")
    return f"FAIL: {failures}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])
