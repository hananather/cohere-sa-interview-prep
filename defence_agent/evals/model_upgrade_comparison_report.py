"""Model-upgrade comparison report for Command A Plus evals."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from defence_agent.evals.harness import EvalHarnessError, load_registry
from defence_agent.evals.route_comparison_report import QUALITY_METRICS, ROUTE_LABELS, ROUTE_ORDER


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT_REPORT_JSON = (
    PACKAGE_ROOT / "data" / "evals" / "reports" / "command_a_model_upgrade_route_report.json"
)
DEFAULT_PLUS_REPORT_JSON = (
    PACKAGE_ROOT / "data" / "evals" / "reports" / "command_a_plus_model_upgrade_route_report.json"
)
DEFAULT_COMPARISON_JSON = (
    PACKAGE_ROOT / "data" / "evals" / "reports" / "command_a_plus_model_upgrade_comparison_report.json"
)
DEFAULT_COMPARISON_MD = (
    PACKAGE_ROOT / "data" / "evals" / "reports" / "command_a_plus_model_upgrade_comparison_report.md"
)

GROUP_ORDER = ("combined_24", "canonical_20", "challenge_4")
GROUP_LABELS = {
    "combined_24": "Combined scored",
    "canonical_20": "Canonical scored",
    "challenge_4": "Challenge scored",
}


def build_model_upgrade_comparison_report(
    *,
    current_report_path: Path | str = DEFAULT_CURRENT_REPORT_JSON,
    plus_report_path: Path | str = DEFAULT_PLUS_REPORT_JSON,
    current_key: str = "current",
    plus_key: str = "command_a_plus",
) -> dict[str, Any]:
    """Compare two route-comparison reports generated from the same registry."""

    current = _load_report(current_report_path)
    plus = _load_report(plus_report_path)
    if current.get("registry_sha256") != plus.get("registry_sha256"):
        raise EvalHarnessError("Model comparison requires both reports to use the same registry SHA-256.")

    registry = load_registry(current["registry_path"])
    case_groups = _case_groups(registry)
    case_slices = _case_slices(registry)
    model_reports = {current_key: current, plus_key: plus}
    model_rows = {
        model_key: _report_rows(report, case_groups=case_groups, case_slices=case_slices)
        for model_key, report in model_reports.items()
    }
    scorecards = {
        model_key: _scorecards_by_group(rows)
        for model_key, rows in model_rows.items()
    }
    case_count_audit = _case_count_audit(model_rows, registry)
    report = {
        "schema_version": "model_upgrade_comparison_report.v1",
        "registry_path": current["registry_path"],
        "registry_sha256": current["registry_sha256"],
        "models": {
            current_key: _model_descriptor(current, current_report_path),
            plus_key: _model_descriptor(plus, plus_report_path),
        },
        "sample_design": _sample_design(registry),
        "case_count_audit": case_count_audit,
        "comparability": _comparability(model_reports),
        "scorecards": scorecards,
        "deltas": _model_deltas(scorecards, before_key=current_key, after_key=plus_key),
        "bilingual_performance": _slice_scorecards(model_rows, "bilingual"),
        "operational_metrics": _operational_rows(model_reports),
        "reviewer_metrics": _reviewer_rows(model_reports),
        "case_movement": _case_movement(model_rows, before_key=current_key, after_key=plus_key),
        "presentation_gate": _presentation_gate(
            scorecards,
            model_reports,
            case_count_audit=case_count_audit,
            before_key=current_key,
            after_key=plus_key,
        ),
        "limitations": _limitations(),
    }
    report["markdown"] = model_upgrade_comparison_markdown(report, before_key=current_key, after_key=plus_key)
    return report


def write_model_upgrade_comparison_report(
    report: dict[str, Any],
    *,
    json_path: Path | str = DEFAULT_COMPARISON_JSON,
    md_path: Path | str = DEFAULT_COMPARISON_MD,
) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    markdown = str(payload.pop("markdown", ""))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


def model_upgrade_comparison_markdown(
    report: dict[str, Any],
    *,
    before_key: str = "current",
    after_key: str = "command_a_plus",
) -> str:
    before = report["models"][before_key]
    after = report["models"][after_key]
    gate = report["presentation_gate"]
    comparability = report["comparability"]
    lines = [
        "# Command A Plus Model-Upgrade Eval Report",
        "",
        "## Executive Readout",
        f"- Compared `{before['model_id']}` against `{after['model_id']}` across Simple RAG, Research Agent, and Research + Reviewer.",
        "- Valid claim boundary: compare model deltas within the same route and same harness only.",
        "- Invalid claim boundary: do not compare old saved runs, old orchestration layers, or different agent harnesses as if they isolate model quality.",
        f"- Comparability status: `{comparability['status']}`.",
        f"- Registry SHA-256: `{report['registry_sha256']}`.",
        f"- Presentation gate: `{gate['recommendation']}`.",
    ]
    lines.extend(f"- {reason}" for reason in gate["reasons"])
    lines.extend(
        [
            "",
            "## Sample Design",
            _markdown_table(
                ["Group", "Cases"],
                [[GROUP_LABELS.get(item["group"], item["group"]), item["count"]] for item in report["sample_design"]["group_counts"]],
            ),
            "",
            _markdown_table(
                ["Slice", "Cases"],
                [[item["slice"], item["count"]] for item in report["sample_design"]["slice_counts"]],
            ),
            "",
            "## Case Count Audit",
            f"- Status: `{report['case_count_audit']['status']}`.",
            f"- Expected full case count per route/model: `{report['case_count_audit']['expected_case_count']}`.",
            f"- Matched scored case count: `{report['case_count_audit']['matched_case_count']}`.",
            "",
            _markdown_table(
                ["Model", "Route", "Scored", "Missing expected", "Unexpected"],
                [
                    [
                        row["model"],
                        ROUTE_LABELS[row["route"]],
                        row["scored_case_count"],
                        row["missing_expected_count"],
                        row["unexpected_case_count"],
                    ]
                    for row in report["case_count_audit"]["rows"]
                ],
            ),
            "",
            "## Harness Comparability",
            _markdown_table(
                ["Route", "Comparable", "Current harness", "Plus harness", "Current implementation", "Plus implementation"],
                [
                    [
                        ROUTE_LABELS[row["route"]],
                        row["comparable"],
                        row["before_harness_id"],
                        row["after_harness_id"],
                        row["before_route_implementation_id"],
                        row["after_route_implementation_id"],
                    ]
                    for row in comparability["routes"]
                ],
            ),
            "",
            "## Combined Route Scorecard",
            _markdown_table(
                [
                    "Model",
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
                ],
                [
                    _scorecard_row(model, route, report["scorecards"][model]["combined_24"][route])
                    for model in (before_key, after_key)
                    for route in ROUTE_ORDER
                ],
            ),
            "",
            "## Command A Plus Delta",
            _markdown_table(
                [
                    "Group",
                    "Route",
                    "Task success",
                    "Answerability",
                    "Retrieval R",
                    "Retrieval P",
                    "Citation R",
                    "Citation doc R",
                    "Citation doc P",
                ],
                [
                    [
                        GROUP_LABELS.get(item["group"], item["group"]),
                        ROUTE_LABELS[item["route"]],
                        _fmt_signed(item.get("end_to_end_success")),
                        _fmt_signed(item.get("answerability_accuracy")),
                        _fmt_signed(item.get("retrieval_recall")),
                        _fmt_signed(item.get("retrieval_precision")),
                        _fmt_signed(item.get("citation_recall")),
                        _fmt_signed(item.get("citation_doc_recall")),
                        _fmt_signed(item.get("citation_doc_precision")),
                    ]
                    for item in report["deltas"]
                ],
            ),
            "",
            "## Bilingual Citation Slice",
            _markdown_table(
                ["Model", "Route", "N", "Task success", "Citation doc R", "Citation doc P", "Citation source P"],
                [
                    [
                        row["model_label"],
                        ROUTE_LABELS[row["route"]],
                        row["n"],
                        _fmt(row.get("end_to_end_success")),
                        _fmt(row.get("citation_doc_recall")),
                        _fmt(row.get("citation_doc_precision")),
                        _fmt(row.get("citation_source_precision_proxy")),
                    ]
                    for row in report["bilingual_performance"]
                ],
            ),
            "",
            "## Operational Metrics",
            _markdown_table(
                [
                    "Model",
                    "Route",
                    "N",
                    "Mean latency",
                    "P95 latency",
                    "Input tokens",
                    "Output tokens",
                    "Mean captured cost",
                ],
                [
                    [
                        row["model_label"],
                        ROUTE_LABELS[row["route"]],
                        row["n"],
                        _seconds(row["mean_latency_seconds"]),
                        _seconds(row["p95_latency_seconds"]),
                        row["billed_input_tokens"],
                        row["billed_output_tokens"],
                        _usd(row["captured_generation_cost_mean_usd"]),
                    ]
                    for row in report["operational_metrics"]
                ],
            ),
            "",
            "Cost caveat: cost is captured final-generation billing only. Planner, reviewer, Embed, and Rerank costs are not fully recorded in these transcripts.",
            "",
            "## Reviewer Gate Metrics",
            _markdown_table(
                [
                    "Model",
                    "Release gate rate",
                    "Human gate rate",
                    "Failed gated",
                    "Failed released",
                    "Estimated citation precision",
                ],
                [
                    [
                        row["model_label"],
                        _fmt(row.get("release_gate_rate")),
                        _fmt(row.get("human_gate_rate")),
                        row.get("failed_cases_human_gated"),
                        row.get("failed_cases_released"),
                        _fmt(row.get("reviewer_estimated_citation_precision")),
                    ]
                    for row in report["reviewer_metrics"]
                ],
            ),
            "",
            "## Case Movement",
            _markdown_table(
                ["Route", "Improved", "Regressed", "Unchanged pass", "Unchanged fail"],
                [
                    [
                        ROUTE_LABELS[item["route"]],
                        item["improved"],
                        item["regressed"],
                        item["unchanged_pass"],
                        item["unchanged_fail"],
                    ]
                    for item in report["case_movement"]
                ],
            ),
            "",
            "## Limitations",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _load_report(path: Path | str) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        raise EvalHarnessError(f"Route report not found: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _model_descriptor(report: dict[str, Any], report_path: Path | str) -> dict[str, str]:
    return {
        "model_label": str(report.get("model_label", "")),
        "model_id": str(report.get("model_id", "")),
        "report_path": str(report_path),
    }


def _case_groups(registry: dict[str, Any]) -> dict[str, str]:
    return {
        str(case.get("id", "")): str(case.get("model_upgrade_group", "canonical_20"))
        for case in registry.get("cases", []) or []
        if isinstance(case, dict)
    }


def _case_slices(registry: dict[str, Any]) -> dict[str, str]:
    return {
        str(case.get("id", "")): str(case.get("eval_slice", "unknown"))
        for case in registry.get("cases", []) or []
        if isinstance(case, dict)
    }


def _sample_design(registry: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in registry.get("cases", []) or [] if isinstance(case, dict)]
    group_counts = Counter(str(case.get("model_upgrade_group", "canonical_20")) for case in cases)
    slice_counts = Counter(str(case.get("eval_slice", "unknown")) for case in cases)
    return {
        "case_count": len(cases),
        "group_counts": [{"group": key, "count": group_counts[key]} for key in sorted(group_counts)],
        "slice_counts": [{"slice": key, "count": slice_counts[key]} for key in sorted(slice_counts)],
    }


def _case_count_audit(
    model_rows: dict[str, list[dict[str, Any]]],
    registry: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = {
        str(case.get("id", "")).strip()
        for case in registry.get("cases", []) or []
        if isinstance(case, dict) and str(case.get("id", "")).strip()
    }
    route_sets: dict[tuple[str, str], set[str]] = {}
    rows = []
    for model_key, all_rows in model_rows.items():
        for route in ROUTE_ORDER:
            ids = {
                str(row.get("case_id", "")).strip()
                for row in all_rows
                if row.get("route") == route and str(row.get("case_id", "")).strip()
            }
            route_sets[(model_key, route)] = ids
            rows.append(
                {
                    "model": model_key,
                    "route": route,
                    "scored_case_count": len(ids),
                    "missing_expected_count": len(expected_ids - ids),
                    "unexpected_case_count": len(ids - expected_ids),
                }
            )

    sets = list(route_sets.values())
    same_sets = bool(sets) and all(ids == sets[0] for ids in sets)
    same_counts = len({len(ids) for ids in sets}) <= 1
    no_unexpected = all(not (ids - expected_ids) for ids in sets)
    matched_case_count = len(sets[0]) if same_sets and sets else 0
    if same_sets and no_unexpected and matched_case_count == len(expected_ids):
        status = "complete"
    elif same_sets and same_counts and no_unexpected:
        status = "matched_subset"
    else:
        status = "mismatched"
    return {
        "status": status,
        "expected_case_count": len(expected_ids),
        "matched_case_count": matched_case_count,
        "same_case_sets": same_sets,
        "same_case_counts": same_counts,
        "rows": rows,
    }


def _report_rows(
    report: dict[str, Any],
    *,
    case_groups: dict[str, str],
    case_slices: dict[str, str],
) -> list[dict[str, Any]]:
    route_by_run = {Path(path).name: route for route, path in report["route_dirs"].items()}
    rows = []
    for row in report.get("base_harness_report", {}).get("rows", []) or []:
        route = route_by_run.get(str(row.get("transcript_run", "")))
        if route is None:
            continue
        case_id = str(row.get("case_id", ""))
        item = dict(row)
        item["route"] = route
        item["group"] = case_groups.get(case_id, "canonical_20")
        item["declared_slice"] = case_slices.get(case_id, str(row.get("declared_slice") or row.get("slice", "")))
        item["model_label"] = str(report.get("model_label", ""))
        item["model_id"] = str(report.get("model_id", ""))
        rows.append(item)
    return rows


def _scorecards_by_group(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    scorecards: dict[str, dict[str, dict[str, Any]]] = {}
    for group in GROUP_ORDER:
        if group == "combined_24":
            group_rows = rows
        else:
            group_rows = [row for row in rows if row.get("group") == group]
        scorecards[group] = {
            route: _quality_scorecard([row for row in group_rows if row.get("route") == route])
            for route in ROUTE_ORDER
        }
    return scorecards


def _slice_scorecards(model_rows: dict[str, list[dict[str, Any]]], slice_fragment: str) -> list[dict[str, Any]]:
    rows = []
    for model_key, all_rows in model_rows.items():
        for route in ROUTE_ORDER:
            route_rows = [
                row
                for row in all_rows
                if row.get("route") == route and slice_fragment in str(row.get("declared_slice", ""))
            ]
            score = _quality_scorecard(route_rows)
            rows.append(
                {
                    "model": model_key,
                    "model_label": str(route_rows[0].get("model_label", model_key)) if route_rows else model_key,
                    "route": route,
                    **score,
                }
            )
    return rows


def _quality_scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    item: dict[str, Any] = {"n": len(rows)}
    for metric in QUALITY_METRICS:
        item[metric] = _mean(row.get(metric) for row in rows)
    return item


def _model_deltas(
    scorecards: dict[str, dict[str, dict[str, dict[str, Any]]]],
    *,
    before_key: str,
    after_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for group in GROUP_ORDER:
        for route in ROUTE_ORDER:
            before = scorecards[before_key][group][route]
            after = scorecards[after_key][group][route]
            item = {"group": group, "route": route}
            for metric in QUALITY_METRICS:
                item[metric] = _sub(after.get(metric), before.get(metric))
            rows.append(item)
    return rows


def _operational_rows(model_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model_key, report in model_reports.items():
        for route, values in report["operational_metrics"].items():
            latency = values.get("latency_seconds", {})
            rows.append(
                {
                    "model": model_key,
                    "model_label": str(report.get("model_label", model_key)),
                    "route": route,
                    "n": values.get("n"),
                    "mean_latency_seconds": latency.get("mean"),
                    "median_latency_seconds": latency.get("median"),
                    "p95_latency_seconds": latency.get("p95"),
                    "billed_input_tokens": values.get("billed_input_tokens"),
                    "billed_output_tokens": values.get("billed_output_tokens"),
                    "captured_generation_cost_mean_usd": values.get("captured_generation_cost_mean_usd"),
                    "captured_generation_cost_total_usd": values.get("captured_generation_cost_total_usd"),
                }
            )
    return rows


def _reviewer_rows(model_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model_key, report in model_reports.items():
        reviewer = report.get("reviewer_metrics", {})
        rows.append(
            {
                "model": model_key,
                "model_label": str(report.get("model_label", model_key)),
                "case_count": reviewer.get("case_count"),
                "release_gate_rate": reviewer.get("release_gate_rate"),
                "human_gate_rate": reviewer.get("human_gate_rate"),
                "failed_cases_human_gated": reviewer.get("failed_cases_human_gated"),
                "failed_cases_released": reviewer.get("failed_cases_released"),
                "reviewer_estimated_citation_precision": reviewer.get("reviewer_estimated_citation_precision"),
                "reviewed_citation_coverage": reviewer.get("reviewed_citation_coverage"),
            }
        )
    return rows


def _case_movement(
    model_rows: dict[str, list[dict[str, Any]]],
    *,
    before_key: str,
    after_key: str,
) -> list[dict[str, Any]]:
    output = []
    for route in ROUTE_ORDER:
        before = {
            str(row.get("case_id")): bool(row.get("passed"))
            for row in model_rows[before_key]
            if row.get("route") == route
        }
        after = {
            str(row.get("case_id")): bool(row.get("passed"))
            for row in model_rows[after_key]
            if row.get("route") == route
        }
        cases = sorted(set(before) & set(after))
        output.append(
            {
                "route": route,
                "improved": sum(1 for case_id in cases if not before[case_id] and after[case_id]),
                "regressed": sum(1 for case_id in cases if before[case_id] and not after[case_id]),
                "unchanged_pass": sum(1 for case_id in cases if before[case_id] and after[case_id]),
                "unchanged_fail": sum(1 for case_id in cases if not before[case_id] and not after[case_id]),
            }
        )
    return output


def _presentation_gate(
    scorecards: dict[str, dict[str, dict[str, dict[str, Any]]]],
    model_reports: dict[str, dict[str, Any]],
    *,
    case_count_audit: dict[str, Any],
    before_key: str,
    after_key: str,
) -> dict[str, Any]:
    comparability = _comparability(model_reports)
    deltas = _model_deltas(scorecards, before_key=before_key, after_key=after_key)
    combined = {item["route"]: item for item in deltas if item["group"] == "combined_24"}
    reviewed_delta = combined["reviewed_agent"].get("end_to_end_success")
    simple_delta = combined["simple_rag"].get("end_to_end_success")
    reviewed_citation_delta = combined["reviewed_agent"].get("citation_doc_precision")
    before_failed_released = int(model_reports[before_key].get("reviewer_metrics", {}).get("failed_cases_released") or 0)
    after_failed_released = int(model_reports[after_key].get("reviewer_metrics", {}).get("failed_cases_released") or 0)

    reasons = []
    reasons.append(f"Harness comparability status is {comparability['status']}.")
    if isinstance(simple_delta, int | float):
        reasons.append(f"Simple RAG task-success delta is {_fmt_signed(simple_delta)}.")
    if isinstance(reviewed_delta, int | float):
        reasons.append(f"Research + Reviewer task-success delta is {_fmt_signed(reviewed_delta)}.")
    if isinstance(reviewed_citation_delta, int | float):
        reasons.append(f"Research + Reviewer citation-document precision delta is {_fmt_signed(reviewed_citation_delta)}.")
    reasons.append(f"Reviewer failed-release count moved from {before_failed_released} to {after_failed_released}.")
    reasons.append(
        "Case coverage status is "
        f"{case_count_audit['status']}: "
        f"{case_count_audit['matched_case_count']} of "
        f"{case_count_audit['expected_case_count']} cases scored per comparable route/model."
    )

    improves_accuracy = any(
        isinstance(value, int | float) and value > 0
        for value in (simple_delta, reviewed_delta)
    )
    trust_not_worse = after_failed_released <= before_failed_released and (
        not isinstance(reviewed_citation_delta, int | float) or reviewed_citation_delta >= 0
    )
    if comparability["status"] != "comparable":
        recommendation = "internal_only"
    elif case_count_audit["status"] != "complete":
        recommendation = "internal_only"
    elif improves_accuracy and trust_not_worse:
        recommendation = "headline_candidate"
    elif trust_not_worse:
        recommendation = "appendix_or_efficiency_story"
    else:
        recommendation = "internal_only"
    return {"recommendation": recommendation, "reasons": reasons}


def _comparability(model_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = list(model_reports)
    if len(keys) != 2:
        return {"status": "unknown", "routes": []}
    before = model_reports[keys[0]].get("orchestration_contract", {})
    after = model_reports[keys[1]].get("orchestration_contract", {})
    rows = []
    for route in ROUTE_ORDER:
        before_route = before.get(route, {}) if isinstance(before, dict) else {}
        after_route = after.get(route, {}) if isinstance(after, dict) else {}
        comparable = (
            before_route.get("status") in {"complete", "partial"}
            and after_route.get("status") in {"complete", "partial"}
            and before_route.get("harness_id") == after_route.get("harness_id")
            and before_route.get("route_implementation_id") == after_route.get("route_implementation_id")
            and before_route.get("chunk_strategy") == after_route.get("chunk_strategy")
            and before_route.get("retrieval_mode") == after_route.get("retrieval_mode")
        )
        rows.append(
            {
                "route": route,
                "comparable": comparable,
                "before_harness_id": str(before_route.get("harness_id", "missing")),
                "after_harness_id": str(after_route.get("harness_id", "missing")),
                "before_route_implementation_id": str(before_route.get("route_implementation_id", "missing")),
                "after_route_implementation_id": str(after_route.get("route_implementation_id", "missing")),
                "before_status": str(before_route.get("status", "missing")),
                "after_status": str(after_route.get("status", "missing")),
            }
        )
    status = "comparable" if all(row["comparable"] for row in rows) else "not_comparable"
    return {"status": status, "routes": rows}


def _limitations() -> list[str]:
    return [
        "Model-upgrade claims are only valid when both models are rerun through the same current orchestration layer.",
        "Do not compare a saved baseline from a previous agent harness against a fresh run from a different agent harness.",
        "Route-to-route comparisons answer system-design questions; same-route model deltas answer model-upgrade questions.",
        "This is a single live run per route and model, so do not present it as statistically powered.",
        "The dataset is synthetic and prototype-sized.",
        "The model comparison isolates planner and final-generation model changes only when retrieval, rerank, chunking, corpus, and reviewer model are held fixed.",
        "Captured cost does not include all planner, reviewer, Embed, and Rerank billing telemetry.",
        "Citation precision still combines deterministic source precision and lexical support proxies unless a human or calibrated-judge citation label file is added.",
    ]


def _scorecard_row(model: str, route: str, values: dict[str, Any]) -> list[Any]:
    return [
        model,
        ROUTE_LABELS[route],
        values["n"],
        _fmt(values.get("end_to_end_success")),
        _fmt(values.get("answerability_accuracy")),
        _fmt(values.get("retrieval_recall")),
        _fmt(values.get("retrieval_precision")),
        _fmt(values.get("citation_recall")),
        _fmt(values.get("citation_doc_recall")),
        _fmt(values.get("citation_doc_precision")),
        _fmt(values.get("citation_source_precision_proxy")),
    ]


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return round(mean(numeric), 4) if numeric else None


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


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])
