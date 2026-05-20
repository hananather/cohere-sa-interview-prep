"""Layered offline eval metrics for saved Defence Agent runs.

The live demo registry already validates pass/fail readiness. This module turns
the same registry plus saved transcript JSON into scored, notebook-friendly
metrics without calling Cohere.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PACKAGE_ROOT / "data" / "evals" / "demo_query_registry.yaml"
DEFAULT_TRANSCRIPTS_DIR = PACKAGE_ROOT / "data" / "transcripts"
DEFAULT_TRANSCRIPT_RUN_NAMES = (
    "live_readiness_final_20260511_013937",
    "insufficient_evidence_strategy_realignment_20260511_131708",
)

QUALITY_METRICS = (
    "retrieval_recall",
    "retrieval_precision",
    "facet_recall",
    "generation_fact_recall",
    "forbidden_fact_absence",
    "answerability_accuracy",
    "citation_recall",
    "citation_precision",
    "citation_doc_recall",
    "citation_doc_precision",
    "citation_source_precision_proxy",
    "citation_support_precision_lexical",
    "end_to_end_success",
)

TARGET_SAMPLE_SIZES = {
    "public_single_query": 6,
    "multi_query": 6,
    "bilingual": 6,
    "scanned_manual": 6,
    "docx_origin": 6,
    "acl_answerable": 6,
    "acl_refusal": 6,
    "insufficient_evidence": 6,
    "follow_up": 6,
    "adversarial": 6,
}


class EvalHarnessError(RuntimeError):
    """Raised when the offline eval inputs cannot be scored."""


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load a registry YAML file."""

    registry_path = Path(path)
    if not registry_path.exists():
        raise EvalHarnessError(f"Registry not found: {registry_path}")
    return _load_registry_with_includes(registry_path, seen=set())


def _load_registry_with_includes(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise EvalHarnessError(f"Registry include cycle detected at: {path}")
    seen.add(resolved)

    registry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(registry, dict):
        raise EvalHarnessError(f"Registry must be a mapping: {path}")

    merged_cases: list[dict[str, Any]] = []
    include_paths = registry.get("include_registries", []) or registry.get("includes", []) or []
    for include_path in include_paths:
        child_path = Path(str(include_path))
        if not child_path.is_absolute():
            child_path = path.parent / child_path
        child = _load_registry_with_includes(child_path, seen=seen)
        merged_cases.extend(case for case in child.get("cases", []) or [] if isinstance(case, dict))

    merged_cases.extend(case for case in registry.get("cases", []) or [] if isinstance(case, dict))
    result = dict(registry)
    result.pop("include_registries", None)
    result.pop("includes", None)
    result["cases"] = merged_cases
    _validate_unique_case_ids(result["cases"], path)
    return result


def _validate_unique_case_ids(cases: list[dict[str, Any]], path: Path) -> None:
    counts = Counter(str(case.get("id", "")).strip() for case in cases if str(case.get("id", "")).strip())
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    if duplicates:
        raise EvalHarnessError(f"Duplicate registry case id(s) in {path}: {', '.join(duplicates)}")


def load_citation_precision_labels(path: Path | str | None) -> dict[str, bool]:
    """Load human or judge labels for citation/source support pairs."""

    if path is None:
        return {}
    label_path = Path(path)
    if not label_path.exists():
        raise EvalHarnessError(f"Citation precision label file not found: {label_path}")
    raw = yaml.safe_load(label_path.read_text(encoding="utf-8")) or {}
    entries = raw.get("labels", raw) if isinstance(raw, dict) else raw
    labels: dict[str, bool] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key", "")).strip()
        supported = _label_supported(entry)
        if key and supported is not None:
            labels[key] = supported
    return labels


def registry_cases_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return registry cases keyed by id."""

    cases = [case for case in registry.get("cases", []) or [] if isinstance(case, dict)]
    return {str(case.get("id", "")): case for case in cases if str(case.get("id", "")).strip()}


def default_transcript_paths(root: Path | str = DEFAULT_TRANSCRIPTS_DIR) -> list[Path]:
    """Return the default presentation-readiness transcript directories."""

    transcript_root = Path(root)
    return [transcript_root / name for name in DEFAULT_TRANSCRIPT_RUN_NAMES if (transcript_root / name).is_dir()]


def load_outcomes(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Load transcript outcome JSON from directories or individual files."""

    outcomes: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file_path in files:
            if not file_path.exists():
                raise EvalHarnessError(f"Transcript path not found: {file_path}")
            try:
                outcome = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise EvalHarnessError(f"Invalid transcript JSON: {file_path}") from exc
            if not isinstance(outcome, dict):
                continue
            outcome["_transcript_run"] = path.name if path.is_dir() else file_path.parent.name
            outcome["_transcript_file"] = file_path.name
            outcomes.append(outcome)
    return outcomes


def summarize_transcript_runs(
    *,
    registry_path: Path | str = DEFAULT_REGISTRY,
    transcript_paths: Iterable[Path | str] | None = None,
    include_citation_support: bool = False,
    citation_label_path: Path | str | None = None,
) -> dict[str, Any]:
    """Score transcript runs and return aggregate metrics."""

    registry = load_registry(registry_path)
    cases = registry_cases_by_id(registry)
    paths = list(transcript_paths) if transcript_paths is not None else default_transcript_paths()
    if not paths:
        raise EvalHarnessError("No transcript paths provided and no default transcript runs found.")
    outcomes = load_outcomes(paths)
    source_text_lookup = _load_page_text_lookup() if include_citation_support else None
    citation_labels = load_citation_precision_labels(citation_label_path)
    rows: list[dict[str, Any]] = []
    unknown_case_ids: list[str] = []
    for outcome in outcomes:
        case_id = str(outcome.get("parent_id") or outcome.get("case_id") or "")
        case = cases.get(case_id)
        if case is None:
            unknown = str(outcome.get("case_id", "")).strip()
            if unknown:
                unknown_case_ids.append(unknown)
            continue
        rows.append(
            score_outcome(
                outcome,
                case,
                source_text_lookup=source_text_lookup,
                citation_labels=citation_labels,
            )
        )
    if not rows:
        raise EvalHarnessError("No transcript outcomes matched registry cases.")
    return {
        "registry_path": str(registry_path),
        "transcript_paths": [str(path) for path in paths],
        "citation_label_path": str(citation_label_path) if citation_label_path else None,
        "sample_distribution": _sample_distribution(rows),
        "unique_case_distribution": _unique_case_distribution(rows),
        "target_sample_coverage": _target_sample_coverage(rows),
        "overall": _aggregate_rows(rows),
        "by_slice": _aggregate_by(rows, "slice"),
        "by_run": _aggregate_by(rows, "transcript_run"),
        "by_case": _case_consistency(rows),
        "failure_examples": _failure_examples(rows),
        "pilot_findings": _pilot_findings(rows),
        "rows": rows,
        "unknown_case_ids": unknown_case_ids,
        "notes": _report_notes(rows, unknown_case_ids),
    }


def citation_review_queue(
    *,
    registry_path: Path | str = DEFAULT_REGISTRY,
    transcript_paths: Iterable[Path | str] | None = None,
    include_source_text: bool = False,
) -> list[dict[str, Any]]:
    """Export citation/source pairs for human or judge support labeling."""

    registry = load_registry(registry_path)
    cases = registry_cases_by_id(registry)
    paths = list(transcript_paths) if transcript_paths is not None else default_transcript_paths()
    if not paths:
        raise EvalHarnessError("No transcript paths provided and no default transcript runs found.")
    outcomes = load_outcomes(paths)
    source_text_lookup = _load_page_text_lookup() if include_source_text else None
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        case_id = str(outcome.get("parent_id") or outcome.get("case_id") or "")
        case = cases.get(case_id)
        if case is None:
            continue
        final_audit = _dict(outcome.get("final_answer_audit"))
        for item in _citation_pairs(outcome, final_audit):
            citation = item["citation"]
            source = item["source"]
            source_text = _source_text_for_citation(source, source_text_lookup or {}) if source_text_lookup else ""
            rows.append(
                {
                    "key": item["key"],
                    "case_id": str(outcome.get("case_id", "")),
                    "parent_id": str(outcome.get("parent_id", "")),
                    "transcript_run": str(outcome.get("_transcript_run", "")),
                    "transcript_file": str(outcome.get("_transcript_file", "")),
                    "slice": _case_slice(case),
                    "persona_id": str(case.get("persona_id", outcome.get("persona_id", ""))),
                    "query": str(case.get("query", outcome.get("query", ""))),
                    "answer_span": str(citation.get("text", "") or ""),
                    "citation_start": _to_int(citation.get("start")),
                    "citation_end": _to_int(citation.get("end")),
                    "citation_index": item["citation_index"],
                    "source_index": item["source_index"],
                    "source_id": _source_identity(source),
                    "doc_id": str(source.get("doc_id", "")),
                    "page": source.get("page"),
                    "title": str(source.get("title", "")),
                    "source_excerpt": _source_excerpt(source_text),
                    "lexical_supported": (
                        _citation_text_supported_lexically(str(citation.get("text", "") or ""), source_text)
                        if source_text
                        else None
                    ),
                    "supported": None,
                    "review_notes": "",
                }
            )
    return rows


def score_outcome(
    outcome: dict[str, Any],
    case: dict[str, Any],
    *,
    source_text_lookup: dict[str, str] | None = None,
    citation_labels: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Compute layered metrics for one saved run outcome."""

    final_audit = _dict(outcome.get("final_answer_audit"))
    retrieval = _dict(final_audit.get("retrieval"))
    generation = _dict(final_audit.get("generation"))
    validation = _dict(outcome.get("citation_validation") or generation.get("citation_resolution"))
    citation_quality = _dict(outcome.get("citation_quality") or generation.get("citation_quality"))
    answer = str(outcome.get("answer", ""))
    expected_groups = _expected_doc_groups(case)
    acceptable_docs = set().union(*expected_groups) if expected_groups else set()
    retrieved_doc_ids = _retrieved_doc_ids(outcome)
    cited_doc_ids = _cited_doc_ids(final_audit)
    expected_refusal = bool(case.get("expected_refusal"))
    documents_sent = _documents_sent(outcome, generation)
    citation_precision, citation_label_count, citation_supported_count = _citation_precision_from_labels(
        outcome,
        final_audit,
        citation_labels or {},
    )

    row = {
        "case_id": str(outcome.get("case_id", "")),
        "parent_id": str(outcome.get("parent_id", "")),
        "transcript_run": str(outcome.get("_transcript_run", "")),
        "transcript_file": str(outcome.get("_transcript_file", "")),
        "slice": _case_slice(case),
        "persona_id": str(case.get("persona_id", outcome.get("persona_id", ""))),
        "query": str(case.get("query", outcome.get("query", ""))),
        "expected_refusal": expected_refusal,
        "passed": bool(outcome.get("passed", False)),
        "failure_count": len(outcome.get("failures", []) or []),
        "failures": "; ".join(str(item) for item in outcome.get("failures", []) or []),
        "search_count": _to_int(outcome.get("search_count") or retrieval.get("search_count")),
        "documents_sent_to_model": documents_sent,
        "citation_count": _to_int(outcome.get("citation_count") or validation.get("citation_count")),
        "citation_precision_label_count": citation_label_count,
        "citation_precision_supported_count": citation_supported_count,
        "retrieved_doc_ids": sorted(retrieved_doc_ids),
        "cited_doc_ids": sorted(cited_doc_ids),
        "expected_doc_groups": [sorted(group) for group in expected_groups],
    }

    row.update(
        {
            "retrieval_recall": _group_recall(expected_groups, retrieved_doc_ids),
            "retrieval_precision": _precision(retrieved_doc_ids, acceptable_docs),
            "facet_recall": _facet_recall(case, retrieved_doc_ids, cited_doc_ids),
            "generation_fact_recall": _fact_recall(case.get("expected_answer_facts", []), answer),
            "forbidden_fact_absence": _forbidden_absence(case.get("forbidden_answer_facts", []), answer),
            "answerability_accuracy": _answerability_accuracy(case, answer, documents_sent, retrieval),
            "citation_recall": _citation_recall(citation_quality, validation),
            "citation_precision": citation_precision,
            "citation_doc_recall": _group_recall(expected_groups, cited_doc_ids) if not expected_refusal else None,
            "citation_doc_precision": _precision(cited_doc_ids, acceptable_docs) if not expected_refusal else None,
            "citation_source_precision_proxy": _citation_source_precision_proxy(final_audit),
            "citation_support_precision_lexical": _citation_support_precision_lexical(
                final_audit,
                source_text_lookup,
            ),
            "end_to_end_success": 1.0 if outcome.get("passed") else 0.0,
        }
    )
    row["metric_gaps"] = _metric_gaps(row)
    return row


def markdown_report(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for notebooks and terminal output."""

    lines = [
        "# Defence Agent Evaluation Harness Report",
        "",
        "## Inputs",
        f"- Registry: `{report['registry_path']}`",
        "- Transcript runs:",
    ]
    lines.extend(f"  - `{path}`" for path in report["transcript_paths"])
    lines.extend(
        [
            "",
            "## Sample Distribution",
            _markdown_table(
                ["Slice", "Transcript outcomes"],
                [[item["slice"], item["count"]] for item in report["sample_distribution"]],
            ),
            "",
            "## Unique Query Distribution",
            _markdown_table(
                ["Slice", "Unique cases"],
                [[item["slice"], item["count"]] for item in report["unique_case_distribution"]],
            ),
            "",
            "## Coverage Against 60-Case Target Bank",
            _markdown_table(
                ["Slice", "Current", "Target", "Missing"],
                [
                    [item["slice"], item["current"], item["target"], item["missing"]]
                    for item in report["target_sample_coverage"]
                ],
            ),
            "",
            "## Overall Metrics",
            _metric_table(report["overall"]),
            "",
            "## Pilot Findings",
            *[f"- {finding}" for finding in report.get("pilot_findings", [])],
            "",
            "## Metrics By Slice",
            _markdown_table(
                [
                    "Slice",
                    "N",
                    "Pass",
                    "Retrieval R",
                    "Retrieval P",
                    "Citation R",
                    "Citation P",
                    "Citation P proxy",
                    "Lexical support P",
                    "Answerability",
                ],
                [
                    [
                        item["group"],
                        item["n"],
                        _fmt(item.get("end_to_end_success")),
                        _fmt(item.get("retrieval_recall")),
                        _fmt(item.get("retrieval_precision")),
                        _fmt(item.get("citation_recall")),
                        _fmt(item.get("citation_precision")),
                        _fmt(item.get("citation_source_precision_proxy")),
                        _fmt(item.get("citation_support_precision_lexical")),
                        _fmt(item.get("answerability_accuracy")),
                    ]
                    for item in report["by_slice"]
                ],
            ),
            "",
            "## Case Consistency",
            _markdown_table(
                ["Case", "Trials", "Pass rate", "All passed", "Any passed"],
                [
                    [
                        item["case_id"],
                        item["trials"],
                        _fmt(item["pass_rate"]),
                        item["all_trials_passed"],
                        item["any_trial_passed"],
                    ]
                    for item in report["by_case"]
                ],
            ),
            "",
            "## Failure Examples",
            _markdown_table(
                ["Run", "Case", "Slice", "Failures"],
                [
                    [item["transcript_run"], item["case_id"], item["slice"], item["failures"]]
                    for item in report.get("failure_examples", [])
                ]
                or [["n/a", "n/a", "n/a", "No failures in scored transcript outcomes."]],
            ),
            "",
            "## Notes",
        ]
    )
    lines.extend(f"- {note}" for note in report.get("notes", []))
    return "\n".join(lines) + "\n"


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"n": len(rows)}
    for metric in QUALITY_METRICS:
        if metric == "citation_precision":
            aggregate[metric] = _citation_precision_aggregate(rows)
        else:
            aggregate[metric] = _mean_metric(rows, metric)
    aggregate["avg_search_count"] = _mean_metric(rows, "search_count")
    aggregate["avg_documents_sent_to_model"] = _mean_metric(rows, "documents_sent_to_model")
    aggregate["avg_citation_count"] = _mean_metric(rows, "citation_count")
    aggregate["citation_precision_label_count"] = sum(
        int(row.get("citation_precision_label_count", 0) or 0) for row in rows
    )
    return aggregate


def _aggregate_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    aggregates = []
    for group, group_rows in sorted(grouped.items()):
        item = {"group": group}
        item.update(_aggregate_rows(group_rows))
        aggregates.append(item)
    return aggregates


def _sample_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("slice", "unknown")) for row in rows)
    return [{"slice": key, "count": counts[key]} for key in sorted(counts)]


def _unique_case_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_ids_by_slice: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        case_ids_by_slice[str(row.get("slice", "unknown"))].add(str(row.get("case_id", "")))
    return [
        {"slice": key, "count": len(case_ids_by_slice[key])}
        for key in sorted(case_ids_by_slice)
    ]


def _target_sample_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_counts = {item["slice"]: int(item["count"]) for item in _unique_case_distribution(rows)}
    return [
        {
            "slice": slice_name,
            "current": unique_counts.get(slice_name, 0),
            "target": target,
            "missing": max(target - unique_counts.get(slice_name, 0), 0),
        }
        for slice_name, target in TARGET_SAMPLE_SIZES.items()
    ]


def _case_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_id", ""))].append(row)
    summaries: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(grouped.items()):
        passed = [bool(row.get("passed")) for row in case_rows]
        summaries.append(
            {
                "case_id": case_id,
                "slice": str(case_rows[0].get("slice", "")),
                "trials": len(case_rows),
                "pass_rate": round(sum(1 for item in passed if item) / len(passed), 4),
                "all_trials_passed": all(passed),
                "any_trial_passed": any(passed),
            }
        )
    return summaries


def _failure_examples(rows: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for row in rows:
        if row.get("passed"):
            continue
        examples.append(
            {
                "transcript_run": str(row.get("transcript_run", "")),
                "case_id": str(row.get("case_id", "")),
                "slice": str(row.get("slice", "")),
                "failures": str(row.get("failures", "")) or "failed_without_recorded_reason",
            }
        )
    return examples[:limit]


def _pilot_findings(rows: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    overall = _aggregate_rows(rows)
    unique_total = sum(item["count"] for item in _unique_case_distribution(rows))
    findings.append(f"Scored {len(rows)} transcript outcomes across {unique_total} unique registry cases.")
    pass_rate = overall.get("end_to_end_success")
    if isinstance(pass_rate, float):
        findings.append(f"Observed end-to-end pass rate is {_fmt(pass_rate)} across scored outcomes.")
    weak_cases = [item for item in _case_consistency(rows) if not item["all_trials_passed"]]
    if weak_cases:
        findings.append(
            "Repeated-run consistency gaps remain in: "
            + ", ".join(item["case_id"] for item in weak_cases[:5])
            + "."
        )
    missing_slices = [
        item["slice"]
        for item in _target_sample_coverage(rows)
        if int(item["missing"]) > 0
    ]
    if missing_slices:
        findings.append(
            "The current saved-run corpus is not a balanced pilot set; missing target coverage includes "
            + ", ".join(missing_slices[:6])
            + "."
        )
    precision_values = [
        (str(item["group"]), item.get("retrieval_precision"))
        for item in _aggregate_by(rows, "slice")
        if isinstance(item.get("retrieval_precision"), float)
    ]
    if precision_values:
        weakest_slice, weakest_value = min(precision_values, key=lambda item: item[1])
        findings.append(
            f"Lowest retrieval precision is in `{weakest_slice}` at {_fmt(weakest_value)}; inspect whether broad context is useful or noisy."
        )
    findings.append(_semantic_citation_precision_finding(rows))
    findings.append(
        _citation_support_finding(rows)
    )
    return findings


def _semantic_citation_precision_finding(rows: list[dict[str, Any]]) -> str:
    label_count = sum(int(row.get("citation_precision_label_count", 0) or 0) for row in rows)
    value = _citation_precision_aggregate(rows)
    if value is None:
        return (
            "Semantic citation precision has not been labeled yet; export the citation review queue "
            "and score human or calibrated-judge labels to turn citation support into a real precision metric."
        )
    return f"Semantic citation precision is {_fmt(value)} across {label_count} labeled citation/source pairs."


def _citation_support_finding(rows: list[dict[str, Any]]) -> str:
    value = _mean_metric(rows, "citation_support_precision_lexical")
    if value is None:
        return (
            "True citation support precision remains a calibrated judge or human-review task; "
            "rerun with citation-support scoring to add a deterministic lexical support proxy."
        )
    return (
        f"Lexical citation support precision is {_fmt(value)} across scoreable citation/source pairs; "
        "a calibrated judge or human review is still required for semantic support precision."
    )


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), int | float)]
    return round(mean(values), 4) if values else None


def _citation_precision_aggregate(rows: list[dict[str, Any]]) -> float | None:
    total = sum(int(row.get("citation_precision_label_count", 0) or 0) for row in rows)
    if total == 0:
        return None
    supported = sum(int(row.get("citation_precision_supported_count", 0) or 0) for row in rows)
    return round(supported / total, 4)


def _expected_doc_groups(case: dict[str, Any]) -> list[set[str]]:
    expected = _string_set(case.get("expected_doc_ids"))
    allowed_alt = _string_set(case.get("allowed_alt_doc_ids"))
    if not expected:
        return []
    if len(expected) == 1 and allowed_alt:
        return [expected | allowed_alt]
    return [{doc_id} for doc_id in sorted(expected)]


def _facet_recall(case: dict[str, Any], retrieved_doc_ids: set[str], cited_doc_ids: set[str]) -> float | None:
    facets = [facet for facet in case.get("expected_facets", []) or [] if isinstance(facet, dict)]
    if not facets:
        return None
    available = retrieved_doc_ids | cited_doc_ids
    hits = 0
    scored = 0
    for facet in facets:
        expected = _string_set(facet.get("expected_doc_ids"))
        allowed_alt = _string_set(facet.get("allowed_alt_doc_ids"))
        acceptable = expected | allowed_alt
        if not acceptable:
            continue
        scored += 1
        hits += 1 if acceptable & available else 0
    return round(hits / scored, 4) if scored else None


def _group_recall(groups: list[set[str]], observed: set[str]) -> float | None:
    if not groups:
        return None
    return round(sum(1 for group in groups if group & observed) / len(groups), 4)


def _precision(observed: set[str], acceptable: set[str]) -> float | None:
    if not observed or not acceptable:
        return None
    return round(len(observed & acceptable) / len(observed), 4)


def _fact_recall(expected_facts: Any, answer: str) -> float | None:
    facts = [str(fact) for fact in expected_facts or [] if str(fact)]
    if not facts:
        return None
    text = answer.lower()
    return round(sum(1 for fact in facts if fact.lower() in text) / len(facts), 4)


def _forbidden_absence(forbidden_facts: Any, answer: str) -> float | None:
    facts = [str(fact) for fact in forbidden_facts or [] if str(fact)]
    if not facts:
        return None
    text = answer.lower()
    violations = sum(1 for fact in facts if fact.lower() in text)
    return round(1.0 - violations / len(facts), 4)


def _answerability_accuracy(
    case: dict[str, Any],
    answer: str,
    documents_sent: int,
    retrieval: dict[str, Any],
) -> float | None:
    expected_refusal = bool(case.get("expected_refusal"))
    explicit_expected_documents = case.get("expected_documents_sent_to_model")
    answerability_records = [item for item in retrieval.get("answerability", []) or [] if isinstance(item, dict)]
    retrieval_refused = any(item.get("answerable") is False for item in answerability_records)
    actual_refusal = retrieval_refused or _is_refusal_answer(answer)
    if expected_refusal:
        expected_zero_docs = explicit_expected_documents == 0 if explicit_expected_documents is not None else True
        return 1.0 if actual_refusal and (not expected_zero_docs or documents_sent == 0) else 0.0
    if explicit_expected_documents == 0:
        return 1.0 if documents_sent == 0 else 0.0
    return 1.0 if not actual_refusal else 0.0


def _citation_recall(citation_quality: dict[str, Any], validation: dict[str, Any]) -> float | None:
    raw = citation_quality.get("citation_recall_proxy")
    if isinstance(raw, int | float):
        return round(float(raw), 4)
    coverage = _dict(validation.get("coverage"))
    claim_count = _to_int(coverage.get("claim_count"))
    covered = _to_int(coverage.get("covered_claim_count"))
    return round(covered / claim_count, 4) if claim_count else None


def _citation_precision_from_labels(
    outcome: dict[str, Any],
    final_audit: dict[str, Any],
    citation_labels: dict[str, bool],
) -> tuple[float | None, int, int]:
    if not citation_labels:
        return None, 0, 0
    total = 0
    supported = 0
    for item in _citation_pairs(outcome, final_audit):
        label = citation_labels.get(item["key"])
        if label is None:
            continue
        total += 1
        supported += 1 if label else 0
    return (round(supported / total, 4), total, supported) if total else (None, 0, 0)


def _citation_source_precision_proxy(final_audit: dict[str, Any]) -> float | None:
    citations = [item for item in final_audit.get("citations", []) or [] if isinstance(item, dict)]
    if not citations:
        return None
    retrieval = _dict(final_audit.get("retrieval"))
    generation = _dict(final_audit.get("generation"))
    allowed_keys = _source_keys(retrieval.get("sources_sent_to_answer"), include_doc_id=False)
    allowed_keys |= {str(value) for value in generation.get("cohere_document_ids", []) or [] if str(value)}
    excluded_keys = _source_keys(retrieval.get("excluded_sources"), include_doc_id=True)
    total = 0
    resolved = 0
    for citation in citations:
        for source in citation.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            total += 1
            source_keys = _source_keys([source], include_doc_id=False)
            exclusion_keys = _source_keys([source], include_doc_id=True)
            if source_keys & allowed_keys and not (exclusion_keys & excluded_keys):
                resolved += 1
    return round(resolved / total, 4) if total else None


def _citation_pairs(outcome: dict[str, Any], final_audit: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    citations = [item for item in final_audit.get("citations", []) or [] if isinstance(item, dict)]
    for citation_index, citation in enumerate(citations):
        sources = [item for item in citation.get("sources", []) or [] if isinstance(item, dict)]
        for source_index, source in enumerate(sources):
            pairs.append(
                {
                    "key": _citation_pair_key(outcome, citation, source, citation_index, source_index),
                    "citation": citation,
                    "source": source,
                    "citation_index": citation_index,
                    "source_index": source_index,
                }
            )
    return pairs


def _citation_pair_key(
    outcome: dict[str, Any],
    citation: dict[str, Any],
    source: dict[str, Any],
    citation_index: int,
    source_index: int,
) -> str:
    run = str(outcome.get("_transcript_run") or outcome.get("transcript_run") or "unknown_run")
    file_name = str(outcome.get("_transcript_file") or outcome.get("transcript_file") or "unknown_file")
    span = f"{_to_int(citation.get('start'))}-{_to_int(citation.get('end'))}"
    return (
        f"{run}/{file_name}::citation:{citation_index}:source:{source_index}:"
        f"span:{span}:source:{_source_identity(source)}"
    )


def _source_identity(source: dict[str, Any]) -> str:
    for field in ("source_id", "chunk_id", "cohere_document_id", "citation_id"):
        value = str(source.get(field, "")).strip()
        if value:
            return value
    doc_id = str(source.get("doc_id", "")).strip()
    page = str(source.get("page", "")).strip()
    if doc_id and page:
        return f"{doc_id}:page:{page}"
    return doc_id or "unknown_source"


def _citation_support_precision_lexical(
    final_audit: dict[str, Any],
    source_text_lookup: dict[str, str] | None,
) -> float | None:
    if source_text_lookup is None:
        return None
    citations = [item for item in final_audit.get("citations", []) or [] if isinstance(item, dict)]
    if not citations:
        return None
    total = 0
    supported = 0
    for citation in citations:
        cited_text = str(citation.get("text", "") or "")
        for source in citation.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            source_text = _source_text_for_citation(source, source_text_lookup)
            if not source_text:
                continue
            total += 1
            if _citation_text_supported_lexically(cited_text, source_text):
                supported += 1
    return round(supported / total, 4) if total else None


def _source_text_for_citation(source: dict[str, Any], source_text_lookup: dict[str, str]) -> str:
    for field in ("source_id", "chunk_id", "cohere_document_id", "citation_id"):
        value = str(source.get(field, "")).strip()
        if value and value in source_text_lookup:
            return source_text_lookup[value]
    doc_id = str(source.get("doc_id", "")).strip()
    page = str(source.get("page", "")).strip()
    if doc_id and page.isdigit():
        page_id = f"{doc_id}_page_{int(page):03d}"
        return source_text_lookup.get(page_id, "")
    return ""


def _source_excerpt(source_text: str, *, limit: int = 900) -> str:
    excerpt = re.sub(r"\s+", " ", source_text).strip()
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit - 3].rstrip() + "..."


def _citation_text_supported_lexically(cited_text: str, source_text: str) -> bool:
    cited_normalized = _normalize_text(cited_text)
    source_normalized = _normalize_text(source_text)
    if not cited_normalized or not source_normalized:
        return False
    if cited_normalized in source_normalized:
        return True

    cited_numbers = set(re.findall(r"\d+(?:\.\d+)?", cited_normalized))
    source_numbers = set(re.findall(r"\d+(?:\.\d+)?", source_normalized))
    if cited_numbers and not cited_numbers.issubset(source_numbers):
        return False

    cited_tokens = _support_tokens(cited_normalized)
    if not cited_tokens:
        return False
    source_tokens = _support_tokens(source_normalized)
    overlap = len(cited_tokens & source_tokens)
    required = min(len(cited_tokens), max(2, round(len(cited_tokens) * 0.45)))
    return overlap >= required


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _support_tokens(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", value.lower())
        if len(token) > 2 and token not in stopwords
    }


def _load_page_text_lookup() -> dict[str, str]:
    from defence_agent.retrieval.document_pages import load_document_pages

    return {page.page_id: page.text for page in load_document_pages(PACKAGE_ROOT / "data" / "corpus")}


def _retrieved_doc_ids(outcome: dict[str, Any]) -> set[str]:
    doc_ids = _string_set(outcome.get("doc_ids"))
    for audit_key in ("first_answer_audit", "final_answer_audit"):
        audit = _dict(outcome.get(audit_key))
        retrieval = _dict(audit.get("retrieval"))
        for group_key in ("sources_sent_to_answer", "authorized_sources"):
            for source in retrieval.get(group_key, []) or []:
                if isinstance(source, dict) and source.get("doc_id"):
                    doc_ids.add(str(source["doc_id"]))
    return doc_ids


def _cited_doc_ids(final_audit: dict[str, Any]) -> set[str]:
    doc_ids: set[str] = set()
    for citation in final_audit.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        for source in citation.get("sources", []) or []:
            if isinstance(source, dict) and source.get("doc_id"):
                doc_ids.add(str(source["doc_id"]))
    return doc_ids


def _documents_sent(outcome: dict[str, Any], generation: dict[str, Any]) -> int:
    raw = outcome.get("documents_sent_to_model", generation.get("document_count", 0))
    return _to_int(raw)


def _source_keys(sources: Any, *, include_doc_id: bool) -> set[str]:
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


def _case_slice(case: dict[str, Any]) -> str:
    explicit_slice = str(case.get("eval_slice", "")).strip()
    if explicit_slice in TARGET_SAMPLE_SIZES:
        return explicit_slice
    case_id = str(case.get("id", "")).lower()
    query = str(case.get("query", "")).lower()
    persona = str(case.get("persona_id", "")).lower()
    source_types = _string_set(case.get("expected_source_types"))
    expected_docs = _string_set(case.get("expected_doc_ids"))
    if case.get("expected_refusal"):
        if case.get("expected_excluded_doc_ids"):
            return "acl_refusal"
        return "insufficient_evidence"
    if case.get("follow_up"):
        return "follow_up"
    if "scanned" in case_id or "official_public_scanned_manual_excerpt" in source_types:
        return "scanned_manual"
    if "docx" in case_id or "official_public_docx_origin_pdf" in source_types:
        return "docx_origin"
    if (
        "fr" == str(case.get("target_answer_language", "")).lower()
        or "french" in case_id
        or "multilingual" in case_id
        or "quelles" in query
        or case.get("allowed_alt_doc_ids")
    ):
        return "bilingual"
    if "secret" in persona or any(doc.startswith("SYN-") for doc in expected_docs):
        return "acl_answerable"
    if int(case.get("min_search_calls", 1) or 1) > 1 or len(expected_docs) > 1:
        return "multi_query"
    return "public_single_query"


def _metric_gaps(row: dict[str, Any]) -> list[str]:
    gaps = []
    if row.get("citation_source_precision_proxy") is not None:
        if row.get("citation_precision") is None:
            gaps.append("semantic_citation_precision_unlabeled")
        if row.get("citation_support_precision_lexical") is None:
            gaps.append("citation_support_precision_not_scored")
    if row.get("generation_fact_recall") is None and not row.get("expected_refusal"):
        gaps.append("no_gold_answer_facts_for_correctness")
    if row.get("retrieval_recall") is None and not row.get("expected_refusal"):
        gaps.append("no_gold_documents_for_retrieval")
    return gaps


def _report_notes(rows: list[dict[str, Any]], unknown_case_ids: list[str]) -> list[str]:
    notes = [
        "Runs are scored offline from saved transcript JSON; no live Cohere calls are made.",
        "Citation recall is the automated claim-span coverage proxy recorded by the app.",
        "Citation precision is only populated from human or calibrated-judge citation/source support labels.",
        "Citation source precision proxy checks whether citation source IDs resolve to authorized documents sent to generation.",
        "Generation correctness is deterministic only where registry cases define expected or forbidden answer facts.",
    ]
    if unknown_case_ids:
        notes.append(f"Skipped transcript files with case IDs missing from the registry: {', '.join(sorted(unknown_case_ids))}.")
    if any(row.get("metric_gaps") for row in rows):
        notes.append("Some rows intentionally report metric gaps where the current registry lacks gold answers or claim-level labels.")
    if any(row.get("citation_support_precision_lexical") is not None for row in rows):
        notes.append(
            "Lexical citation support precision reloads corpus page text by citation source ID and checks cited span overlap. "
            "It is deterministic and inspectable, but it is not a replacement for semantic judge or human validation."
        )
    return notes


def _label_supported(entry: dict[str, Any]) -> bool | None:
    raw = entry.get("supported", entry.get("judgment"))
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"supported", "true", "yes", "1", "pass"}:
        return True
    if value in {"unsupported", "not_supported", "false", "no", "0", "fail", "contradicted"}:
        return False
    return None


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _metric_table(metrics: dict[str, Any]) -> str:
    rows = []
    for key, value in metrics.items():
        if key == "n":
            rows.append(["sample_size", value])
        else:
            rows.append([key, _fmt(value)])
    return _markdown_table(["Metric", "Value"], rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_set(value: Any) -> set[str]:
    return {str(item) for item in value or [] if str(item)}


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_refusal_answer(answer: str) -> bool:
    text = answer.lower()
    return any(
        phrase in text
        for phrase in (
            "do not have enough",
            "not enough",
            "evidence is insufficient",
            "insufficient evidence",
            "no authorized evidence",
            "cannot answer",
        )
    )
