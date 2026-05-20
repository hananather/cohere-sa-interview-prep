"""Client-presentable Defence Agent eval report.

This report is intentionally less flattering than the small model-quality
scorecard. It uses the balanced 60-case pilot bank, separates model-quality
metrics from deterministic security invariants, and reports operational
latency/cost telemetry where the saved artifacts support it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import yaml

from defence_agent.evals.harness import (
    DEFAULT_TRANSCRIPTS_DIR,
    EvalHarnessError,
    load_registry,
    summarize_transcript_runs,
)
from defence_agent.evals.model_quality_report import DEFAULT_CITATION_LABELS
from defence_agent.retrieval.document_pages import load_document_pages


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEFENSIBLE_REGISTRY = PACKAGE_ROOT / "data" / "evals" / "pilot_eval_bank.yaml"
DEFAULT_AMBIGUOUS_REGISTRY = PACKAGE_ROOT / "data" / "evals" / "ambiguous_clarification_eval_bank.yaml"
DEFAULT_FULL_PILOT_TRANSCRIPT_DIR = DEFAULT_TRANSCRIPTS_DIR / "eval_full_registry_20260514_020915"
DEFAULT_REVIEWER_RUN_DIR = (
    PACKAGE_ROOT
    / "data"
    / "evals"
    / "reviewer_challenge_runs"
    / "reviewed_answer_challenges_20260520_040245"
)
DEFAULT_DEFENSIBLE_REPORT_JSON = PACKAGE_ROOT / "data" / "evals" / "reports" / "defensible_eval_report.json"
DEFAULT_DEFENSIBLE_REPORT_MD = PACKAGE_ROOT / "data" / "evals" / "reports" / "defensible_eval_report.md"

COMMAND_A_INPUT_USD_PER_1M = 2.5
COMMAND_A_OUTPUT_USD_PER_1M = 10.0
APPROX_TOKEN_MULTIPLIER = 1.25
GENERATION_PROMPT_OVERHEAD_TOKENS = 600

MODEL_QUALITY_SLICES = {
    "public_single_query",
    "multi_query",
    "bilingual",
    "scanned_manual",
    "docx_origin",
    "follow_up",
    "insufficient_evidence",
}
SECURITY_INVARIANT_SLICES = {"acl_answerable", "acl_refusal", "adversarial"}
HARD_COMMON_SLICES = {"multi_query", "bilingual", "scanned_manual", "docx_origin", "follow_up"}


def build_defensible_eval_report(
    *,
    registry_path: Path | str = DEFAULT_DEFENSIBLE_REGISTRY,
    transcript_dir: Path | str = DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
    citation_label_path: Path | str | None = DEFAULT_CITATION_LABELS,
    ambiguous_registry_path: Path | str = DEFAULT_AMBIGUOUS_REGISTRY,
    reviewer_run_dir: Path | str = DEFAULT_REVIEWER_RUN_DIR,
) -> dict[str, Any]:
    """Build a defensible eval report from saved pilot artifacts."""

    registry = load_registry(registry_path)
    transcript_paths = transcript_paths_for_registry(registry, Path(transcript_dir))
    scored = summarize_transcript_runs(
        registry_path=registry_path,
        transcript_paths=transcript_paths,
        include_citation_support=True,
        citation_label_path=citation_label_path,
    )
    ambiguous_registry = load_registry(ambiguous_registry_path) if Path(ambiguous_registry_path).exists() else {}
    report = {
        "schema_version": "defensible_eval_report.v1",
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha256(registry),
        "ambiguous_registry_path": str(ambiguous_registry_path),
        "ambiguous_registry_sha256": registry_sha256(ambiguous_registry) if ambiguous_registry else None,
        "transcript_dir": str(transcript_dir),
        "citation_label_path": str(citation_label_path) if citation_label_path else None,
        "sample_design": sample_design(registry, ambiguous_registry=ambiguous_registry),
        "scorecards": scorecards(scored),
        "operational_metrics": operational_metrics(scored, transcript_dir=Path(transcript_dir)),
        "reviewer_challenge_metrics": reviewer_challenge_metrics(Path(reviewer_run_dir)),
        "base_harness_report": scored,
        "limitations": limitations(ambiguous_registry),
    }
    report["markdown"] = defensible_eval_markdown(report)
    return report


def transcript_paths_for_registry(registry: dict[str, Any], transcript_dir: Path) -> list[Path]:
    """Return saved transcript files matching registry case ids."""

    case_ids = {
        str(case.get("id", "")).strip()
        for case in registry.get("cases", []) or []
        if isinstance(case, dict) and str(case.get("id", "")).strip()
    }
    paths = [transcript_dir / f"{case_id}.json" for case_id in sorted(case_ids)]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise EvalHarnessError(f"No transcript files found under {transcript_dir}")
    return existing


def write_defensible_eval_report(report: dict[str, Any], *, json_path: Path, md_path: Path) -> None:
    """Write JSON and Markdown reports."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    markdown = str(payload.pop("markdown", ""))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


def registry_sha256(registry: dict[str, Any]) -> str:
    """Stable sample hash over the case contract, not YAML formatting."""

    cases = []
    for case in registry.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        cases.append(
            {
                "id": case.get("id", ""),
                "parent_id": case.get("parent_id", ""),
                "persona_id": case.get("persona_id", ""),
                "eval_slice": case.get("eval_slice", ""),
                "query": case.get("query", ""),
                "follow_up": case.get("follow_up", ""),
                "expected_refusal": bool(case.get("expected_refusal")),
                "expected_clarification": bool(case.get("expected_clarification")),
                "expected_doc_ids": sorted(str(item) for item in case.get("expected_doc_ids", []) or []),
                "allowed_alt_doc_ids": sorted(str(item) for item in case.get("allowed_alt_doc_ids", []) or []),
                "expected_excluded_doc_ids": sorted(
                    str(item) for item in case.get("expected_excluded_doc_ids", []) or []
                ),
                "expected_answer_facts": sorted(str(item) for item in case.get("expected_answer_facts", []) or []),
                "forbidden_answer_facts": sorted(str(item) for item in case.get("forbidden_answer_facts", []) or []),
                "min_search_calls": case.get("min_search_calls"),
                "min_citations": case.get("min_citations"),
            }
        )
    blob = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sample_design(registry: dict[str, Any], *, ambiguous_registry: dict[str, Any]) -> dict[str, Any]:
    """Describe why the sample is defensible."""

    cases = [case for case in registry.get("cases", []) or [] if isinstance(case, dict)]
    ambiguous_cases = [case for case in ambiguous_registry.get("cases", []) or [] if isinstance(case, dict)]
    slice_counts = Counter(str(case.get("eval_slice", "unknown")) for case in cases)
    answerable = [case for case in cases if not case.get("expected_refusal")]
    unanswerable = [case for case in cases if case.get("expected_refusal")]
    hard_common = [case for case in cases if str(case.get("eval_slice", "")) in HARD_COMMON_SLICES]
    contrast_groups = _contrast_groups(cases)
    return {
        "scored_case_count": len(cases),
        "scored_slice_counts": dict(sorted(slice_counts.items())),
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(unanswerable),
        "hard_common_case_count": len(hard_common),
        "security_invariant_case_count": sum(
            count for key, count in slice_counts.items() if key in SECURITY_INVARIANT_SLICES
        ),
        "contrast_group_count": len(contrast_groups),
        "contrast_groups": contrast_groups,
        "ambiguous_clarification_case_count": len(ambiguous_cases),
        "ambiguous_clarification_status": (
            "defined_not_scored_in_saved_full_pilot" if ambiguous_cases else "missing"
        ),
        "selection_policy": [
            "Use a balanced 60-case pilot rather than the smaller 15-case model-quality demo slice.",
            "Keep six cases per scored slice so a single easy slice cannot dominate the average.",
            "Separate stochastic model-quality metrics from deterministic security invariants.",
            "Include answerable, unanswerable, hard/common, bilingual, mixed-format, follow-up, adversarial, and contrastive cases.",
            "Track ambiguity as an explicit clarification bank instead of pretending existing transcripts measure it.",
        ],
    }


def scorecards(scored: dict[str, Any]) -> dict[str, Any]:
    """Return layered scorecards without blending incompatible metrics."""

    rows = scored.get("rows", [])
    model_rows = [row for row in rows if row.get("slice") in MODEL_QUALITY_SLICES]
    answerable_rows = [row for row in model_rows if not row.get("expected_refusal")]
    unanswerable_rows = [row for row in model_rows if row.get("expected_refusal")]
    hard_rows = [row for row in model_rows if row.get("slice") in HARD_COMMON_SLICES]
    security_rows = [row for row in rows if row.get("slice") in SECURITY_INVARIANT_SLICES]
    security_scorecard = _scorecard(security_rows)
    security_scorecard.update(
        {
            "retrieval_recall": None,
            "retrieval_precision": None,
            "citation_recall": None,
            "citation_precision": None,
        }
    )
    return {
        "all_saved_pilot": _scorecard(rows),
        "model_quality_only": _scorecard(model_rows),
        "answerable_model_quality": _scorecard(answerable_rows),
        "unanswerable_evidence_gap": _scorecard(unanswerable_rows),
        "hard_common": _scorecard(hard_rows),
        "security_invariants": security_scorecard,
        "by_slice": scored.get("by_slice", []),
        "failure_examples": scored.get("failure_examples", []),
    }


def operational_metrics(scored: dict[str, Any], *, transcript_dir: Path) -> dict[str, Any]:
    """Report measured latency when present and estimates when telemetry is missing."""

    text_lookup = _page_text_lookup()
    rows = []
    for row in scored.get("rows", []):
        path = transcript_dir / str(row.get("transcript_file", ""))
        if not path.exists():
            continue
        outcome = json.loads(path.read_text(encoding="utf-8"))
        metrics = _operational_row(row, outcome, text_lookup)
        rows.append(metrics)
    return {
        "status": "complete" if rows else "missing",
        "pricing_assumption": {
            "model": "command-a-03-2025 / Command A family",
            "input_usd_per_1m_tokens": COMMAND_A_INPUT_USD_PER_1M,
            "output_usd_per_1m_tokens": COMMAND_A_OUTPUT_USD_PER_1M,
            "source": "Cohere Command A docs, checked 2026-05-20",
            "excludes": "Embed and Rerank billing because saved transcripts do not include billed_units for those calls.",
        },
        "telemetry_coverage": {
            "rows": len(rows),
            "measured_latency_rows": sum(1 for item in rows if item.get("measured_latency_seconds") is not None),
            "estimated_token_rows": sum(1 for item in rows if item.get("estimated_total_tokens") is not None),
            "measured_billed_token_rows": sum(1 for item in rows if item.get("billed_total_tokens") is not None),
        },
        "overall": _operational_aggregate(rows),
        "by_slice": _operational_by(rows, "slice"),
        "rows": rows,
        "note": (
            "The saved 60-case pilot did not record wall-clock latency or Cohere billed_units. "
            "Token and generation-cost values are estimates from saved query text, reloaded source-page text, "
            "and saved answer text. Future live runs now write measured latency in the transcript."
        ),
    }


def reviewer_challenge_metrics(run_dir: Path) -> dict[str, Any]:
    """Summarize measured Reviewer Agent challenge runs."""

    if not run_dir.exists():
        return {"status": "missing", "reason": f"reviewer run not found: {run_dir}"}
    rows = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name == "run_summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        reviewer = payload.get("reviewer", {}) if isinstance(payload.get("reviewer"), dict) else {}
        total = int(reviewer.get("total_citation_count") or 0)
        verified = int(reviewer.get("verified_citation_count") or 0)
        rows.append(
            {
                "case_id": payload.get("case_id", path.stem),
                "difficulty": payload.get("difficulty"),
                "latency_seconds": _elapsed_seconds(payload.get("started_at"), payload.get("completed_at")),
                "credibility_score": reviewer.get("credibility_score"),
                "verified_citation_count": verified,
                "total_citation_count": total,
                "reviewer_estimated_citation_precision": round(verified / total, 4) if total else None,
                "expectation_failure_count": len(payload.get("expectation_failures", []) or []),
                "reviewer_added_value": payload.get("reviewer_added_value", ""),
                "release_gate": reviewer.get("release_gate", ""),
            }
        )
    return {
        "status": "complete",
        "run_dir": str(run_dir),
        "case_count": len(rows),
        "overall": {
            "task_success_rate": _rate(0 == int(row.get("expectation_failure_count") or 0) for row in rows),
            "average_credibility_score": _mean(row.get("credibility_score") for row in rows),
            "reviewer_estimated_citation_precision": _weighted_precision(rows),
            "latency_seconds": _distribution(row.get("latency_seconds") for row in rows),
        },
        "rows": rows,
        "note": (
            "Reviewer challenge latency is measured from saved started_at/completed_at timestamps. "
            "Reviewer citation precision remains reviewer-estimated until an invalid-citation ground-truth set is curated."
        ),
    }


def limitations(ambiguous_registry: dict[str, Any]) -> list[str]:
    """Known limitations to keep the report honest."""

    items = [
        "The corpus is still synthetic and prototype-sized, so absolute scores should not be sold as production readiness.",
        "The 60-case pilot is much more defensible than the 15-case slice, but it is still one run per case.",
        "Citation precision is based on a 35-pair support-label sample, not a full SME-complete citation study.",
        "Saved full-pilot transcripts do not contain true billed_units, so token and cost values are estimates.",
        "Saved full-pilot transcripts do not contain true wall-clock latency; latency is measured for the reviewer challenge run only.",
    ]
    if ambiguous_registry:
        items.append(
            "Clarification-specific ambiguous cases are defined, but they need a fresh live run before claiming measured clarification accuracy."
        )
    return items


def defensible_eval_markdown(report: dict[str, Any]) -> str:
    """Render the client-presentable report."""

    score = report["scorecards"]
    ops = report["operational_metrics"]
    reviewer = report["reviewer_challenge_metrics"]
    design = report["sample_design"]
    lines = [
        "# Defence Agent Defensible Eval Report",
        "",
        "## Executive Readout",
        "- The earlier 15-case scorecard was useful for debugging, but too small and too clean for a client-facing trust claim.",
        "- The defensible scorecard uses the balanced 60-case pilot with six cases per scored slice.",
        "- ACL behavior is reported as a security invariant, not blended into stochastic model-quality metrics.",
        "- Ambiguous clarification cases are defined separately and marked as pending live scoring.",
        "- Operational metrics now include estimated token/cost data for the 60-case saved pilot and measured latency for the Reviewer Agent challenge run.",
        "",
        "## Sample Contract",
        f"- Scored registry: `{report['registry_path']}`",
        f"- Scored registry SHA-256: `{report['registry_sha256']}`",
        f"- Ambiguous clarification registry: `{report['ambiguous_registry_path']}`",
        f"- Ambiguous registry SHA-256: `{report.get('ambiguous_registry_sha256') or 'n/a'}`",
        f"- Scored cases: {design['scored_case_count']}",
        f"- Answerable cases: {design['answerable_case_count']}",
        f"- Unanswerable or refusal-expected cases: {design['unanswerable_case_count']}",
        f"- Hard/common cases: {design['hard_common_case_count']}",
        f"- Security-invariant cases: {design['security_invariant_case_count']}",
        f"- Contrast groups: {design['contrast_group_count']}",
        f"- Ambiguous clarification cases: {design['ambiguous_clarification_case_count']} ({design['ambiguous_clarification_status']})",
        "",
        _markdown_table(
            ["Slice", "Cases"],
            [[name, count] for name, count in design["scored_slice_counts"].items()],
        ),
        "",
        "Selection policy:",
        *[f"- {item}" for item in design["selection_policy"]],
        "",
        "## Layered Scorecards",
        _markdown_table(
            ["Layer", "N", "Task Success", "Retrieval R", "Retrieval P", "Citation R", "Citation P", "Answerability"],
            [
                _scorecard_row("All saved pilot", score["all_saved_pilot"]),
                _scorecard_row("Model quality only", score["model_quality_only"]),
                _scorecard_row("Answerable model quality", score["answerable_model_quality"]),
                _scorecard_row("Unanswerable evidence gap", score["unanswerable_evidence_gap"]),
                _scorecard_row("Hard/common", score["hard_common"]),
                _scorecard_row("Security invariants", score["security_invariants"]),
            ],
        ),
        "",
        "## Metrics By Slice",
        _markdown_table(
            ["Slice", "N", "Task Success", "Retrieval R", "Retrieval P", "Citation R", "Citation P", "Answerability"],
            [
                [
                    item["group"],
                    item["n"],
                    _fmt(item.get("end_to_end_success")),
                    _fmt(item.get("retrieval_recall")),
                    _fmt(item.get("retrieval_precision")),
                    _fmt(item.get("citation_recall")),
                    _fmt(item.get("citation_precision")),
                    _fmt(item.get("answerability_accuracy")),
                ]
                for item in score["by_slice"]
            ],
        ),
        "",
        "## Operational Metrics",
        f"- Pricing assumption: Command A input ${COMMAND_A_INPUT_USD_PER_1M}/1M tokens, output ${COMMAND_A_OUTPUT_USD_PER_1M}/1M tokens.",
        f"- Coverage: {ops['telemetry_coverage']['estimated_token_rows']} estimated-token rows; {ops['telemetry_coverage']['measured_latency_rows']} measured-latency rows in the 60-case saved pilot.",
        f"- Note: {ops['note']}",
        "",
        _operational_markdown(ops),
        "",
        "## Reviewer Agent Challenge Metrics",
        _reviewer_markdown(reviewer),
        "",
        "## Failure Examples",
        _markdown_table(
            ["Case", "Slice", "Failures"],
            [
                [item.get("case_id", ""), item.get("slice", ""), item.get("failures", "")]
                for item in score.get("failure_examples", [])
            ]
            or [["n/a", "n/a", "No failures recorded."]],
        ),
        "",
        "## Limitations",
        *[f"- {item}" for item in report["limitations"]],
    ]
    return "\n".join(lines) + "\n"


def _scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "task_success": _mean(row.get("end_to_end_success") for row in rows),
        "retrieval_recall": _mean(row.get("retrieval_recall") for row in rows),
        "retrieval_precision": _mean(row.get("retrieval_precision") for row in rows),
        "citation_recall": _mean(row.get("citation_recall") for row in rows),
        "citation_precision": _citation_precision(rows),
        "answerability_accuracy": _mean(row.get("answerability_accuracy") for row in rows),
        "refusal_accuracy": _mean(row.get("answerability_accuracy") for row in rows if row.get("expected_refusal")),
    }


def _scorecard_row(label: str, item: dict[str, Any]) -> list[Any]:
    return [
        label,
        item.get("n", 0),
        _fmt(item.get("task_success")),
        _fmt(item.get("retrieval_recall")),
        _fmt(item.get("retrieval_precision")),
        _fmt(item.get("citation_recall")),
        _fmt(item.get("citation_precision")),
        _fmt(item.get("answerability_accuracy")),
    ]


def _operational_row(row: dict[str, Any], outcome: dict[str, Any], text_lookup: dict[str, str]) -> dict[str, Any]:
    audit = outcome.get("final_answer_audit", {}) if isinstance(outcome.get("final_answer_audit"), dict) else {}
    retrieval = audit.get("retrieval", {}) if isinstance(audit.get("retrieval"), dict) else {}
    sources = [source for source in retrieval.get("sources_sent_to_answer", []) or [] if isinstance(source, dict)]
    source_text = "\n".join(text_lookup.get(_source_key(source), "") for source in sources)
    query = str(outcome.get("query", ""))
    answer = str(outcome.get("answer", ""))
    estimated_input = (
        _approx_tokens(query)
        + _approx_tokens(source_text)
        + GENERATION_PROMPT_OVERHEAD_TOKENS
    )
    estimated_output = _approx_tokens(answer)
    estimated_cost = _generation_cost(estimated_input, estimated_output)
    billed_input = _nested_number(outcome, ("operational_metrics", "billed_units", "input_tokens"))
    billed_output = _nested_number(outcome, ("operational_metrics", "billed_units", "output_tokens"))
    return {
        "case_id": row.get("case_id"),
        "slice": row.get("slice"),
        "search_count": row.get("search_count"),
        "documents_sent_to_model": row.get("documents_sent_to_model"),
        "citation_count": row.get("citation_count"),
        "measured_latency_seconds": _first_number(
            outcome.get("latency_seconds"),
            outcome.get("elapsed_seconds"),
            outcome.get("run_elapsed_seconds"),
            _nested_number(outcome, ("operational_metrics", "latency_seconds")),
        ),
        "estimated_input_tokens": estimated_input,
        "estimated_output_tokens": estimated_output,
        "estimated_total_tokens": estimated_input + estimated_output,
        "estimated_generation_cost_usd": round(estimated_cost, 6),
        "billed_input_tokens": billed_input,
        "billed_output_tokens": billed_output,
        "billed_total_tokens": (billed_input + billed_output) if billed_input is not None and billed_output is not None else None,
    }


def _page_text_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for page in load_document_pages():
        lookup[page.page_id] = page.text
        lookup[f"{page.doc_id}_page_{page.page_number:03d}"] = page.text
    return lookup


def _source_key(source: dict[str, Any]) -> str:
    if source.get("source_id"):
        return str(source["source_id"])
    if source.get("chunk_id"):
        return str(source["chunk_id"])
    doc_id = source.get("doc_id")
    page = source.get("page") or source.get("page_number")
    if doc_id and page:
        return f"{doc_id}_page_{int(page):03d}"
    return ""


def _operational_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "estimated_input_tokens": _distribution(row.get("estimated_input_tokens") for row in rows),
        "estimated_output_tokens": _distribution(row.get("estimated_output_tokens") for row in rows),
        "estimated_total_tokens": _distribution(row.get("estimated_total_tokens") for row in rows),
        "estimated_generation_cost_usd": _distribution(row.get("estimated_generation_cost_usd") for row in rows),
        "measured_latency_seconds": _distribution(row.get("measured_latency_seconds") for row in rows),
        "search_count": _distribution(row.get("search_count") for row in rows),
        "documents_sent_to_model": _distribution(row.get("documents_sent_to_model") for row in rows),
        "citation_count": _distribution(row.get("citation_count") for row in rows),
    }


def _operational_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    result = []
    for group, group_rows in sorted(grouped.items()):
        item = {"group": group, "n": len(group_rows)}
        item.update(_operational_aggregate(group_rows))
        result.append(item)
    return result


def _operational_markdown(report: dict[str, Any]) -> str:
    overall = report.get("overall", {})
    rows = [
        ["Estimated input tokens/query", _dist_cells(overall.get("estimated_input_tokens", {}))],
        ["Estimated output tokens/query", _dist_cells(overall.get("estimated_output_tokens", {}))],
        ["Estimated generation cost/query", _dist_cells(overall.get("estimated_generation_cost_usd", {}), money=True)],
        ["Searches/query", _dist_cells(overall.get("search_count", {}))],
        ["Documents sent/query", _dist_cells(overall.get("documents_sent_to_model", {}))],
        ["Citations/query", _dist_cells(overall.get("citation_count", {}))],
    ]
    slice_rows = [
        [
            item["group"],
            item["n"],
            _fmt(item.get("estimated_generation_cost_usd", {}).get("mean"), money=True),
            _fmt(item.get("estimated_generation_cost_usd", {}).get("p95"), money=True),
            _fmt(item.get("estimated_input_tokens", {}).get("mean")),
            _fmt(item.get("estimated_output_tokens", {}).get("mean")),
        ]
        for item in report.get("by_slice", [])
    ]
    return "\n".join(
        [
            _markdown_table(["Metric", "Mean / Median / P95 / Total"], rows),
            "",
            _markdown_table(["Slice", "N", "Avg Cost", "P95 Cost", "Avg Input Tok", "Avg Output Tok"], slice_rows),
        ]
    )


def _reviewer_markdown(report: dict[str, Any]) -> str:
    if report.get("status") != "complete":
        return f"- Missing: {report.get('reason', 'not available')}"
    overall = report["overall"]
    latency = overall.get("latency_seconds", {})
    rows = [
        ["Cases", report.get("case_count")],
        ["Task success rate", _fmt(overall.get("task_success_rate"))],
        ["Average credibility score", _fmt(overall.get("average_credibility_score"))],
        ["Reviewer-estimated citation precision", _fmt(overall.get("reviewer_estimated_citation_precision"))],
        ["Latency mean / median / p95 / max", _latency_cells(latency)],
    ]
    case_rows = [
        [
            item.get("case_id", ""),
            item.get("difficulty", ""),
            _fmt(item.get("latency_seconds")),
            _fmt(item.get("credibility_score")),
            _fmt(item.get("reviewer_estimated_citation_precision")),
            item.get("release_gate", ""),
        ]
        for item in report.get("rows", [])
    ]
    return "\n".join(
        [
            f"- Reviewer run: `{report['run_dir']}`",
            f"- Note: {report['note']}",
            "",
            _markdown_table(["Metric", "Value"], rows),
            "",
            _markdown_table(["Case", "Difficulty", "Latency s", "Credibility", "Citation P", "Gate"], case_rows),
        ]
    )


def _contrast_groups(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        case_id = str(case.get("id", ""))
        parent_id = str(case.get("parent_id", "")) or case_id.split("__variant_", 1)[0]
        groups[parent_id].append(case_id)
    return [
        {"group_id": group_id, "case_count": len(case_ids), "case_ids": sorted(case_ids)}
        for group_id, case_ids in sorted(groups.items())
        if len(case_ids) > 1
    ]


def _elapsed_seconds(started_at: Any, completed_at: Any) -> float | None:
    if not started_at or not completed_at:
        return None
    try:
        start = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((end - start).total_seconds(), 4)


def _weighted_precision(rows: list[dict[str, Any]]) -> float | None:
    total = sum(int(row.get("total_citation_count") or 0) for row in rows)
    if not total:
        return None
    verified = sum(int(row.get("verified_citation_count") or 0) for row in rows)
    return round(verified / total, 4)


def _citation_precision(rows: list[dict[str, Any]]) -> float | None:
    total = sum(int(row.get("citation_precision_label_count") or 0) for row in rows)
    if not total:
        return None
    supported = sum(int(row.get("citation_precision_supported_count") or 0) for row in rows)
    return round(supported / total, 4)


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None, "total": None}
    ordered = sorted(numeric)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return {
        "n": len(numeric),
        "mean": round(mean(numeric), 4),
        "median": round(median(numeric), 4),
        "p95": round(ordered[p95_index], 4),
        "max": round(max(numeric), 4),
        "total": round(sum(numeric), 4),
    }


def _dist_cells(item: dict[str, Any], *, money: bool = False) -> str:
    return (
        f"{_fmt(item.get('mean'), money=money)} / "
        f"{_fmt(item.get('median'), money=money)} / "
        f"{_fmt(item.get('p95'), money=money)} / "
        f"{_fmt(item.get('total'), money=money)}"
    )


def _latency_cells(item: dict[str, Any]) -> str:
    return (
        f"{_fmt(item.get('mean'))} / "
        f"{_fmt(item.get('median'))} / "
        f"{_fmt(item.get('p95'))} / "
        f"{_fmt(item.get('max'))}"
    )


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(re.findall(r"\S+", text)) * APPROX_TOKEN_MULTIPLIER))


def _generation_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens * COMMAND_A_INPUT_USD_PER_1M)
        + (output_tokens * COMMAND_A_OUTPUT_USD_PER_1M)
    ) / 1_000_000


def _nested_number(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, int | float) else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, int | float):
            return float(value)
    return None


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return round(mean(numeric), 4) if numeric else None


def _rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    if not items:
        return None
    return round(sum(1 for item in items if item) / len(items), 4)


def _fmt(value: Any, *, money: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"${value:.4f}" if money else f"{value:.3f}"
    if isinstance(value, int):
        return f"${value:.4f}" if money else str(value)
    return str(value)


def _cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = ["| " + " | ".join(_cell(header) for header in headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    rendered.extend("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows)
    return "\n".join(rendered)
