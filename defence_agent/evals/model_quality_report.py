"""Curated model-quality eval reporting for Defence Agent.

This module keeps deterministic access-control checks out of the model-quality
scorecard. It focuses on the parts the model can actually affect: retrieval,
reranking, answerability, generation, citations, and reviewer-agent gating.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import yaml

from defence_agent.auth.context import DEFAULT_PERSONA_ID, DEMO_USERS
from defence_agent.auth.policy import policy_engine
from defence_agent.evals.harness import (
    DEFAULT_TRANSCRIPTS_DIR,
    EvalHarnessError,
    load_registry,
    summarize_transcript_runs,
)
from defence_agent.retrieval import chroma_index
from defence_agent.retrieval.chunks import pages_by_id, retrieval_chunks_for_pages
from defence_agent.retrieval.document_pages import load_document_pages


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_QUALITY_REGISTRY = PACKAGE_ROOT / "data" / "evals" / "model_quality_eval_bank.yaml"
DEFAULT_FULL_PILOT_TRANSCRIPT_DIR = DEFAULT_TRANSCRIPTS_DIR / "eval_full_registry_20260514_020915"
DEFAULT_MODEL_QUALITY_REPORT_JSON = PACKAGE_ROOT / "data" / "evals" / "reports" / "model_quality_eval_report.json"
DEFAULT_MODEL_QUALITY_REPORT_MD = PACKAGE_ROOT / "data" / "evals" / "reports" / "model_quality_eval_report.md"
DEFAULT_CITATION_LABELS = PACKAGE_ROOT / "data" / "evals" / "citation_precision_full_pilot_labels.yaml"
DEFAULT_REVIEWER_RUN_ROOT = PACKAGE_ROOT / "data" / "evals" / "reviewer_challenge_runs"

RETRIEVAL_TOP_KS = (5, 8)
_LABEL_PAIR_RE = re.compile(
    r"::citation:(?P<citation>\d+):source:(?P<source>\d+):span:(?P<start>\d+)-(?P<end>\d+):source:(?P<source_id>.+)$"
)


def build_model_quality_report(
    *,
    registry_path: Path | str = DEFAULT_MODEL_QUALITY_REGISTRY,
    transcript_dir: Path | str = DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
    citation_label_path: Path | str | None = DEFAULT_CITATION_LABELS,
    include_citation_support: bool = True,
    include_rerank_uplift: bool = False,
    reviewer_run_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build the full curated model-quality report."""

    registry = load_registry(registry_path)
    transcript_paths = transcript_paths_for_registry(registry, Path(transcript_dir))
    scored = summarize_transcript_runs(
        registry_path=registry_path,
        transcript_paths=transcript_paths,
        include_citation_support=include_citation_support,
        citation_label_path=citation_label_path,
    )
    report = {
        "schema_version": "model_quality_eval_report.v1",
        "registry_path": str(registry_path),
        "transcript_dir": str(transcript_dir),
        "scored_transcript_count": len(scored["rows"]),
        "metric_contract": {
            "acl_enforcement": "excluded_from_model_quality_metrics",
            "reason": (
                "Access control is deterministic metadata filtering. The model should never receive "
                "restricted evidence, so ACL enforcement is tested as an invariant, not as model behavior."
            ),
        },
        "summary": _model_quality_summary(scored),
        "base_harness_report": scored,
        "rerank_uplift": (
            retrieval_uplift_report(registry_path=registry_path, top_ks=RETRIEVAL_TOP_KS)
            if include_rerank_uplift
            else {"status": "skipped", "reason": "live retrieval diagnostics were not requested"}
        ),
        "bilingual_citations": bilingual_citation_report(
            scored,
            transcript_dir=Path(transcript_dir),
            citation_label_path=citation_label_path,
        ),
        "reviewer_agent": reviewer_agent_report(reviewer_run_dir),
    }
    report["markdown"] = model_quality_markdown(report)
    return report


def transcript_paths_for_registry(registry: dict[str, Any], transcript_dir: Path) -> list[Path]:
    """Return transcript files for the exact curated case ids."""

    case_ids = {
        str(case.get("id", "")).strip()
        for case in registry.get("cases", []) or []
        if isinstance(case, dict) and str(case.get("id", "")).strip()
    }
    paths = [transcript_dir / f"{case_id}.json" for case_id in sorted(case_ids)]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise EvalHarnessError(f"No curated transcript files found under {transcript_dir}")
    return existing


def write_model_quality_report(report: dict[str, Any], *, json_path: Path, md_path: Path) -> None:
    """Write JSON and Markdown reports."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    markdown = str(payload.pop("markdown", ""))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


def _model_quality_summary(scored: dict[str, Any]) -> dict[str, Any]:
    rows = scored["rows"]
    refusal_rows = [row for row in rows if row.get("expected_refusal")]
    answer_rows = [row for row in rows if not row.get("expected_refusal")]
    overall = scored["overall"]
    return {
        "sample_size": overall.get("n", len(rows)),
        "answerable_case_count": len(answer_rows),
        "refusal_case_count": len(refusal_rows),
        "end_to_end_success": overall.get("end_to_end_success"),
        "retrieval_recall": overall.get("retrieval_recall"),
        "retrieval_precision": overall.get("retrieval_precision"),
        "facet_recall": overall.get("facet_recall"),
        "generation_fact_recall": overall.get("generation_fact_recall"),
        "forbidden_fact_absence": overall.get("forbidden_fact_absence"),
        "answerability_accuracy": overall.get("answerability_accuracy"),
        "refusal_accuracy": _mean(row.get("answerability_accuracy") for row in refusal_rows),
        "answerable_accuracy": _mean(row.get("answerability_accuracy") for row in answer_rows),
        "citation_recall": overall.get("citation_recall"),
        "citation_precision": overall.get("citation_precision"),
        "citation_precision_label_count": overall.get("citation_precision_label_count"),
        "citation_source_precision_proxy": overall.get("citation_source_precision_proxy"),
        "citation_support_precision_lexical": overall.get("citation_support_precision_lexical"),
        "slices": dict(Counter(str(row.get("slice", "unknown")) for row in rows)),
    }


def retrieval_uplift_report(
    *,
    registry_path: Path | str = DEFAULT_MODEL_QUALITY_REGISTRY,
    top_ks: Iterable[int] = RETRIEVAL_TOP_KS,
) -> dict[str, Any]:
    """Run live retrieval diagnostics and compare before/after reranking."""

    registry = load_registry(registry_path)
    cases = [
        case
        for case in registry.get("cases", []) or []
        if isinstance(case, dict) and case.get("expected_doc_ids") and not case.get("expected_refusal")
    ]
    if not cases:
        return {"status": "skipped", "reason": "no retrieval-labeled cases"}

    chroma_index.build_index(force=False)
    pages = load_document_pages()
    page_lookup = pages_by_id(pages)
    chunks = retrieval_chunks_for_pages(pages, chroma_index.get_settings().chunk_strategy)
    collection = chroma_index._client().get_collection(chroma_index._collection_name())

    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(_retrieval_uplift_rows(case, chunks=chunks, collection=collection, page_lookup=page_lookup, top_ks=top_ks))
    return {
        "status": "complete",
        "case_count": len(cases),
        "top_ks": list(top_ks),
        "overall": _aggregate_retrieval_rows(rows),
        "by_case": _retrieval_case_rows(rows),
        "rows": rows,
        "notes": [
            "bm25_pre_rerank is the lexical baseline with no embedding model.",
            "vector_pre_rerank is the embedding-only baseline before Cohere Rerank.",
            "hybrid_pre_rerank merges lexical and embedding candidates before Cohere Rerank.",
            "hybrid_post_rerank is the current production-style retrieval path.",
        ],
    }


def _retrieval_uplift_rows(
    case: dict[str, Any],
    *,
    chunks: list[Any],
    collection: Any,
    page_lookup: dict[str, Any],
    top_ks: Iterable[int],
) -> list[dict[str, Any]]:
    query = str(case.get("query", ""))
    persona_id = str(case.get("persona_id", DEFAULT_PERSONA_ID))
    auth = DEMO_USERS.get(persona_id, DEMO_USERS[DEFAULT_PERSONA_ID])
    allowed_access = list(policy_engine.acl_filter(auth).allowed_classifications)
    language = chroma_index._normalize_language_filter(str(case.get("language", "any")))
    status_filter = chroma_index._normalize_status_filter(str(case.get("status_filter", "approved")))
    max_top_k = max(int(k) for k in top_ks)
    vector_candidates, _excluded = chroma_index._vector_candidates(
        query=query,
        auth=auth,
        collection=collection,
        top_k=max_top_k,
        allowed_access=allowed_access,
        status_filter=status_filter,
        language=language,
    )
    bm25_candidates = chroma_index._bm25_candidates(
        query=query,
        chunks=chunks,
        auth=auth,
        top_k=max_top_k,
        status_filter=status_filter,
        language=language,
    )
    hybrid_candidates = chroma_index._merge_retrieval_candidates(vector_candidates + bm25_candidates)
    stages = {
        "bm25_pre_rerank": bm25_candidates,
        "vector_pre_rerank": vector_candidates,
        "hybrid_pre_rerank": hybrid_candidates,
        "bm25_post_rerank": chroma_index._rerank(query, bm25_candidates),
        "vector_post_rerank": chroma_index._rerank(query, vector_candidates),
        "hybrid_post_rerank": chroma_index._rerank(query, hybrid_candidates),
    }

    rows: list[dict[str, Any]] = []
    for top_k in top_ks:
        for stage_name, ranked in stages.items():
            selected = chroma_index._select_sources(ranked, int(top_k), language=language)
            promoted = chroma_index._promote_sources_to_parent_pages(selected, page_lookup)
            rows.append(
                {
                    "case_id": str(case.get("id", "")),
                    "slice": str(case.get("eval_slice", "")),
                    "query": query,
                    "top_k": int(top_k),
                    "stage": stage_name,
                    "candidate_count": len(ranked),
                    **_score_ranked_sources(promoted, case),
                }
            )
    return rows


def _score_ranked_sources(sources: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    expected_groups = _expected_doc_groups(case)
    acceptable = set().union(*expected_groups) if expected_groups else set()
    doc_ids = [str(source.get("doc_id", "")) for source in sources if source.get("doc_id")]
    relevances = [1 if doc_id in acceptable else 0 for doc_id in doc_ids]
    novelty_relevances = _novel_group_relevances(doc_ids, expected_groups)
    return {
        "doc_ids": doc_ids,
        "precision": round(sum(relevances) / len(relevances), 4) if relevances else None,
        "recall": _group_recall(expected_groups, set(doc_ids)),
        "mrr": _mrr(relevances),
        "ndcg": _ndcg(novelty_relevances, ideal_count=len(expected_groups)),
    }


def _aggregate_retrieval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["stage"]), int(row["top_k"])), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (stage, top_k), group_rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        aggregates.append(
            {
                "stage": stage,
                "top_k": top_k,
                "n": len(group_rows),
                "precision": _mean(row.get("precision") for row in group_rows),
                "recall": _mean(row.get("recall") for row in group_rows),
                "mrr": _mean(row.get("mrr") for row in group_rows),
                "ndcg": _mean(row.get("ndcg") for row in group_rows),
            }
        )
    return aggregates


def _retrieval_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted_stages = {"bm25_pre_rerank", "vector_pre_rerank", "hybrid_pre_rerank", "hybrid_post_rerank"}
    return [
        row
        for row in rows
        if row["stage"] in wanted_stages and int(row["top_k"]) == max(RETRIEVAL_TOP_KS)
    ]


def reviewer_agent_report(reviewer_run_dir: Path | str | None = None) -> dict[str, Any]:
    """Summarize reviewer-agent outputs without treating them as ground truth."""

    run_dir = Path(reviewer_run_dir) if reviewer_run_dir else _latest_reviewer_run_dir()
    if run_dir is None or not run_dir.exists():
        return {"status": "missing", "reason": "no reviewer challenge run found"}
    rows = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name == "run_summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        reviewer = payload.get("reviewer", {}) if isinstance(payload.get("reviewer"), dict) else {}
        rows.append(
            {
                "case_id": payload.get("case_id", path.stem),
                "reviewer_added_value": payload.get("reviewer_added_value", ""),
                "release_gate": reviewer.get("release_gate", ""),
                "credibility_score": reviewer.get("credibility_score"),
                "verified_citation_count": reviewer.get("verified_citation_count"),
                "unverified_citation_count": reviewer.get("unverified_citation_count"),
                "total_citation_count": reviewer.get("total_citation_count"),
                "expectation_failure_count": len(payload.get("expectation_failures", []) or []),
            }
        )
    total_reviewed = sum(int(row.get("total_citation_count") or 0) for row in rows)
    verified = sum(int(row.get("verified_citation_count") or 0) for row in rows)
    unverified = sum(int(row.get("unverified_citation_count") or 0) for row in rows)
    return {
        "status": "complete",
        "run_dir": str(run_dir),
        "case_count": len(rows),
        "average_credibility_score": _mean(row.get("credibility_score") for row in rows),
        "reviewer_estimated_citation_precision": round(verified / total_reviewed, 4) if total_reviewed else None,
        "verified_citation_count": verified,
        "unverified_citation_count": unverified,
        "total_reviewed_citation_count": total_reviewed,
        "human_gate_count": sum(1 for row in rows if row.get("release_gate") == "human_continue_or_stop_required"),
        "value_counts": dict(Counter(str(row.get("reviewer_added_value", "")) for row in rows)),
        "rows": rows,
        "note": (
            "This summarizes reviewer-agent lift and estimated citation support. It is not yet a calibrated "
            "reviewer-performance metric because invalid-citation ground truth has not been curated."
        ),
    }


def bilingual_citation_report(
    scored: dict[str, Any],
    *,
    transcript_dir: Path,
    citation_label_path: Path | str | None,
) -> dict[str, Any]:
    """Summarize bilingual citation quality and evidence-language behavior."""

    rows = [row for row in scored.get("rows", []) if row.get("slice") == "bilingual"]
    if not rows:
        return {"status": "skipped", "reason": "no bilingual cases in curated scorecard"}

    payloads = _bilingual_transcript_payloads(rows, transcript_dir)
    supported_examples, unsupported_examples = _bilingual_label_examples(rows, payloads, citation_label_path)
    comparison = {
        "bilingual": _metric_snapshot(rows),
        "non_bilingual": _metric_snapshot([row for row in scored.get("rows", []) if row.get("slice") != "bilingual"]),
    }

    return {
        "status": "complete",
        "case_count": len(rows),
        "comparison": comparison,
        "source_language_mix": _bilingual_source_language_mix(rows, payloads),
        "best_supported_citations": supported_examples[:5],
        "unsupported_citation_examples": unsupported_examples[:5],
        "finding": _bilingual_finding(comparison),
        "note": (
            "Bilingual citation recall checks whether claims are cited. Semantic citation precision checks whether "
            "the cited source actually supports the cited claim span."
        ),
    }


def _bilingual_transcript_payloads(rows: list[dict[str, Any]], transcript_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        transcript_file = str(row.get("transcript_file", ""))
        path = transcript_dir / transcript_file
        if not case_id or not transcript_file or not path.exists():
            continue
        payloads[case_id] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def _bilingual_label_examples(
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    citation_label_path: Path | str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_ids = {str(row.get("case_id", "")) for row in rows}
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for entry in _citation_label_entries(citation_label_path):
        case_id = _case_id_from_label_key(str(entry.get("key", "")))
        if case_id not in case_ids:
            continue
        example = _citation_label_example(entry, payloads.get(case_id, {}))
        if example.get("supported"):
            supported.append(example)
        else:
            unsupported.append(example)
    return supported, unsupported


def _citation_label_entries(citation_label_path: Path | str | None) -> list[dict[str, Any]]:
    if citation_label_path is None:
        return []
    path = Path(citation_label_path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("labels", raw) if isinstance(raw, dict) else raw
    return [entry for entry in entries or [] if isinstance(entry, dict) and str(entry.get("key", "")).strip()]


def _case_id_from_label_key(key: str) -> str:
    transcript_name = key.split("::", 1)[0]
    return Path(transcript_name).stem


def _citation_label_example(entry: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    key = str(entry.get("key", ""))
    match = _LABEL_PAIR_RE.search(key)
    citation_index = int(match.group("citation")) if match else None
    source_index = int(match.group("source")) if match else None
    source_id = match.group("source_id") if match else ""
    citation = _citation_at(payload, citation_index)
    source = _source_at(citation, source_index)
    cited_text = str(citation.get("text", ""))
    if not cited_text and match:
        answer = str(payload.get("answer", ""))
        cited_text = answer[int(match.group("start")) : int(match.group("end"))]
    return {
        "case_id": _case_id_from_label_key(key),
        "supported": bool(entry.get("supported")),
        "citation_index": citation_index,
        "source_index": source_index,
        "source_id": source_id or str(source.get("source_id", "")),
        "language": str(source.get("language", "")) or _language_from_source_id(source_id),
        "page": source.get("page"),
        "title": str(source.get("title", "")),
        "cited_text": _shorten(cited_text, 180),
        "note": str(entry.get("notes", "")),
    }


def _citation_at(payload: dict[str, Any], index: int | None) -> dict[str, Any]:
    if index is None:
        return {}
    citations = _dict(payload.get("final_answer_audit")).get("citations", []) or []
    if 0 <= index < len(citations) and isinstance(citations[index], dict):
        return citations[index]
    return {}


def _source_at(citation: dict[str, Any], index: int | None) -> dict[str, Any]:
    if index is None:
        return {}
    sources = citation.get("sources", []) or []
    if 0 <= index < len(sources) and isinstance(sources[index], dict):
        return sources[index]
    return {}


def _language_from_source_id(source_id: str) -> str:
    if "-FR" in source_id:
        return "fr"
    if "-EN" in source_id:
        return "en"
    return ""


def _bilingual_source_language_mix(
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id", ""))
        payload = payloads.get(case_id, {})
        unique_sources: dict[str, str] = {}
        for citation in _dict(payload.get("final_answer_audit")).get("citations", []) or []:
            if not isinstance(citation, dict):
                continue
            for source in citation.get("sources", []) or []:
                if not isinstance(source, dict):
                    continue
                source_id = str(source.get("source_id", ""))
                if source_id:
                    unique_sources[source_id] = str(source.get("language", "")) or _language_from_source_id(source_id)
        language_counts = Counter(language or "unknown" for language in unique_sources.values())
        summaries.append(
            {
                "case_id": case_id,
                "answer_language": "fr" if str(row.get("query", "")).lower().startswith("quelles ") else "en",
                "unique_cited_source_count": len(unique_sources),
                "unique_cited_sources_by_language": dict(sorted(language_counts.items())),
                "cited_doc_ids": row.get("cited_doc_ids", []),
            }
        )
    return summaries


def _metric_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_count = sum(int(row.get("citation_precision_label_count", 0) or 0) for row in rows)
    supported_count = sum(int(row.get("citation_precision_supported_count", 0) or 0) for row in rows)
    return {
        "case_count": len(rows),
        "end_to_end_success": _mean(row.get("end_to_end_success") for row in rows),
        "answerability_accuracy": _mean(row.get("answerability_accuracy") for row in rows),
        "retrieval_recall": _mean(row.get("retrieval_recall") for row in rows),
        "retrieval_precision": _mean(row.get("retrieval_precision") for row in rows),
        "citation_recall": _mean(row.get("citation_recall") for row in rows),
        "semantic_citation_precision": round(supported_count / label_count, 4) if label_count else None,
        "semantic_label_count": label_count,
        "citation_source_precision_proxy": _mean(row.get("citation_source_precision_proxy") for row in rows),
        "lexical_citation_support_precision": _mean(row.get("citation_support_precision_lexical") for row in rows),
    }


def _bilingual_finding(comparison: dict[str, dict[str, Any]]) -> str:
    bilingual = comparison.get("bilingual", {})
    non_bilingual = comparison.get("non_bilingual", {})
    bilingual_precision = bilingual.get("semantic_citation_precision")
    non_bilingual_precision = non_bilingual.get("semantic_citation_precision")
    bilingual_recall = bilingual.get("citation_recall")
    retrieval_recall = bilingual.get("retrieval_recall")
    if isinstance(bilingual_precision, int | float) and isinstance(non_bilingual_precision, int | float):
        if bilingual_precision < non_bilingual_precision:
            return (
                "Bilingual retrieval and citation recall are strong "
                f"(retrieval_recall={_fmt(retrieval_recall)}, citation_recall={_fmt(bilingual_recall)}), "
                "but labeled semantic citation precision is lower than the non-bilingual cohort "
                f"({_fmt(bilingual_precision)} vs {_fmt(non_bilingual_precision)}). The weakness is support precision, "
                "not source discovery."
            )
        return (
            "The labeled bilingual sample is not worse than the non-bilingual cohort on semantic citation precision, "
            f"with retrieval_recall={_fmt(retrieval_recall)} and citation_recall={_fmt(bilingual_recall)}."
        )
    return "Bilingual citation recall is available, but semantic citation precision needs more support labels."


def model_quality_markdown(report: dict[str, Any]) -> str:
    """Render a concise Markdown report."""

    summary = report["summary"]
    lines = [
        "# Defence Agent Model Quality Eval Report",
        "",
        "## Metric Contract",
        "- ACL enforcement is excluded from model-quality metrics.",
        "- Access control is a deterministic metadata invariant: restricted evidence should not reach model context.",
        "- Model-quality scoring focuses on retrieval, reranking, answerability, generation, citation quality, and reviewer-agent lift.",
        "",
        "## Curated Dataset",
        f"- Scored transcripts: {report['scored_transcript_count']}",
        f"- Answerable cases: {summary['answerable_case_count']}",
        f"- Insufficient-evidence refusal cases: {summary['refusal_case_count']}",
        f"- Slices: {', '.join(f'{key}={value}' for key, value in sorted(summary['slices'].items()))}",
        "",
        "## Overall Model Metrics",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["end_to_end_success", _fmt(summary.get("end_to_end_success"))],
                ["retrieval_recall", _fmt(summary.get("retrieval_recall"))],
                ["retrieval_precision", _fmt(summary.get("retrieval_precision"))],
                ["facet_recall", _fmt(summary.get("facet_recall"))],
                ["generation_fact_recall", _fmt(summary.get("generation_fact_recall"))],
                ["forbidden_fact_absence", _fmt(summary.get("forbidden_fact_absence"))],
                ["answerability_accuracy", _fmt(summary.get("answerability_accuracy"))],
                ["refusal_accuracy", _fmt(summary.get("refusal_accuracy"))],
                ["answerable_accuracy", _fmt(summary.get("answerable_accuracy"))],
            ],
        ),
        "",
        "## Citation Metrics",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["citation_recall", _fmt(summary.get("citation_recall"))],
                ["semantic_citation_precision", _fmt(summary.get("citation_precision"))],
                ["semantic_label_count", _fmt(summary.get("citation_precision_label_count"))],
                ["citation_source_precision_proxy", _fmt(summary.get("citation_source_precision_proxy"))],
                ["lexical_citation_support_precision", _fmt(summary.get("citation_support_precision_lexical"))],
            ],
        ),
        "",
        "## Bilingual Citation Analysis",
        _bilingual_markdown(report.get("bilingual_citations", {})),
        "",
        "## Rerank Uplift",
        _rerank_markdown(report.get("rerank_uplift", {})),
        "",
        "## Reviewer Agent Lift",
        _reviewer_markdown(report.get("reviewer_agent", {})),
        "",
        "## Layered Metrics By Slice",
        _slice_markdown(report["base_harness_report"]),
        "",
        "## Failure Examples",
        _failure_markdown(report["base_harness_report"]),
        "",
        "## Metric Notes",
        *[f"- {note}" for note in report["base_harness_report"].get("notes", [])],
    ]
    return "\n".join(lines)


def _rerank_markdown(report: dict[str, Any]) -> str:
    if report.get("status") != "complete":
        return f"- Skipped: {report.get('reason', 'not available')}"
    aggregates = report.get("overall", [])
    stage_lookup = {(item["stage"], item["top_k"]): item for item in aggregates}
    rows = []
    for top_k in report.get("top_ks", []):
        hybrid_pre = stage_lookup.get(("hybrid_pre_rerank", top_k), {})
        hybrid_post = stage_lookup.get(("hybrid_post_rerank", top_k), {})
        rows.extend(
            [
                [
                    item["stage"],
                    item["top_k"],
                    _fmt(item.get("precision")),
                    _fmt(item.get("recall")),
                ]
                for item in aggregates
                if item["top_k"] == top_k
            ]
        )
        if hybrid_pre and hybrid_post:
            rows.append(
                [
                    "hybrid_rerank_uplift",
                    top_k,
                    _delta(hybrid_post.get("precision"), hybrid_pre.get("precision")),
                    _delta(hybrid_post.get("recall"), hybrid_pre.get("recall")),
                ]
            )
    return "\n".join(
        [
            "- Presentation-facing rerank uplift is reported as precision@K and recall@K.",
            "- MRR and nDCG remain in the JSON diagnostics, but they are not headline trust metrics.",
            _markdown_table(["Stage", "K", "Precision@K", "Recall@K"], rows),
        ]
    )


def _bilingual_markdown(report: dict[str, Any]) -> str:
    if report.get("status") != "complete":
        return f"- Skipped: {report.get('reason', 'not available')}"
    comparison = report.get("comparison", {})
    cohort_rows = []
    for name in ("bilingual", "non_bilingual"):
        item = comparison.get(name, {})
        cohort_rows.append(
            [
                name,
                item.get("case_count", 0),
                _fmt(item.get("end_to_end_success")),
                _fmt(item.get("retrieval_recall")),
                _fmt(item.get("retrieval_precision")),
                _fmt(item.get("citation_recall")),
                _fmt(item.get("semantic_citation_precision")),
                _fmt(item.get("semantic_label_count")),
            ]
        )
    language_rows = [
        [
            item.get("case_id", ""),
            item.get("answer_language", ""),
            item.get("unique_cited_source_count", 0),
            json.dumps(item.get("unique_cited_sources_by_language", {}), sort_keys=True),
        ]
        for item in report.get("source_language_mix", [])
    ]
    supported_rows = [
        [
            item.get("case_id", ""),
            item.get("language", ""),
            item.get("source_id", ""),
            item.get("cited_text", ""),
            item.get("note", ""),
        ]
        for item in report.get("best_supported_citations", [])
    ]
    unsupported_rows = [
        [
            item.get("case_id", ""),
            item.get("language", ""),
            item.get("source_id", ""),
            item.get("cited_text", ""),
            item.get("note", ""),
        ]
        for item in report.get("unsupported_citation_examples", [])
    ]
    sections = [
        f"- Finding: {report.get('finding', '')}",
        f"- Note: {report.get('note', '')}",
        "",
        _markdown_table(
            ["Cohort", "N", "Pass", "Retrieval R", "Retrieval P", "Citation R", "Semantic Citation P", "Labels"],
            cohort_rows,
        ),
        "",
        _markdown_table(["Case", "Answer Lang", "Unique Cited Sources", "Source Lang Mix"], language_rows),
    ]
    if supported_rows:
        sections.extend(
            [
                "",
                "Best supported bilingual citation labels:",
                _markdown_table(["Case", "Lang", "Source", "Cited Text", "Why Supported"], supported_rows),
            ]
        )
    if unsupported_rows:
        sections.extend(
            [
                "",
                "Unsupported bilingual citation labels to target with the Reviewer Agent:",
                _markdown_table(["Case", "Lang", "Source", "Cited Text", "Why Unsupported"], unsupported_rows),
            ]
        )
    return "\n".join(sections)


def _reviewer_markdown(report: dict[str, Any]) -> str:
    if report.get("status") != "complete":
        return f"- Skipped: {report.get('reason', 'not available')}"
    return "\n".join(
        [
            f"- Reviewer run: `{report['run_dir']}`",
            f"- Cases: {report['case_count']}",
            f"- Average credibility score: {_fmt(report.get('average_credibility_score'))}",
            f"- Reviewer-estimated citation precision: {_fmt(report.get('reviewer_estimated_citation_precision'))}",
            f"- Verified / total reviewed citations: {report['verified_citation_count']} / {report['total_reviewed_citation_count']}",
            f"- Human-gate count: {report['human_gate_count']}",
            f"- Note: {report['note']}",
        ]
    )


def _slice_markdown(report: dict[str, Any]) -> str:
    rows = []
    for item in report.get("by_slice", []):
        rows.append(
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
        )
    return _markdown_table(
        ["Slice", "N", "Pass", "Retrieval R", "Retrieval P", "Citation R", "Citation P", "Answerability"],
        rows,
    )


def _failure_markdown(report: dict[str, Any]) -> str:
    examples = report.get("failure_examples", [])
    if not examples:
        return "- No failures in scored curated transcript outcomes."
    return _markdown_table(
        ["Run", "Case", "Slice", "Failures"],
        [
            [
                item.get("transcript_run", ""),
                item.get("case_id", ""),
                item.get("slice", ""),
                item.get("failures", ""),
            ]
            for item in examples
        ],
    )


def _latest_reviewer_run_dir() -> Path | None:
    if not DEFAULT_REVIEWER_RUN_ROOT.exists():
        return None
    dirs = [path for path in DEFAULT_REVIEWER_RUN_ROOT.iterdir() if path.is_dir() and (path / "run_summary.json").exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: (_reviewer_case_file_count(path), path.stat().st_mtime))


def _reviewer_case_file_count(path: Path) -> int:
    return sum(1 for item in path.glob("*.json") if item.name != "run_summary.json")


def _expected_doc_groups(case: dict[str, Any]) -> list[set[str]]:
    expected = _string_set(case.get("expected_doc_ids"))
    allowed_alt = _string_set(case.get("allowed_alt_doc_ids"))
    if not expected:
        return []
    if len(expected) == 1 and allowed_alt:
        return [expected | allowed_alt]
    return [{doc_id} for doc_id in sorted(expected)]


def _group_recall(groups: list[set[str]], observed: set[str]) -> float | None:
    if not groups:
        return None
    return round(sum(1 for group in groups if group & observed) / len(groups), 4)


def _mrr(relevances: list[int]) -> float | None:
    for index, relevance in enumerate(relevances, start=1):
        if relevance:
            return round(1.0 / index, 4)
    return 0.0 if relevances else None


def _ndcg(relevances: list[int], *, ideal_count: int) -> float | None:
    if not relevances:
        return None
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))
    ideal_rels = [1] * min(ideal_count, len(relevances))
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal_rels))
    return round(dcg / idcg, 4) if idcg else None


def _novel_group_relevances(doc_ids: list[str], groups: list[set[str]]) -> list[int]:
    """Score ranking gain once per expected evidence group."""

    if not groups:
        return [0 for _ in doc_ids]
    seen_groups: set[int] = set()
    relevances: list[int] = []
    for doc_id in doc_ids:
        group_index = next((index for index, group in enumerate(groups) if doc_id in group and index not in seen_groups), None)
        if group_index is None:
            relevances.append(0)
            continue
        seen_groups.add(group_index)
        relevances.append(1)
    return relevances


def _string_set(value: Any) -> set[str]:
    return {str(item) for item in value or [] if str(item)}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return round(mean(numeric), 4) if numeric else None


def _delta(after: Any, before: Any) -> str:
    if not isinstance(after, int | float) or not isinstance(before, int | float):
        return "n/a"
    value = round(float(after) - float(before), 4)
    return f"{value:+.4f}"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _shorten(value: str, max_length: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = ["| " + " | ".join(_cell(header) for header in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rendered.extend("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows)
    return "\n".join(rendered)
