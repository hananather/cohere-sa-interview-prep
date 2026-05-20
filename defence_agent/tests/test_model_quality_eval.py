from __future__ import annotations

from pathlib import Path

from defence_agent.evals.model_quality_report import (
    DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
    DEFAULT_MODEL_QUALITY_REGISTRY,
    build_model_quality_report,
    load_registry,
    _score_ranked_sources,
    transcript_paths_for_registry,
)


def test_curated_model_quality_registry_has_no_acl_metric_slice() -> None:
    registry = load_registry(DEFAULT_MODEL_QUALITY_REGISTRY)
    slices = {str(case.get("eval_slice", "")) for case in registry.get("cases", [])}

    assert "acl_refusal" not in slices
    assert "insufficient_evidence_refusal" in slices


def test_transcript_selection_uses_exact_curated_files() -> None:
    registry = load_registry(DEFAULT_MODEL_QUALITY_REGISTRY)
    paths = transcript_paths_for_registry(registry, DEFAULT_FULL_PILOT_TRANSCRIPT_DIR)
    names = {path.name for path in paths}

    assert "acl_unclassified_sensor_fusion_release_rule.json" not in names
    assert "insufficient_evidence_planning_topic.json" in names
    assert all(Path(path).exists() for path in paths)


def test_model_quality_report_scores_saved_curated_transcripts_without_live_calls() -> None:
    report = build_model_quality_report(
        registry_path=DEFAULT_MODEL_QUALITY_REGISTRY,
        transcript_dir=DEFAULT_FULL_PILOT_TRANSCRIPT_DIR,
        citation_label_path=None,
        include_citation_support=False,
        include_rerank_uplift=False,
    )

    summary = report["summary"]
    assert summary["sample_size"] >= 10
    assert summary["refusal_case_count"] >= 2
    assert summary["citation_recall"] is not None
    assert report["metric_contract"]["acl_enforcement"] == "excluded_from_model_quality_metrics"
    assert report["rerank_uplift"]["status"] == "skipped"
    assert report["bilingual_citations"]["status"] == "complete"
    assert report["bilingual_citations"]["comparison"]["bilingual"]["case_count"] >= 1
    assert "Bilingual Citation Analysis" in report["markdown"]


def test_retrieval_ndcg_is_bounded_when_same_expected_doc_repeats() -> None:
    case = {"expected_doc_ids": ["DOC1", "DOC2"]}
    sources = [
        {"doc_id": "DOC1"},
        {"doc_id": "DOC1"},
        {"doc_id": "DOC1"},
        {"doc_id": "DOC2"},
    ]

    row = _score_ranked_sources(sources, case)

    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["ndcg"] <= 1.0
