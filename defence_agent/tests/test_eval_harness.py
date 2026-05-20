from __future__ import annotations

from pathlib import Path

import yaml

from defence_agent.evals.harness import (
    TARGET_SAMPLE_SIZES,
    citation_review_queue,
    load_registry,
    load_citation_precision_labels,
    markdown_report,
    score_outcome,
    summarize_transcript_runs,
)


def test_eval_harness_scores_retrieval_generation_and_citations() -> None:
    case = {
        "id": "public_fact",
        "persona_id": "clearance_unclassified",
        "query": "What does DOC1 say?",
        "expected_doc_ids": ["DOC1"],
        "expected_facets": [{"id": "main", "expected_doc_ids": ["DOC1"]}],
        "expected_answer_facts": ["alpha"],
        "min_citations": 1,
    }
    outcome = {
        "case_id": "public_fact",
        "passed": True,
        "doc_ids": ["DOC1", "DOC2"],
        "answer": "The answer includes alpha.",
        "citation_count": 1,
        "citation_validation": {
            "passed": True,
            "citation_count": 1,
            "coverage": {"claim_count": 1, "covered_claim_count": 1, "uncited_claim_count": 0},
        },
        "final_answer_audit": {
            "retrieval": {
                "sources_sent_to_answer": [
                    {"doc_id": "DOC1", "chunk_id": "DOC1_page_001", "page": 1},
                    {"doc_id": "DOC2", "chunk_id": "DOC2_page_001", "page": 1},
                ],
                "authorized_sources": [],
            },
            "generation": {
                "document_count": 2,
                "cohere_document_ids": ["DOC1_page_001", "DOC2_page_001"],
                "citation_quality": {"citation_recall_proxy": 1.0},
            },
            "citations": [
                {
                    "start": 0,
                    "end": 26,
                    "text": "The answer includes alpha.",
                    "sources": [{"source_id": "DOC1_page_001", "doc_id": "DOC1", "page": 1}],
                }
            ],
        },
    }

    row = score_outcome(outcome, case)

    assert row["retrieval_recall"] == 1.0
    assert row["retrieval_precision"] == 0.5
    assert row["facet_recall"] == 1.0
    assert row["generation_fact_recall"] == 1.0
    assert row["citation_recall"] == 1.0
    assert row["citation_doc_recall"] == 1.0
    assert row["citation_doc_precision"] == 1.0
    assert row["citation_source_precision_proxy"] == 1.0
    assert row["citation_support_precision_lexical"] is None
    assert row["end_to_end_success"] == 1.0


def test_eval_harness_scores_lexical_citation_support_precision() -> None:
    case = {
        "id": "public_fact",
        "persona_id": "clearance_unclassified",
        "query": "What does DOC1 say?",
        "expected_doc_ids": ["DOC1"],
    }
    outcome = {
        "case_id": "public_fact",
        "passed": True,
        "doc_ids": ["DOC1"],
        "answer": "The alpha threshold is 0.7342.",
        "citation_count": 2,
        "citation_validation": {
            "passed": True,
            "citation_count": 2,
            "coverage": {"claim_count": 2, "covered_claim_count": 2, "uncited_claim_count": 0},
        },
        "final_answer_audit": {
            "retrieval": {
                "sources_sent_to_answer": [
                    {"doc_id": "DOC1", "chunk_id": "DOC1_page_001", "page": 1},
                    {"doc_id": "DOC1", "chunk_id": "DOC1_page_002", "page": 2},
                ]
            },
            "generation": {"cohere_document_ids": ["DOC1_page_001", "DOC1_page_002"]},
            "citations": [
                {
                    "start": 0,
                    "end": 29,
                    "text": "The alpha threshold is 0.7342.",
                    "sources": [{"source_id": "DOC1_page_001", "doc_id": "DOC1", "page": 1}],
                },
                {
                    "start": 0,
                    "end": 20,
                    "text": "Unsupported beta finding.",
                    "sources": [{"source_id": "DOC1_page_002", "doc_id": "DOC1", "page": 2}],
                },
            ],
        },
    }

    row = score_outcome(
        outcome,
        case,
        source_text_lookup={
            "DOC1_page_001": "The alpha threshold is 0.7342 and it applies to release.",
            "DOC1_page_002": "This page covers unrelated gamma planning notes.",
        },
    )

    assert row["citation_support_precision_lexical"] == 0.5


def test_eval_harness_scores_semantic_citation_precision_labels() -> None:
    case = {
        "id": "public_fact",
        "persona_id": "clearance_unclassified",
        "query": "What does DOC1 say?",
        "expected_doc_ids": ["DOC1"],
    }
    outcome = {
        "_transcript_run": "run_a",
        "_transcript_file": "public_fact.json",
        "case_id": "public_fact",
        "passed": True,
        "doc_ids": ["DOC1"],
        "answer": "Alpha is supported; beta is not.",
        "final_answer_audit": {
            "retrieval": {"sources_sent_to_answer": [{"doc_id": "DOC1", "chunk_id": "DOC1_page_001", "page": 1}]},
            "generation": {"cohere_document_ids": ["DOC1_page_001"], "document_count": 1},
            "citations": [
                {
                    "start": 0,
                    "end": 18,
                    "text": "Alpha is supported",
                    "sources": [{"source_id": "DOC1_page_001", "doc_id": "DOC1", "page": 1}],
                },
                {
                    "start": 20,
                    "end": 31,
                    "text": "beta is not",
                    "sources": [{"source_id": "DOC1_page_001", "doc_id": "DOC1", "page": 1}],
                },
            ],
        },
    }
    labels = {
        "run_a/public_fact.json::citation:0:source:0:span:0-18:source:DOC1_page_001": True,
        "run_a/public_fact.json::citation:1:source:0:span:20-31:source:DOC1_page_001": False,
    }

    row = score_outcome(outcome, case, citation_labels=labels)

    assert row["citation_precision"] == 0.5
    assert row["citation_precision_label_count"] == 2
    assert row["citation_precision_supported_count"] == 1


def test_loads_citation_precision_labels_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text(
        """
labels:
  - key: supported-key
    supported: true
  - key: unsupported-key
    judgment: unsupported
  - key: skipped-key
    judgment: unsure
""",
        encoding="utf-8",
    )

    labels = load_citation_precision_labels(path)

    assert labels == {"supported-key": True, "unsupported-key": False}


def test_load_registry_resolves_include_registries(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """
suite: base
cases:
  - id: base_case
    query: Base?
""",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        """
suite: child
include_registries:
  - base.yaml
cases:
  - id: child_case
    query: Child?
""",
        encoding="utf-8",
    )

    registry = load_registry(child)

    assert [case["id"] for case in registry["cases"]] == ["base_case", "child_case"]
    assert "include_registries" not in registry


def test_eval_harness_scores_refusal_and_forbidden_fact_absence() -> None:
    case = {
        "id": "refusal",
        "persona_id": "clearance_unclassified",
        "query": "What is the secret?",
        "expected_refusal": True,
        "expected_documents_sent_to_model": 0,
        "expected_excluded_doc_ids": ["DOC_SECRET"],
        "forbidden_answer_facts": ["secret-value"],
    }
    outcome = {
        "case_id": "refusal",
        "passed": True,
        "doc_ids": [],
        "excluded_doc_ids": ["DOC_SECRET"],
        "documents_sent_to_model": 0,
        "answer": "I do not have enough authorized evidence to answer.",
        "final_answer_audit": {
            "retrieval": {
                "answerability": [{"answerable": False, "reason": "denied_source_matches_query"}],
                "excluded_sources": [{"doc_id": "DOC_SECRET", "chunk_id": "DOC_SECRET_page_001"}],
            },
            "generation": {"document_count": 0},
            "citations": [],
        },
    }

    row = score_outcome(outcome, case)

    assert row["slice"] == "acl_refusal"
    assert row["answerability_accuracy"] == 1.0
    assert row["forbidden_fact_absence"] == 1.0
    assert row["citation_recall"] is None
    assert row["end_to_end_success"] == 1.0


def test_eval_harness_scores_model_grounded_insufficiency_refusal() -> None:
    case = {
        "id": "insufficient",
        "persona_id": "clearance_unclassified",
        "query": "What is the approved 2031 submarine basing schedule?",
        "expected_refusal": True,
        "expected_documents_sent_to_model": 8,
        "forbidden_answer_facts": ["2031 basing schedule"],
    }
    outcome = {
        "case_id": "insufficient",
        "passed": True,
        "doc_ids": ["CA-DEF-POL-2024-EN"],
        "documents_sent_to_model": 8,
        "answer": "Evidence is insufficient.",
        "final_answer_audit": {
            "retrieval": {
                "answerability": [{"answerable": False, "reason": "insufficient_authorized_evidence"}],
                "sources_sent_to_answer": [
                    {"doc_id": "CA-DEF-POL-2024-EN", "chunk_id": f"CA-DEF-POL-2024-EN_page_{index:03d}"}
                    for index in range(1, 9)
                ],
            },
            "generation": {"document_count": 8},
            "citations": [],
        },
    }

    row = score_outcome(outcome, case)

    assert row["slice"] == "insufficient_evidence"
    assert row["answerability_accuracy"] == 1.0
    assert row["forbidden_fact_absence"] == 1.0
    assert row["end_to_end_success"] == 1.0


def test_eval_harness_summarizes_real_default_transcripts() -> None:
    report = summarize_transcript_runs()

    assert report["overall"]["n"] >= 6
    assert report["overall"]["end_to_end_success"] is not None
    assert any(item["slice"] == "acl_refusal" for item in report["sample_distribution"])
    assert any(item["slice"] == "scanned_manual" and item["missing"] > 0 for item in report["target_sample_coverage"])
    rendered = markdown_report(report)
    assert "Coverage Against 60-Case Target Bank" in rendered
    assert "Citation recall is the automated claim-span coverage proxy" in rendered
    assert "citation_recall" in rendered


def test_pilot_citation_labels_score_real_default_transcripts() -> None:
    report = summarize_transcript_runs(
        include_citation_support=True,
        citation_label_path=Path("defence_agent/data/evals/citation_precision_pilot_labels.yaml"),
    )

    assert report["overall"]["citation_precision_label_count"] == 20
    assert report["overall"]["citation_precision"] == 0.85
    assert any(item["group"] == "bilingual" and item["citation_precision"] == 0.5 for item in report["by_slice"])


def test_citation_review_queue_exports_stable_label_rows(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
cases:
  - id: public_fact
    persona_id: clearance_unclassified
    query: What does DOC1 say?
    expected_doc_ids:
      - DOC1
""",
        encoding="utf-8",
    )
    transcript_dir = tmp_path / "run_a"
    transcript_dir.mkdir()
    (transcript_dir / "public_fact.json").write_text(
        """
{
  "case_id": "public_fact",
  "passed": true,
  "answer": "Alpha is supported.",
  "final_answer_audit": {
    "citations": [
      {
        "start": 0,
        "end": 18,
        "text": "Alpha is supported",
        "sources": [
          {"source_id": "DOC1_page_001", "doc_id": "DOC1", "page": 1, "title": "Doc One"}
        ]
      }
    ]
  }
}
""",
        encoding="utf-8",
    )

    rows = citation_review_queue(registry_path=registry_path, transcript_paths=[transcript_dir])

    assert rows[0]["key"] == "run_a/public_fact.json::citation:0:source:0:span:0-18:source:DOC1_page_001"
    assert rows[0]["answer_span"] == "Alpha is supported"
    assert rows[0]["supported"] is None


def test_pilot_eval_bank_has_balanced_target_distribution() -> None:
    path = Path("defence_agent/data/evals/pilot_eval_bank.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = data["cases"]

    assert len(cases) == sum(TARGET_SAMPLE_SIZES.values())
    for slice_name, target in TARGET_SAMPLE_SIZES.items():
        assert sum(1 for case in cases if case.get("eval_slice") == slice_name) == target
