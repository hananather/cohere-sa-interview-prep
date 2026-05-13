from __future__ import annotations

import importlib.util
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("google.adk") is None,
    reason="google-adk optional dependency is not installed",
)


@pytest.fixture(scope="session")
def live_index_ready() -> None:
    from defence_agent.retrieval.chroma_index import build_index

    result = build_index(force=False)
    assert result["embedding_backend"] == "embed-v4.0"


def _demo_registry_script() -> Any:
    path = ROOT / "defence_agent" / "scripts" / "run_demo_query_registry.py"
    spec = importlib.util.spec_from_file_location("run_demo_query_registry", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minimal_adk_agent_imports() -> None:
    from defence_agent.agent import root_agent

    assert root_agent.name == "defence_agent"


def test_settings_require_real_cohere_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from defence_agent.config import Settings

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, COHERE_API_KEY=" ")


def test_settings_expose_no_mock_cohere_switch() -> None:
    from defence_agent.config import Settings

    fields = set(Settings.model_fields)

    assert "use_mock_cohere" not in fields
    assert "vector_size" not in fields


@pytest.mark.live
def test_live_index_reuses_existing_pages(live_index_ready: None) -> None:
    from defence_agent.config import get_settings
    from defence_agent.retrieval.chroma_index import build_index

    result = build_index(force=False)

    assert result["embedding_backend"] == "embed-v4.0"
    assert result["embedding_dimension"] == get_settings().cohere_embed_output_dimension
    assert result["parsed_pages"] == 207
    assert result["embedded_pages"] == 0
    assert result["skipped"] is True


@pytest.mark.slow_offline
def test_stale_page_hash_requires_reembedding() -> None:
    from defence_agent.retrieval.chroma_index import _metadata_for_index, _page_needs_embedding, parse_only

    page = parse_only()[0]
    current_metadata = _metadata_for_index(page)
    stale_metadata = dict(current_metadata)
    stale_metadata["page_image_sha256"] = "stale"

    assert _page_needs_embedding(page, current_metadata) is False
    assert _page_needs_embedding(page, stale_metadata) is True
    assert _page_needs_embedding(page, None) is True


def test_agent_prompt_defaults_english_questions_to_language_any() -> None:
    from defence_agent.prompts import AGENT_INSTRUCTION

    assert "primarily a search and retrieval agent" in AGENT_INSTRUCTION
    assert "answer it completely" in AGENT_INSTRUCTION
    assert 'Use language="any" for doctrine questions' in AGENT_INSTRUCTION
    assert 'Use language="en" for English questions unless the user asks for French.' not in AGENT_INSTRUCTION
    assert "call search_documents separately for each" in AGENT_INSTRUCTION
    assert "Rewrite broad user questions into focused retrieval queries" in AGENT_INSTRUCTION
    assert "Use top_k=16 for broad planning" in AGENT_INSTRUCTION


def test_search_documents_tool_description_is_session_controlled_and_bounded() -> None:
    from defence_agent.tools import search_documents

    doc = search_documents.__doc__ or ""

    assert "ADK session state" in doc
    assert "not from model-controlled" in doc
    assert "Excluded source text is never returned" in doc
    assert "1..24" in doc
    assert '"approved", "draft", "superseded", or "any"' in doc
    assert '"any", "en", or "fr"' in doc


@pytest.mark.slow_offline
def test_manifest_loader_parses_public_and_synthetic_pdfs() -> None:
    from defence_agent.retrieval.chroma_index import parse_only

    pages = parse_only()
    doc_ids = {page.doc_id for page in pages}

    assert len(pages) == 207
    assert {"CA-AI-STRAT-2024-EN", "NATO-STRAT-CONCEPT-2022-FR"}.issubset(doc_ids)
    assert {"SYN-FUSION-S-RELEASE-001", "SYN-FUSION-TS-ANNEX-002"}.issubset(doc_ids)


@pytest.mark.slow_offline
def test_manifest_loader_preserves_docx_origin_normalization_metadata() -> None:
    from defence_agent.retrieval.chroma_index import parse_only

    pages = parse_only()
    doc_pages = [page for page in pages if page.doc_id == "UK-MOD-ASOEM-2023-EN"]

    assert len(doc_pages) == 14
    metadata = doc_pages[0].metadata
    assert metadata["source_type"] == "official_public_docx_origin_pdf"
    assert metadata["source_format"] == "docx"
    assert metadata["normalized_format"] == "pdf"
    assert metadata["normalization_method"] == "official_pdf_pair"
    assert metadata["source_docx_url"].endswith("ASOEM_Issue_2.docx")
    assert metadata["source_pdf_url"].endswith("ASOEM_Issue_2.pdf")
    assert "PDF pair" in metadata["provenance_note"]


@pytest.mark.live
def test_unclassified_persona_cannot_see_secret_doc(live_index_ready: None) -> None:
    from defence_agent.tools import search_documents

    class FakeToolContext:
        state = {"persona_id": "clearance_unclassified"}

    result = search_documents(
        "For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?",
        tool_context=FakeToolContext(),
    )

    assert result["persona_id"] == "clearance_unclassified"
    assert result["allowed_access"] == ["unclassified"]
    assert result["policy_decision"] == "refuse"
    assert result["answerability"]["reason"] == "denied_source_matches_query"
    assert any(source["doc_id"] == "SYN-FUSION-S-RELEASE-001" for source in result["excluded_sources"])
    assert not any(source["doc_id"] == "SYN-FUSION-S-RELEASE-001" for source in result["authorized_sources"])
    assert FakeToolContext.state["last_search_sources"] == []
    assert FakeToolContext.state["last_search_audit"]["sources_sent_to_answer"] == []
    assert FakeToolContext.state["last_search_audit"]["excluded_sources"]
    assert all("text" not in source for source in FakeToolContext.state["last_search_audit"]["excluded_sources"])
    assert result["authorized_sources"]


@pytest.mark.live
def test_secret_persona_can_see_secret_doc_from_adk_session_state(live_index_ready: None) -> None:
    from defence_agent.tools import search_documents

    class FakeToolContext:
        state = {"persona_id": "clearance_secret"}

    result = search_documents(
        "For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?",
        tool_context=FakeToolContext(),
    )
    source_ids = {source["doc_id"] for source in result["authorized_sources"]}

    assert result["persona_id"] == "clearance_secret"
    assert result["allowed_access"] == ["unclassified", "secret"]
    assert result["policy_decision"] == "allow"
    assert "SYN-FUSION-S-RELEASE-001" in source_ids
    assert FakeToolContext.state["last_search_persona_id"] == "clearance_secret"
    assert FakeToolContext.state["last_search_citations"]
    assert FakeToolContext.state["last_search_sources"]
    assert FakeToolContext.state["last_search_audit"]["persona_id"] == "clearance_secret"
    assert FakeToolContext.state["last_search_audit"]["sources_sent_to_answer"]
    assert FakeToolContext.state["search_history_audits"]
    assert FakeToolContext.state["search_history_sources"]


@pytest.mark.live
def test_secret_persona_cannot_see_top_secret_doc_but_top_secret_persona_can(live_index_ready: None) -> None:
    from defence_agent.tools import search_documents

    class SecretToolContext:
        state = {"persona_id": "clearance_secret"}

    class TopSecretToolContext:
        state = {"persona_id": "clearance_top_secret"}

    query = "What routing codeword opens the restricted relay path for the Fusion Model?"
    secret_result = search_documents(query, tool_context=SecretToolContext())
    top_secret_result = search_documents(query, tool_context=TopSecretToolContext())

    secret_source_ids = {source["doc_id"] for source in secret_result["authorized_sources"]}
    top_secret_source_ids = {source["doc_id"] for source in top_secret_result["authorized_sources"]}

    assert secret_result["policy_decision"] == "refuse"
    assert "SYN-FUSION-TS-ANNEX-002" not in secret_source_ids
    assert any(source["doc_id"] == "SYN-FUSION-TS-ANNEX-002" for source in secret_result["excluded_sources"])
    assert top_secret_result["policy_decision"] == "allow"
    assert "SYN-FUSION-TS-ANNEX-002" in top_secret_source_ids


@pytest.mark.live
def test_search_documents_uses_public_mixed_corpus(live_index_ready: None) -> None:
    from defence_agent.tools import search_documents

    result = search_documents(
        "What are the DND CAF AI Strategy lines of effort?",
        status_filter="any",
    )
    source_ids = {source["doc_id"] for source in result["authorized_sources"]}

    assert source_ids
    assert "CA-AI-STRAT-2024-EN" in source_ids
    source = result["authorized_sources"][0]
    assert source["source_type"] == "official_public_pdf"
    assert source["source_pdf_path"].endswith(".pdf")
    assert source["page_image_sha256"]


@pytest.mark.live
def test_search_documents_retrieves_docx_origin_normalized_source(live_index_ready: None) -> None:
    from defence_agent.tools import search_documents

    class FakeToolContext:
        state: dict[str, Any] = {"persona_id": "clearance_unclassified"}

    result = search_documents(
        "Aviation Safe Operating Environment Manual ASOEM Aviation Duty Holders Accountable Managers Heads of Establishment",
        top_k=8,
        status_filter="approved",
        language="en",
        tool_context=FakeToolContext(),
    )
    docx_origin_sources = [
        source for source in result["authorized_sources"] if source["doc_id"] == "UK-MOD-ASOEM-2023-EN"
    ]

    assert docx_origin_sources
    source = docx_origin_sources[0]
    assert source["source_type"] == "official_public_docx_origin_pdf"
    assert source["source_format"] == "docx"
    assert source["normalized_format"] == "pdf"
    assert source["normalization_method"] == "official_pdf_pair"
    assert source["source_docx_url"].endswith("ASOEM_Issue_2.docx")
    assert source["source_pdf_url"].endswith("ASOEM_Issue_2.pdf")
    audit_sources = [
        source
        for source in FakeToolContext.state["last_search_audit"]["authorized_sources"]
        if source["doc_id"] == "UK-MOD-ASOEM-2023-EN"
    ]
    assert audit_sources[0]["source_format"] == "docx"
    assert audit_sources[0]["normalized_format"] == "pdf"
    assert audit_sources[0]["normalization_method"] == "official_pdf_pair"


@pytest.mark.live
def test_search_documents_can_return_multiple_pages_per_document(live_index_ready: None) -> None:
    from defence_agent.tools import search_documents

    result = search_documents(
        "What are the DND CAF AI Strategy lines of effort and related governance priorities?",
        top_k=8,
        status_filter="approved",
        language="any",
    )
    pages_by_doc: dict[str, set[int]] = {}
    for source in result["authorized_sources"]:
        pages_by_doc.setdefault(str(source["doc_id"]), set()).add(int(source["page"]))

    assert any(len(pages) > 1 for pages in pages_by_doc.values())


def test_source_selection_does_not_cap_pages_per_document() -> None:
    from defence_agent.retrieval.chroma_index import _select_sources

    ranked = [
        {"chunk_id": f"DOC1_page_{page:03d}", "doc_id": "DOC1", "rerank_score": 1.0 - page / 100}
        for page in range(1, 7)
    ] + [{"chunk_id": "DOC2_page_001", "doc_id": "DOC2", "rerank_score": 0.01}]

    selected = _select_sources(ranked, top_k=5)

    assert [source["chunk_id"] for source in selected] == [
        "DOC1_page_001",
        "DOC1_page_002",
        "DOC1_page_003",
        "DOC1_page_004",
        "DOC1_page_005",
    ]


def test_source_selection_does_not_inject_low_ranked_docs_for_coverage() -> None:
    from defence_agent.retrieval.chroma_index import _select_sources

    ranked = [
        {"chunk_id": f"DOC1_page_{page:03d}", "doc_id": "DOC1", "rerank_score": 1.0 - page / 100}
        for page in range(1, 7)
    ] + [{"chunk_id": "DOC2_page_001", "doc_id": "DOC2", "rerank_score": 0.01}]

    selected = _select_sources(ranked, top_k=5)

    assert [source["chunk_id"] for source in selected] == [
        "DOC1_page_001",
        "DOC1_page_002",
        "DOC1_page_003",
        "DOC1_page_004",
        "DOC1_page_005",
    ]


def test_source_selection_preserves_close_multilingual_evidence() -> None:
    from defence_agent.retrieval.chroma_index import _select_sources

    ranked = [
        {
            "chunk_id": f"NATO-FR_page_{page:03d}",
            "doc_id": "NATO-FR",
            "language": "fr",
            "rerank_score": 1.0 - page / 100,
        }
        for page in range(1, 6)
    ] + [
        {
            "chunk_id": "NATO-EN_page_002",
            "doc_id": "NATO-EN",
            "language": "en",
            "rerank_score": 0.94,
        }
    ]

    selected = _select_sources(ranked, top_k=5, language="any")

    assert [source["chunk_id"] for source in selected] == [
        "NATO-FR_page_001",
        "NATO-FR_page_002",
        "NATO-FR_page_003",
        "NATO-FR_page_004",
        "NATO-EN_page_002",
    ]


def test_source_selection_does_not_force_weak_multilingual_evidence() -> None:
    from defence_agent.retrieval.chroma_index import _select_sources

    ranked = [
        {
            "chunk_id": f"NATO-FR_page_{page:03d}",
            "doc_id": "NATO-FR",
            "language": "fr",
            "rerank_score": 1.0 - page / 100,
        }
        for page in range(1, 6)
    ] + [
        {
            "chunk_id": "NATO-EN_page_002",
            "doc_id": "NATO-EN",
            "language": "en",
            "rerank_score": 0.41,
        }
    ]

    selected = _select_sources(ranked, top_k=5, language="any")

    assert [source["chunk_id"] for source in selected] == [
        "NATO-FR_page_001",
        "NATO-FR_page_002",
        "NATO-FR_page_003",
        "NATO-FR_page_004",
        "NATO-FR_page_005",
    ]


def test_dated_schedule_query_without_specific_support_is_hard_refusal() -> None:
    from defence_agent.retrieval.chroma_index import _answerability

    answerability = _answerability(
        "approved Arctic submarine basing schedule for 2031",
        [
            {
                "doc_id": "CA-DEF-POL-2024-EN",
                "title": "Our North, Strong and Free",
                "section": "Foreword",
                "text": "General defence policy context.",
                "vector_score": 0.31,
                "rerank_score": 0.04,
            }
        ],
        [],
    )

    assert answerability["answerable"] is False
    assert answerability["reason"] == "insufficient_authorized_evidence"
    assert answerability["best_authorized_overlap"] == 0
    assert answerability["unsupported_specificity"]["missing_year_terms"] == ["2031"]
    assert sorted(answerability["unsupported_specificity"]["missing_scheduled_fact_terms"]) == [
        "basing",
        "schedule",
    ]
    assert answerability["evidence_quality"]["hard_refusal"] is True


def test_low_lexical_overlap_without_specific_schedule_is_audit_signal() -> None:
    from defence_agent.retrieval.chroma_index import _answerability

    answerability = _answerability(
        "approved Arctic submarine posture for planners",
        [
            {
                "doc_id": "CA-DEF-POL-2024-EN",
                "title": "Our North, Strong and Free",
                "section": "Foreword",
                "text": "General defence policy context.",
                "vector_score": 0.31,
                "rerank_score": 0.04,
            }
        ],
        [],
    )

    assert answerability["answerable"] is True
    assert answerability["reason"] == "authorized_sources_available"
    assert answerability["best_authorized_overlap"] == 0
    assert answerability["evidence_quality"]["threshold_policy"] == "audit_only_until_calibrated"
    assert answerability["evidence_quality"]["hard_refusal"] is False


@pytest.mark.live
def test_search_documents_supports_language_filter(live_index_ready: None) -> None:
    from defence_agent.tools import search_documents

    result = search_documents(
        "Quelles sont les taches fondamentales de l OTAN dans le concept strategique?",
        language="fr",
    )
    assert result["filters_applied"]["language"] == "fr"
    assert result["authorized_sources"]
    assert all(source["language"] == "fr" for source in result["authorized_sources"])
    assert any(source["doc_id"] == "NATO-STRAT-CONCEPT-2022-FR" for source in result["authorized_sources"])


def test_native_citation_validation_rejects_uncited_claims() -> None:
    from defence_agent.grounding import _validate_native_citations

    evidence = {
        "doc1_page_001": {"doc_id": "DOC1"},
        "doc2_page_001": {"doc_id": "DOC2"},
    }
    answer = "Canada links AI to modernization. The governance priority is responsible adoption."
    citations = [
        {
            "start": 0,
            "end": 31,
            "text": "Canada links AI to modernization.",
            "sources": [{"source_id": "doc1_page_001", "doc_id": "DOC1"}],
        }
    ]

    validation = _validate_native_citations(
        citations,
        evidence,
        answer=answer,
        query="Compare the two documents for a planning brief.",
    )

    assert validation["passed"] is False
    assert "uncited_claims:1" in validation["errors"]
    assert validation["coverage"]["uncited_claim_count"] == 1
    assert "citations_reference_too_few_documents" in validation["warnings"]


def test_native_citation_validation_accepts_claim_coverage() -> None:
    from defence_agent.grounding import _validate_native_citations

    evidence = {
        "doc1_page_001": {"doc_id": "DOC1"},
        "doc2_page_001": {"doc_id": "DOC2"},
    }
    answer = "Canada links AI to modernization. Governance requires responsible adoption."
    citations = [
        {
            "start": 0,
            "end": 31,
            "text": "Canada links AI to modernization.",
            "sources": [{"source_id": "doc1_page_001", "doc_id": "DOC1"}],
        },
        {
            "start": 32,
            "end": len(answer),
            "text": "Governance requires responsible adoption.",
            "sources": [{"source_id": "doc2_page_001", "doc_id": "DOC2"}],
        },
    ]

    validation = _validate_native_citations(
        citations,
        evidence,
        answer=answer,
        query="Compare the two documents for a planning brief.",
    )

    assert validation["passed"] is True
    assert validation["warnings"] == []
    assert validation["coverage"]["covered_claim_count"] == 2
    assert validation["cited_doc_ids"] == ["DOC1", "DOC2"]


def test_native_citation_validation_warns_on_high_coverage_gap() -> None:
    from defence_agent.grounding import _validate_native_citations

    evidence = {"doc1_page_001": {"doc_id": "DOC1"}}
    claims = [f"Claim {index} is supported by the document." for index in range(1, 11)]
    answer = " ".join(claims)
    citations = []
    offset = 0
    for claim in claims[:9]:
        citations.append(
            {
                "start": offset,
                "end": offset + len(claim),
                "text": claim,
                "sources": [{"source_id": "doc1_page_001", "doc_id": "DOC1"}],
            }
        )
        offset += len(claim) + 1

    validation = _validate_native_citations(citations, evidence, answer=answer, query="Summarize the document.")

    assert validation["passed"] is True
    assert "uncited_claims:1" in validation["warnings"]
    assert validation["coverage"]["uncited_claim_count"] == 1


def test_citation_coverage_handles_decimal_numbers() -> None:
    from defence_agent.grounding import _citation_coverage

    answer = "The candidate still requires a confidence score at or above 0.7342."
    citations = [{"start": 0, "end": len(answer), "sources": [{"source_id": "doc"}]}]

    coverage = _citation_coverage(answer, citations)

    assert coverage["claim_count"] == 1
    assert coverage["uncited_claim_count"] == 0


def test_rerank_formats_candidates_as_structured_yaml() -> None:
    from defence_agent.retrieval import chroma_index

    rerank_document = chroma_index._rerank_document(
        {
            "title": "Fusion Model Release Control",
            "doc_id": "SYN-FUSION-S-RELEASE-001",
            "page": 2,
            "section": "Release threshold",
            "language": "en",
            "status": "approved",
            "version": "1.0",
            "effective_date": "2026-01-01",
            "access_level": "secret",
            "source_format": "pdf",
            "normalized_format": "pdf",
            "text": "Release requires a confidence threshold of 0.7342.",
        }
    )

    assert "title: Fusion Model Release Control" in rerank_document
    assert "doc_id: SYN-FUSION-S-RELEASE-001" in rerank_document
    assert "page: 2" in rerank_document
    assert "source_format: pdf" in rerank_document
    assert "normalized_format: pdf" in rerank_document
    assert "content: Release requires a confidence threshold of 0.7342." in rerank_document


def test_grounding_refuses_without_authorized_sources() -> None:
    from defence_agent.grounding import finalize_answer

    result = finalize_answer(query="What does the source say?", sources=[], fallback_answer="No authorized evidence.")

    assert result.answer == "No authorized evidence."
    assert result.citation_validation["passed"] is False
    assert "no_authorized_sources" in result.citation_validation["errors"]


def test_answer_audit_includes_traceability_metadata() -> None:
    from defence_agent.grounding import GroundedAnswer
    from defence_agent.grounding import COHERE_CITATION_MODE
    from defence_agent.session import _answer_audit

    assert COHERE_CITATION_MODE == "accurate_default"

    retrieval_audit = {
        "query": "What is the threshold?",
        "persona_id": "clearance_secret",
        "allowed_access": ["unclassified", "secret"],
        "filters_applied": {"access_level": ["unclassified", "secret"], "status": "approved", "language": "any"},
        "policy_decision": "allow",
        "answerability": {"answerable": True, "reason": "authorized_sources_available"},
        "authorized_sources": [
            {
                "citation_id": "C1",
                "chunk_id": "SYN-FUSION-S-RELEASE-001_p2",
                "doc_id": "SYN-FUSION-S-RELEASE-001",
                "title": "Fusion Model Release Control",
                "page": 2,
                "language": "en",
                "access_level": "secret",
                "source_type": "synthetic_pdf",
                "source_pdf_path": "/tmp/fusion_model_release_control.pdf",
                "manifest_path": "/tmp/manifest.yaml",
                "page_image_sha256": "abc123",
                "vector_score": 0.8,
                "rerank_score": 0.9,
            }
        ],
        "sources_sent_to_answer": [
            {
                "citation_id": "C1",
                "chunk_id": "SYN-FUSION-S-RELEASE-001_p2",
                "doc_id": "SYN-FUSION-S-RELEASE-001",
                "title": "Fusion Model Release Control",
                "page": 2,
                "language": "en",
                "access_level": "secret",
                "source_type": "synthetic_pdf",
                "source_pdf_path": "/tmp/fusion_model_release_control.pdf",
                "manifest_path": "/tmp/manifest.yaml",
                "page_image_sha256": "abc123",
                "vector_score": 0.8,
                "rerank_score": 0.9,
            }
        ],
        "excluded_sources": [
            {
                "doc_id": "SYN-FUSION-TS-ANNEX-002",
                "title": "Restricted Routing Annex",
                "access_level": "top_secret",
                "reason": "access_denied",
            }
        ],
    }
    grounded = GroundedAnswer(
        answer="The threshold is 0.7342 [C1].",
        raw_answer="The threshold is 0.7342.",
        citations=[
            {
                "type": "cohere_native",
                "start": 17,
                "end": 23,
                "text": "0.7342",
                "sources": [
                    {
                        "source_id": "SYN-FUSION-S-RELEASE-001_p2",
                        "label": "C1",
                        "doc_id": "SYN-FUSION-S-RELEASE-001",
                        "title": "Fusion Model Release Control",
                        "page": 2,
                        "access_level": "secret",
                        "chunk_id": "SYN-FUSION-S-RELEASE-001_p2",
                    }
                ],
            }
        ],
        citation_mode="cohere_native_accurate",
        citation_validation={
            "passed": True,
            "errors": [],
            "citation_count": 1,
            "coverage": {"claim_count": 1, "covered_claim_count": 1, "uncited_claim_count": 0},
        },
        documents_sent=1,
        document_ids=["SYN-FUSION-S-RELEASE-001_p2"],
        model="command-a-03-2025",
    )

    audit = _answer_audit(
        query="What is the threshold?",
        session_id="s_test",
        user_id="clearance_secret",
        persona_id="clearance_secret",
        tool_calls=["search_documents"],
        tool_responses=["search_documents"],
        retrieval_status="retrieval_complete",
        retrieval_audits=[retrieval_audit],
        sources_sent_to_answer=[],
        grounded=grounded,
    )

    assert audit["generation"]["citation_mode"] == "cohere_native_accurate"
    assert audit["generation"]["citation_resolution"]["passed"] is True
    assert audit["generation"]["citation_quality"]["citation_recall_proxy"] == 1.0
    assert audit["generation"]["citation_quality"]["citation_precision"] == "manual_or_llm_judge_required"
    assert audit["generation"]["cohere_document_ids"] == ["SYN-FUSION-S-RELEASE-001_p2"]
    assert audit["retrieval"]["search_queries"] == ["What is the threshold?"]
    assert audit["retrieval"]["excluded_sources"][0]["doc_id"] == "SYN-FUSION-TS-ANNEX-002"
    assert "text" not in audit["retrieval"]["excluded_sources"][0]
    citation_source = audit["citations"][0]["sources"][0]
    assert citation_source["doc_id"] == "SYN-FUSION-S-RELEASE-001"
    assert citation_source["language"] == "en"
    assert citation_source["source_type"] == "synthetic_pdf"
    assert citation_source["source_pdf_path"] == "/tmp/fusion_model_release_control.pdf"
    assert citation_source["page_image_sha256"] == "abc123"
    assert citation_source["rerank_score"] == 0.9


def test_native_citations_resolve_source_metadata_in_answer_audit() -> None:
    from defence_agent.grounding import GroundedAnswer
    from defence_agent.session import _answer_audit

    retrieval_audit = {
        "allowed_access": ["unclassified"],
        "filters_applied": {"access_level": ["unclassified"], "status": "approved", "language": "any"},
        "policy_decision": "allow",
        "answerability": {"answerable": True},
        "authorized_sources": [
            {
                "citation_id": "C1",
                "chunk_id": "CA-AI-STRAT-2024-EN_page_014",
                "doc_id": "CA-AI-STRAT-2024-EN",
                "title": "DND CAF AI Strategy",
                "page": 14,
                "language": "en",
                "access_level": "unclassified",
                "vector_score": 0.7,
                "rerank_score": 0.8,
            }
        ],
        "sources_sent_to_answer": [],
        "excluded_sources": [],
    }
    grounded = GroundedAnswer(
        answer="The strategy has lines of effort [C1].",
        citations=[
            {
                "type": "cohere_native",
                "label": "C1",
                "start": 0,
                "end": 33,
                "text": "The strategy has lines of effort",
                "sources": [
                    {
                        "source_id": "CA-AI-STRAT-2024-EN_page_014",
                        "label": "C1",
                        "doc_id": "CA-AI-STRAT-2024-EN",
                        "title": "DND CAF AI Strategy",
                        "page": 14,
                        "access_level": "unclassified",
                        "chunk_id": "CA-AI-STRAT-2024-EN_page_014",
                    }
                ],
            }
        ],
        citation_mode="cohere_native_accurate_default",
        citation_validation={"passed": True, "citation_count": 1},
    )

    audit = _answer_audit(
        query="Which source?",
        session_id="s_test",
        user_id="clearance_unclassified",
        persona_id="clearance_unclassified",
        tool_calls=["search_documents"],
        tool_responses=["search_documents"],
        retrieval_status="retrieval_complete",
        retrieval_audits=[retrieval_audit],
        sources_sent_to_answer=[],
        grounded=grounded,
    )

    source = audit["citations"][0]["sources"][0]
    assert source["doc_id"] == "CA-AI-STRAT-2024-EN"
    assert source["page"] == 14
    assert source["rerank_score"] == 0.8


def test_registry_citation_source_resolution_accepts_sources_sent_to_answer() -> None:
    registry = _demo_registry_script()
    result = SimpleNamespace(
        answer_audit={
            "retrieval": {
                "sources_sent_to_answer": [
                    {"chunk_id": "DOC1_page_002", "doc_id": "DOC1", "page": 2, "title": "Authorized Page"}
                ],
                "excluded_sources": [],
            },
            "generation": {"cohere_document_ids": ["DOC1_page_002"]},
            "citations": [{"sources": [{"source_id": "DOC1_page_002", "doc_id": "DOC1", "page": 2}]}],
        }
    )

    assert registry._validate_citation_source_resolution(result) == []


def test_registry_citation_source_resolution_rejects_unsent_source_ids() -> None:
    registry = _demo_registry_script()
    result = SimpleNamespace(
        answer_audit={
            "retrieval": {
                "sources_sent_to_answer": [
                    {"chunk_id": "DOC1_page_002", "doc_id": "DOC1", "page": 2, "title": "Authorized Page"}
                ],
                "excluded_sources": [],
            },
            "generation": {"cohere_document_ids": ["DOC1_page_002"]},
            "citations": [{"sources": [{"source_id": "DOC2_page_001", "doc_id": "DOC2", "page": 1}]}],
        }
    )

    assert registry._validate_citation_source_resolution(result) == ["citation_source_not_sent:DOC2_page_001"]


def test_registry_citation_source_resolution_rejects_excluded_sources() -> None:
    registry = _demo_registry_script()
    result = SimpleNamespace(
        answer_audit={
            "retrieval": {
                "sources_sent_to_answer": [
                    {"chunk_id": "DOC1_page_002", "doc_id": "DOC1", "page": 2, "title": "Authorized Page"}
                ],
                "excluded_sources": [{"doc_id": "DOC2", "access_level": "secret", "reason": "access_denied"}],
            },
            "generation": {"cohere_document_ids": ["DOC1_page_002"]},
            "citations": [{"sources": [{"source_id": "DOC2_page_001", "doc_id": "DOC2", "page": 1}]}],
        }
    )

    assert registry._validate_citation_source_resolution(result) == [
        "citation_source_not_sent:DOC2_page_001",
        "citation_source_is_excluded:DOC2_page_001",
    ]


def test_audit_follow_up_returns_prior_source_metadata_without_search() -> None:
    from defence_agent.session import _audit_follow_up_grounded
    from defence_agent.session import _audit_lookup_audit
    from defence_agent.session import _is_audit_follow_up
    from defence_agent.session import _prior_supporting_sources

    prior_audit = {
        "query": "For the new sensor-fusion release workflow, what rule should planning staff follow?",
        "citations": [
            {
                "sources": [
                    {
                        "source_id": "SYN-FUSION-S-RELEASE-001_page_002",
                        "label": "C1",
                        "doc_id": "SYN-FUSION-S-RELEASE-001",
                        "title": "Fusion Model Release Control Procedure",
                        "page": 2,
                        "language": "en",
                        "access_level": "secret",
                        "chunk_id": "SYN-FUSION-S-RELEASE-001_page_002",
                    }
                ]
            }
        ],
        "retrieval": {
            "allowed_access": ["unclassified", "secret"],
            "excluded_sources": [
                {
                    "doc_id": "SYN-FUSION-TS-ANNEX-002",
                    "title": "Fusion Model Restricted Routing Annex",
                    "access_level": "top_secret",
                    "reason": "access_denied",
                }
            ],
        },
    }

    assert _is_audit_follow_up("Which source page supports that, and what access level was required?")
    sources = _prior_supporting_sources(prior_audit)
    grounded = _audit_follow_up_grounded(sources, prior_audit["retrieval"]["excluded_sources"])
    audit = _audit_lookup_audit(
        query="Which source page supports that, and what access level was required?",
        session_id="s_test",
        user_id="clearance_secret",
        persona_id="clearance_secret",
        prior_audit=prior_audit,
        supporting_sources=sources,
        excluded_sources=prior_audit["retrieval"]["excluded_sources"],
        grounded=grounded,
    )

    assert grounded.citation_mode == "answer_audit_lookup"
    assert "SYN-FUSION-S-RELEASE-001 page 2" in grounded.answer
    assert "access level secret" in grounded.answer
    assert audit["retrieval"]["search_count"] == 0
    assert audit["audit_lookup"]["source"] == "last_answer_audit"
    assert all("text" not in source for source in audit["audit_lookup"]["excluded_sources"])


def test_current_turn_audits_use_history_boundary() -> None:
    from defence_agent.session import _turn_audits

    state = {"search_history_audits": [{"query": "old"}, {"query": "new"}], "last_search_audit": {"query": "new"}}

    assert _turn_audits(state, 1) == [{"query": "new"}]


def test_cli_show_audit_prints_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    script_path = ROOT / "defence_agent" / "scripts" / "run_agent_session.py"
    spec = importlib.util.spec_from_file_location("run_agent_session_for_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = SimpleNamespace(
        session_id="s_test",
        user_id="clearance_secret",
        persona_id="clearance_secret",
        events_seen=3,
        tool_calls=["search_documents"],
        citation_mode="cohere_native_accurate",
        citations=[{"text": "0.7342"}],
        citation_validation={"passed": True},
        grounded_model="command-a-03-2025",
        documents_sent_to_model=1,
        retrieval_status="retrieval_complete",
        answer="The threshold is 0.7342 [C1].",
        answer_audit={"citations": [{"text": "0.7342"}], "retrieval": {"authorized_sources": [{"doc_id": "D1"}]}},
    )

    module._print_result("Turn 1", result, show_audit=True)
    output = capsys.readouterr().out
    payload = output.split("answer_audit:\n", 1)[1]

    assert json.loads(payload)["citations"][0]["text"] == "0.7342"


def test_demo_query_registry_contains_natural_demo_cases() -> None:
    registry_path = ROOT / "defence_agent" / "data" / "evals" / "demo_query_registry.yaml"
    registry = json.loads(json.dumps(__import__("yaml").safe_load(registry_path.read_text())))
    cases = {case["id"]: case for case in registry["cases"]}

    assert "natural_multilingual_nato_core_tasks" in cases
    assert "Use French sources" not in cases["natural_multilingual_nato_core_tasks"]["query"]
    assert cases["acl_unclassified_sensor_fusion_release_rule"]["expected_refusal"] is True
    assert cases["trace_follow_up_source_access"]["expected_follow_up_mode"] == "answer_audit_lookup"
    assert sum(len(case.get("query_variants", [])) for case in registry["cases"]) >= 30
    assert cases["flagship_planning_brief_modernization"]["expected_facets"]
    assert cases["multi_query_nato_and_ai_priorities"]["expected_facets"]


def test_demo_registry_facet_validation_checks_expected_sources() -> None:
    script_path = ROOT / "defence_agent" / "scripts" / "run_demo_query_registry.py"
    spec = importlib.util.spec_from_file_location("run_demo_registry_for_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = SimpleNamespace(
        answer_audit={
            "retrieval": {
                "search_queries": ["find NATO core tasks", "find AI Strategy priorities"],
                "authorized_sources": [
                    {"doc_id": "NATO-STRAT-CONCEPT-2022-EN", "page": 4, "source_type": "official_public_pdf"},
                    {"doc_id": "CA-AI-STRAT-2024-EN", "page": 14, "source_type": "official_public_pdf"},
                ],
                "sources_sent_to_answer": [],
                "excluded_sources": [],
            },
            "citations": [
                {"sources": [{"doc_id": "NATO-STRAT-CONCEPT-2022-EN", "page": 4}]},
            ],
        }
    )
    case = {
        "expected_facets": [
            {
                "id": "nato_core_tasks",
                "expected_doc_ids": ["NATO-STRAT-CONCEPT-2022-EN"],
                "min_sources": 1,
                "expected_pages": [{"doc_id": "NATO-STRAT-CONCEPT-2022-EN", "pages": [4]}],
            }
        ]
    }

    assert module._validate_facets(case, result, result) == []


def test_adk_sqlite_sessions_are_isolated_by_user(tmp_path: Path) -> None:
    from google.adk.sessions import DatabaseSessionService

    from defence_agent.session import APP_NAME, ensure_session, sqlite_session_url

    async def scenario() -> None:
        service = DatabaseSessionService(db_url=sqlite_session_url(tmp_path / "sessions.sqlite"))
        shared_session_id = "same_visible_thread_id"
        unclassified_session = await ensure_session(
            service,
            user_id="clearance_unclassified",
            persona_id="clearance_unclassified",
            session_id=shared_session_id,
        )
        secret_session = await ensure_session(
            service,
            user_id="clearance_secret",
            persona_id="clearance_secret",
            session_id=shared_session_id,
        )
        unclassified = await service.get_session(
            app_name=APP_NAME,
            user_id="clearance_unclassified",
            session_id=unclassified_session,
        )
        secret = await service.get_session(
            app_name=APP_NAME,
            user_id="clearance_secret",
            session_id=secret_session,
        )
        assert unclassified is not None
        assert secret is not None
        assert unclassified.state["persona_id"] == "clearance_unclassified"
        assert secret.state["persona_id"] == "clearance_secret"
        await service.close()

    asyncio.run(scenario())
