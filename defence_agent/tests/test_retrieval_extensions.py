from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from docx import Document
from pydantic import ValidationError
from reportlab.pdfgen import canvas

from defence_agent.auth.context import DEMO_USERS
from defence_agent.config import Settings, get_settings
from defence_agent.retrieval.bm25_index import bm25_search
from defence_agent.retrieval.chroma_index import _merge_retrieval_candidates, _promote_sources_to_parent_pages
from defence_agent.retrieval.chunks import RetrievalChunk, pages_by_id, retrieval_chunks_for_pages
from defence_agent.retrieval.document_pages import DocumentPage, load_document_pages


def test_local_parser_normalizes_pdf_and_docx_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("DEFENCE_AGENT_PARSER_BACKEND", "local")
    get_settings.cache_clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_pdf(corpus / "alpha.pdf", "Alpha PDF page text")
    _write_docx(corpus / "bravo.docx", ["Bravo DOCX paragraph", "Second paragraph"])
    (corpus / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "documents": [
                    {"path": "alpha.pdf", "doc_id": "DOC-PDF", "title": "Alpha PDF"},
                    {
                        "path": "bravo.docx",
                        "doc_id": "DOC-DOCX",
                        "title": "Bravo DOCX",
                        "source_format": "docx",
                        "normalized_format": "docx",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    pages = load_document_pages(corpus)

    assert {page.doc_id for page in pages} == {"DOC-PDF", "DOC-DOCX"}
    docx_page = next(page for page in pages if page.doc_id == "DOC-DOCX")
    assert docx_page.page_id == "DOC-DOCX_page_001"
    assert "Bravo DOCX paragraph" in docx_page.text
    assert docx_page.image_data_url == ""
    assert docx_page.metadata["parser_backend"] == "local"
    assert docx_page.metadata["logical_page"] is True


def test_compass_parser_backend_requires_parser_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("DEFENCE_AGENT_PARSER_BACKEND", "compass")
    monkeypatch.delenv("COMPASS_PARSER_URL", raising=False)

    with pytest.raises(ValidationError, match="COMPASS_PARSER_URL is required"):
        Settings()


def test_windowed_chunks_keep_parent_page_lineage() -> None:
    page = _page(
        text=" ".join(f"token{index}" for index in range(260)),
        page_id="DOC1_page_002",
        doc_id="DOC1",
        page=2,
    )

    chunks = retrieval_chunks_for_pages([page], "windowed")

    assert len(chunks) > 1
    assert {chunk.parent_page_id for chunk in chunks} == {"DOC1_page_002"}
    assert {chunk.metadata["page_id"] for chunk in chunks} == {"DOC1_page_002"}
    assert all(chunk.chunk_id.startswith("DOC1_page_002_chunk_") for chunk in chunks)


def test_bm25_returns_lexical_hits_and_filters_restricted_text() -> None:
    public = RetrievalChunk(
        chunk_id="PUB_page_001_chunk_001",
        parent_page_id="PUB_page_001",
        doc_id="PUB",
        page_number=1,
        text="unclassified planning note about maple interoperability",
        metadata={
            "doc_id": "PUB",
            "title": "Public",
            "page": 1,
            "access_level": "unclassified",
            "status": "approved",
            "language": "en",
            "allowed_roles": ["all"],
        },
    )
    secret = RetrievalChunk(
        chunk_id="SEC_page_001_chunk_001",
        parent_page_id="SEC_page_001",
        doc_id="SEC",
        page_number=1,
        text="secret release threshold 0.7342",
        metadata={
            "doc_id": "SEC",
            "title": "Secret",
            "page": 1,
            "access_level": "secret",
            "status": "approved",
            "language": "en",
            "allowed_roles": ["all"],
        },
    )

    unclassified_hits = bm25_search(
        query="release threshold 0.7342",
        chunks=[public, secret],
        auth=DEMO_USERS["clearance_unclassified"],
        top_n=5,
        status_filter="approved",
        language="any",
    )
    secret_hits = bm25_search(
        query="release threshold 0.7342",
        chunks=[public, secret],
        auth=DEMO_USERS["clearance_secret"],
        top_n=5,
        status_filter="approved",
        language="any",
    )

    assert [hit.chunk.doc_id for hit in unclassified_hits] == []
    assert [hit.chunk.doc_id for hit in secret_hits] == ["SEC"]


def test_hybrid_merge_deduplicates_by_parent_page() -> None:
    vector = {
        "chunk_id": "DOC1_page_001_chunk_001",
        "parent_page_id": "DOC1_page_001",
        "retrieval_chunk_id": "DOC1_page_001_chunk_001",
        "retrieval_chunk_text": "vector text",
        "vector_score": 0.6,
        "pre_rerank_score": 0.6,
        "retrieval_modes": ["vector"],
    }
    bm25 = {
        "chunk_id": "DOC1_page_001_chunk_002",
        "parent_page_id": "DOC1_page_001",
        "retrieval_chunk_id": "DOC1_page_001_chunk_002",
        "retrieval_chunk_text": "bm25 text",
        "bm25_score": 0.9,
        "pre_rerank_score": 0.9,
        "retrieval_modes": ["bm25"],
    }

    merged = _merge_retrieval_candidates([vector, bm25])

    assert len(merged) == 1
    assert merged[0]["retrieval_modes"] == ["bm25", "vector"]
    assert merged[0]["retrieval_chunk_id"] == "DOC1_page_001_chunk_002"
    assert merged[0]["bm25_score"] == 0.9
    assert merged[0]["vector_score"] == 0.6


def test_promoted_sources_use_parent_page_ids_for_cohere_citations() -> None:
    page = _page(text="Full parent page text for citations.", page_id="DOC1_page_003", doc_id="DOC1", page=3)
    source = {
        "chunk_id": "DOC1_page_003_chunk_002",
        "parent_page_id": "DOC1_page_003",
        "retrieval_chunk_id": "DOC1_page_003_chunk_002",
        "retrieval_chunk_text": "parent page",
        "rerank_score": 0.8,
    }

    promoted = _promote_sources_to_parent_pages([source], pages_by_id([page]))

    assert promoted[0]["chunk_id"] == "DOC1_page_003"
    assert promoted[0]["retrieval_chunk_id"] == "DOC1_page_003_chunk_002"
    assert promoted[0]["text"] == "Full parent page text for citations."


def _write_pdf(path: Path, text: str) -> None:
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, text)
    pdf.save()


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)


def _page(*, text: str, page_id: str, doc_id: str, page: int) -> DocumentPage:
    return DocumentPage(
        page_id=page_id,
        doc_id=doc_id,
        page_number=page,
        text=text,
        image_data_url="",
        metadata={
            "doc_id": doc_id,
            "page": page,
            "title": "Test Page",
            "access_level": "unclassified",
            "status": "approved",
            "language": "en",
            "allowed_roles": ["all"],
        },
    )
