from __future__ import annotations

from pathlib import Path

from defence_agent.ui.source_catalog import load_source_catalog, source_catalog_for_persona


ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "defence_agent" / "data" / "corpus"


def test_catalog_loads_manifest_documents() -> None:
    rows = load_source_catalog(CORPUS_DIR)

    assert len(rows) == 10


def test_catalog_page_count_total_is_current_manifest_total() -> None:
    rows = load_source_catalog(CORPUS_DIR)

    assert sum(row.pages for row in rows) == 217


def test_docx_origin_row_preserves_normalization_metadata() -> None:
    rows = load_source_catalog(CORPUS_DIR)
    row = next(item for item in rows if item.document_id == "UK-MOD-ASOEM-2023-EN")

    assert row.source_format == "docx"
    assert row.indexed_format == "pdf"
    assert row.normalization_method == "official_pdf_pair"
    assert row.dataset == "DOCX-origin"


def test_scanned_manual_row_is_labeled_as_scanned() -> None:
    rows = load_source_catalog(CORPUS_DIR)
    row = next(item for item in rows if item.document_id == "US-ARMY-FM30-16-1972-SCAN")

    assert row.dataset == "Scanned"
    assert row.source_format == "scanned_pdf"
    assert row.indexed_format == "pdf"
    assert row.pages == 10


def test_persona_a_catalog_shows_only_unclassified_rows() -> None:
    catalog = source_catalog_for_persona("persona_a", corpus_dir=CORPUS_DIR)

    assert len(catalog.rows) == 8
    assert catalog.withheld_count == 2
    assert {row.access_level for row in catalog.rows} == {"unclassified"}
    assert "SYN-FUSION-S-RELEASE-001" not in {row.document_id for row in catalog.rows}
    assert "SYN-FUSION-TS-ANNEX-002" not in {row.document_id for row in catalog.rows}


def test_persona_b_catalog_shows_all_rows() -> None:
    catalog = source_catalog_for_persona("persona_b", corpus_dir=CORPUS_DIR)

    assert len(catalog.rows) == 10
    assert catalog.withheld_count == 0
    assert sum(row.pages for row in catalog.rows) == 217
    assert {"unclassified", "secret", "top_secret"} == {row.access_level for row in catalog.rows}
