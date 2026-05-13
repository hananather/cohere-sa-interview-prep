"""Persona-filtered source catalog helpers for the Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fitz
import yaml

from defence_agent.auth.context import DEMO_USERS
from defence_agent.auth.policy import policy_engine
from defence_agent.ui.view_model import DEFAULT_UI_PERSONA_ID, UiPersona, persona_for_ui_id


DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"
CATALOG_FILTERS = ("All", "Official", "DOCX-origin", "Synthetic")
CATALOG_COLUMNS = (
    "Dataset",
    "Document",
    "Title",
    "Access",
    "Language",
    "Source format",
    "Indexed format",
    "Pages",
    "Owner",
    "Retrieved",
    "Source",
)


@dataclass(frozen=True)
class SourceCatalogRow:
    dataset: str
    document_id: str
    title: str
    access_level: str
    language: str
    source_format: str
    indexed_format: str
    pages: int
    owner: str
    retrieved: str
    source: str
    source_type: str
    normalization_method: str
    allowed_roles: tuple[str, ...]

    def as_table_row(self) -> dict[str, object]:
        return {
            "Dataset": self.dataset,
            "Document": self.document_id,
            "Title": self.title,
            "Access": self.access_level,
            "Language": self.language,
            "Source format": self.source_format,
            "Indexed format": self.indexed_format,
            "Pages": self.pages,
            "Owner": self.owner,
            "Retrieved": self.retrieved,
            "Source": self.source,
        }


@dataclass(frozen=True)
class SourceCatalogView:
    persona: UiPersona
    rows: tuple[SourceCatalogRow, ...]
    withheld_count: int
    total_count: int

    @property
    def visible_pages(self) -> int:
        return sum(row.pages for row in self.rows)

    @property
    def dataset_count(self) -> int:
        return len({row.dataset for row in self.rows})


def load_source_catalog(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> tuple[SourceCatalogRow, ...]:
    """Load all manifest rows and compute page counts from local PDF files."""

    corpus_root = corpus_dir.resolve()
    manifest_path = corpus_root / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Corpus manifest does not exist: {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    documents = manifest.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError(f"{manifest_path} must contain a documents list")

    rows: list[SourceCatalogRow] = []
    for entry in documents:
        if not isinstance(entry, dict):
            raise ValueError(f"{manifest_path} contains a non-object document entry")
        rows.append(_row_from_manifest_entry(entry, corpus_root=corpus_root))
    return tuple(rows)


def source_catalog_for_persona(
    ui_persona_id: str = DEFAULT_UI_PERSONA_ID,
    *,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
) -> SourceCatalogView:
    """Return only rows authorized for the selected UI persona."""

    persona = persona_for_ui_id(ui_persona_id)
    rows = load_source_catalog(corpus_dir=corpus_dir)
    visible_rows = tuple(row for row in rows if _is_authorized(row, persona=persona))
    return SourceCatalogView(
        persona=persona,
        rows=visible_rows,
        withheld_count=len(rows) - len(visible_rows),
        total_count=len(rows),
    )


def filter_source_rows(
    rows: tuple[SourceCatalogRow, ...] | list[SourceCatalogRow],
    *,
    dataset_filter: str = "All",
    search_query: str = "",
) -> tuple[SourceCatalogRow, ...]:
    """Apply UI filters over already-authorized catalog rows."""

    normalized_filter = dataset_filter if dataset_filter in CATALOG_FILTERS else "All"
    query = " ".join(str(search_query or "").lower().split())
    filtered: list[SourceCatalogRow] = []
    for row in rows:
        if normalized_filter != "All" and row.dataset != normalized_filter:
            continue
        if query and not _matches_query(row, query):
            continue
        filtered.append(row)
    return tuple(filtered)


def source_catalog_table_rows(rows: tuple[SourceCatalogRow, ...] | list[SourceCatalogRow]) -> list[dict[str, object]]:
    return [{column: row.as_table_row()[column] for column in CATALOG_COLUMNS} for row in rows]


def _row_from_manifest_entry(entry: dict[str, Any], *, corpus_root: Path) -> SourceCatalogRow:
    pdf_path = _resolve_manifest_pdf(entry, corpus_root=corpus_root)
    return SourceCatalogRow(
        dataset=_dataset_label(entry),
        document_id=str(entry.get("doc_id", "") or ""),
        title=str(entry.get("title", "") or ""),
        access_level=str(entry.get("access_level", "top_secret") or "top_secret"),
        language=str(entry.get("language", "") or ""),
        source_format=str(entry.get("source_format", "pdf") or "pdf"),
        indexed_format=str(entry.get("normalized_format", "pdf") or "pdf"),
        pages=_pdf_page_count(pdf_path),
        owner=str(entry.get("owner", "") or entry.get("source_organization", "") or ""),
        retrieved=str(entry.get("retrieved_date", "") or ""),
        source=_source_label(entry),
        source_type=str(entry.get("source_type", "") or ""),
        normalization_method=str(entry.get("normalization_method", "") or ""),
        allowed_roles=tuple(str(item) for item in entry.get("allowed_roles", ["all"]) or ["all"]),
    )


def _dataset_label(entry: dict[str, Any]) -> str:
    source_type = str(entry.get("source_type", "") or "").lower()
    if bool(entry.get("synthetic")) or source_type.startswith("synthetic"):
        return "Synthetic"
    if str(entry.get("source_format", "") or "").lower() == "docx" or "docx_origin" in source_type:
        return "DOCX-origin"
    return "Official"


def _source_label(entry: dict[str, Any]) -> str:
    for key in ("source_url", "source_pdf_url", "canonical_url", "source_docx_url"):
        value = str(entry.get(key, "") or "").strip()
        if value:
            return value
    if bool(entry.get("synthetic")):
        return "Local synthetic exercise source"
    return ""


def _resolve_manifest_pdf(entry: dict[str, Any], *, corpus_root: Path) -> Path:
    raw_path = entry.get("path") or entry.get("pdf_path")
    if not raw_path:
        raise ValueError("Manifest document entry is missing path")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = corpus_root / path
    resolved = path.resolve(strict=True)
    resolved.relative_to(corpus_root)
    return resolved


def _pdf_page_count(pdf_path: Path) -> int:
    stat = pdf_path.stat()
    return _pdf_page_count_cached(str(pdf_path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _pdf_page_count_cached(pdf_path: str, mtime_ns: int, size: int) -> int:
    # The mtime and size arguments are cache invalidators.
    _ = (mtime_ns, size)
    with fitz.open(pdf_path) as document:
        return len(document)


def _is_authorized(row: SourceCatalogRow, *, persona: UiPersona) -> bool:
    auth = DEMO_USERS[persona.backend_persona_id]
    chunk = SimpleNamespace(
        tenant_id=auth.tenant_id,
        classification=row.access_level,
        allowed_roles_json=json.dumps(list(row.allowed_roles)),
    )
    return policy_engine.can_access_chunk(auth, chunk)


def _matches_query(row: SourceCatalogRow, query: str) -> bool:
    haystack = " ".join(
        (
            row.title,
            row.document_id,
            row.owner,
            row.dataset,
        )
    ).lower()
    return query in haystack
