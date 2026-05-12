"""Manifest-backed source preview helpers for the Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fitz
import yaml

from defence_agent.auth.context import DEMO_USERS
from defence_agent.auth.policy import policy_engine
from defence_agent.ui.view_model import SourceView


DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


@dataclass(frozen=True)
class SourcePreview:
    authorized: bool
    available: bool
    reason: str
    pdf_path: Path | None = None
    page: int | None = None
    page_text: str = ""
    page_image: bytes | None = None
    canonical_url: str = ""
    source_url: str = ""
    source_pdf_url: str = ""
    source_docx_url: str = ""
    official_pdf_page_url: str = ""
    publisher_url: str = ""
    original_docx_url: str = ""
    retrieved_date: str = ""
    source_organization: str = ""
    provenance_note: str = ""
    source_format: str = ""
    normalized_format: str = ""


def resolve_source_preview(
    source: SourceView,
    *,
    backend_persona_id: str,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
) -> SourcePreview:
    """Resolve a selected source to a previewable PDF after an ACL re-check."""

    if backend_persona_id not in DEMO_USERS:
        return SourcePreview(authorized=False, available=False, reason="unknown_persona")
    if not source.doc_id:
        return SourcePreview(authorized=False, available=False, reason="missing_doc_id")
    if not (source.used_in_answer or source.sent_to_model):
        return SourcePreview(authorized=False, available=False, reason="source_not_used_in_answer")

    manifest_entry = _manifest_entry_for_doc(source.doc_id, corpus_dir=corpus_dir)
    if manifest_entry is None:
        return SourcePreview(authorized=False, available=False, reason="missing_manifest_entry")
    if not _is_authorized(manifest_entry, backend_persona_id=backend_persona_id):
        return SourcePreview(authorized=False, available=False, reason="access_denied")

    page = _safe_page_number(source.page)
    lineage = _manifest_lineage(manifest_entry, page=page)
    pdf_path = _resolve_manifest_pdf(manifest_entry, corpus_dir=corpus_dir)
    if pdf_path is None:
        return SourcePreview(
            authorized=True,
            available=False,
            reason="missing_manifest_pdf",
            page=page,
            **lineage,
        )

    page_text = _extract_page_text(pdf_path, page)
    page_image = _render_page_image(pdf_path, page)
    return SourcePreview(
        authorized=True,
        available=True,
        reason="ok",
        pdf_path=pdf_path,
        page=page,
        page_text=page_text,
        page_image=page_image,
        **lineage,
    )


def _manifest_entry_for_doc(doc_id: str, *, corpus_dir: Path) -> dict[str, Any] | None:
    manifest_path = corpus_dir / "manifest.yaml"
    if not manifest_path.exists():
        return None
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    for entry in manifest.get("documents", []) or []:
        if isinstance(entry, dict) and str(entry.get("doc_id", "")) == doc_id:
            return entry
    return None


def _is_authorized(entry: dict[str, Any], *, backend_persona_id: str) -> bool:
    auth = DEMO_USERS[backend_persona_id]
    allowed_roles = entry.get("allowed_roles", ["all"])
    chunk = SimpleNamespace(
        tenant_id=str(entry.get("tenant_id", auth.tenant_id) or auth.tenant_id),
        classification=str(entry.get("access_level", "top_secret") or "top_secret"),
        allowed_roles_json=json.dumps(allowed_roles),
    )
    return policy_engine.can_access_chunk(auth, chunk)


def _resolve_manifest_pdf(entry: dict[str, Any], *, corpus_dir: Path) -> Path | None:
    raw_path = entry.get("path") or entry.get("pdf_path")
    if not raw_path:
        return None
    corpus_root = corpus_dir.resolve()
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = corpus_root / path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(corpus_root)
    except (FileNotFoundError, ValueError):
        return None
    return resolved


def _safe_page_number(value: str) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _manifest_lineage(entry: dict[str, Any], *, page: int | None) -> dict[str, str]:
    source_url = str(entry.get("source_url", "") or "")
    source_pdf_url = str(entry.get("source_pdf_url", "") or "")
    canonical_url = str(entry.get("canonical_url", "") or "")
    candidate_pdf_url = source_pdf_url or source_url
    pdf_url = candidate_pdf_url if _looks_like_pdf_url(candidate_pdf_url) else ""
    return {
        "canonical_url": canonical_url,
        "source_url": source_url,
        "source_pdf_url": source_pdf_url,
        "source_docx_url": str(entry.get("source_docx_url", "") or ""),
        "official_pdf_page_url": _page_url(pdf_url, page),
        "publisher_url": canonical_url,
        "original_docx_url": str(entry.get("source_docx_url", "") or ""),
        "retrieved_date": str(entry.get("retrieved_date", "") or ""),
        "source_organization": str(entry.get("source_organization", "") or entry.get("owner", "") or ""),
        "provenance_note": str(entry.get("provenance_note", "") or ""),
        "source_format": str(entry.get("source_format", "") or ""),
        "normalized_format": str(entry.get("normalized_format", "") or ""),
    }


def _looks_like_pdf_url(url: str) -> bool:
    clean = str(url or "").lower()
    return clean.endswith(".pdf") or ".pdf?" in clean or "/pdf" in clean


def _page_url(url: str, page: int | None) -> str:
    if not url:
        return ""
    if page is None:
        return url
    base = url.split("#", 1)[0]
    return f"{base}#page={page}"


def _extract_page_text(pdf_path: Path, page: int | None) -> str:
    if page is None:
        return ""
    try:
        with fitz.open(pdf_path) as document:
            index = page - 1
            if index < 0 or index >= len(document):
                return ""
            return document[index].get_text("text").strip()
    except Exception:
        return ""


def _render_page_image(pdf_path: Path, page: int | None) -> bytes | None:
    if page is None:
        return None
    try:
        with fitz.open(pdf_path) as document:
            index = page - 1
            if index < 0 or index >= len(document):
                return None
            pixmap = document[index].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            return pixmap.tobytes("png")
    except Exception:
        return None
