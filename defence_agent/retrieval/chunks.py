"""Retrieval chunks that preserve page-level traceability."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from defence_agent.retrieval.document_pages import DocumentPage


WINDOW_TOKEN_COUNT = 180
WINDOW_TOKEN_OVERLAP = 40


@dataclass(frozen=True)
class RetrievalChunk:
    """Search unit mapped back to a citation-ready parent page."""

    chunk_id: str
    parent_page_id: str
    doc_id: str
    page_number: int
    text: str
    metadata: dict[str, Any]


def retrieval_chunks_for_pages(pages: list[DocumentPage], strategy: str = "page") -> list[RetrievalChunk]:
    """Return retrieval chunks while keeping stable page parent IDs."""

    normalized = str(strategy or "page").strip().lower()
    if normalized == "page":
        return [_page_chunk(page) for page in pages]
    if normalized == "windowed":
        chunks: list[RetrievalChunk] = []
        for page in pages:
            chunks.extend(_windowed_chunks(page))
        return chunks
    raise ValueError(f"Unsupported chunk strategy: {strategy}")


def pages_by_id(pages: list[DocumentPage]) -> dict[str, DocumentPage]:
    """Return page objects keyed by stable page ID."""

    return {page.page_id: page for page in pages}


def page_text_sha256(page: DocumentPage) -> str:
    return _sha256_text(page.text)


def chunk_text_sha256(chunk: RetrievalChunk) -> str:
    return _sha256_text(chunk.text)


def _page_chunk(page: DocumentPage) -> RetrievalChunk:
    metadata = _chunk_metadata(page, chunk_id=page.page_id, chunk_index=1, strategy="page")
    return RetrievalChunk(
        chunk_id=page.page_id,
        parent_page_id=page.page_id,
        doc_id=page.doc_id,
        page_number=page.page_number,
        text=page.text,
        metadata=metadata,
    )


def _windowed_chunks(page: DocumentPage) -> list[RetrievalChunk]:
    tokens = _tokens_with_offsets(page.text)
    if not tokens:
        chunk_id = f"{page.page_id}_chunk_001"
        return [
            RetrievalChunk(
                chunk_id=chunk_id,
                parent_page_id=page.page_id,
                doc_id=page.doc_id,
                page_number=page.page_number,
                text=page.text,
                metadata=_chunk_metadata(page, chunk_id=chunk_id, chunk_index=1, strategy="windowed"),
            )
        ]

    chunks: list[RetrievalChunk] = []
    start = 0
    index = 1
    step = max(1, WINDOW_TOKEN_COUNT - WINDOW_TOKEN_OVERLAP)
    while start < len(tokens):
        window = tokens[start : start + WINDOW_TOKEN_COUNT]
        text_start = window[0][1]
        text_end = window[-1][2]
        chunk_text = page.text[text_start:text_end].strip()
        chunk_id = f"{page.page_id}_chunk_{index:03d}"
        chunks.append(
            RetrievalChunk(
                chunk_id=chunk_id,
                parent_page_id=page.page_id,
                doc_id=page.doc_id,
                page_number=page.page_number,
                text=chunk_text,
                metadata=_chunk_metadata(page, chunk_id=chunk_id, chunk_index=index, strategy="windowed"),
            )
        )
        if start + WINDOW_TOKEN_COUNT >= len(tokens):
            break
        start += step
        index += 1
    return chunks


def _chunk_metadata(
    page: DocumentPage,
    *,
    chunk_id: str,
    chunk_index: int,
    strategy: str,
) -> dict[str, Any]:
    metadata = dict(page.metadata)
    metadata.update(
        {
            "chunk_id": chunk_id,
            "parent_page_id": page.page_id,
            "chunk_index": chunk_index,
            "chunk_strategy": strategy,
            "page_id": page.page_id,
            "page_text_sha256": page_text_sha256(page),
        }
    )
    return metadata


def _tokens_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
