from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.config import get_settings
from defence_agent.retrieval.chunks import RetrievalChunk
from defence_agent.retrieval.document_pages import DocumentPage


TEXT_CHUNK_EMBED_BATCH_SIZE = 32


def embedding_dimension() -> int:
    """Return the dimension used by the active embedding backend."""

    return cohere_gateway.embedding_size()


def embedding_backend_name() -> str:
    """Return a human-readable embedding backend label."""

    settings = get_settings()
    return settings.cohere_embed_model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus passages for document search."""

    return cohere_gateway.embed_texts(texts, input_type="search_document")


def embed_query(query: str) -> list[float]:
    """Embed a user query for retrieval."""

    return cohere_gateway.embed_query(query)


def embed_pages(pages: list[DocumentPage]) -> list[list[float]]:
    """Embed rendered PDF pages with Cohere Embed v4 multimodal inputs.

    Each page is sent as text metadata plus the rendered page image, following
    Cohere's multimodal PDF search pattern.
    """

    embeddings: list[list[float]] = []
    for _, batch_embeddings in embed_page_batches(pages):
        embeddings.extend(batch_embeddings)
    return embeddings


def embed_page_batches(pages: list[DocumentPage]) -> Iterator[tuple[list[DocumentPage], list[list[float]]]]:
    """Embed rendered PDF pages in byte-aware live Cohere batches."""

    for page_batch in _batches(pages):
        embeddings = cohere_gateway.embed_inputs(
            [_cohere_page_input(page) for page in page_batch],
            input_type="search_document",
        )
        yield page_batch, embeddings


def embed_retrieval_chunk_batches(
    chunks: list[RetrievalChunk],
    pages_by_parent_id: dict[str, DocumentPage],
) -> Iterator[tuple[list[RetrievalChunk], list[list[float]]]]:
    """Embed retrieval chunks while preserving multimodal page embeddings where possible."""

    text_batch: list[RetrievalChunk] = []
    for chunk in chunks:
        parent = pages_by_parent_id.get(chunk.parent_page_id)
        if _use_parent_page_embedding(chunk, parent):
            if text_batch:
                yield text_batch, _embed_text_chunk_batch(text_batch)
                text_batch = []
            embeddings = cohere_gateway.embed_inputs([_cohere_page_input(parent)], input_type="search_document")
            yield [chunk], embeddings
            continue
        text_batch.append(chunk)
        if len(text_batch) >= TEXT_CHUNK_EMBED_BATCH_SIZE:
            yield text_batch, _embed_text_chunk_batch(text_batch)
            text_batch = []
    if text_batch:
        yield text_batch, _embed_text_chunk_batch(text_batch)


def _embed_text_chunk_batch(chunks: list[RetrievalChunk]) -> list[list[float]]:
    return cohere_gateway.embed_texts(
        [_chunk_embedding_text(chunk) for chunk in chunks],
        input_type="search_document",
    )


def _cohere_page_input(page: DocumentPage) -> dict[str, Any]:
    label = f"{page.doc_id} | page {page.page_number} | {page.metadata.get('title', '')}"
    content = [{"type": "text", "text": label}]
    if page.image_data_url:
        content.append({"type": "image_url", "image_url": {"url": page.image_data_url}})
    return {"content": content}


def _use_parent_page_embedding(chunk: RetrievalChunk, page: DocumentPage | None) -> bool:
    return bool(page and page.image_data_url and chunk.chunk_id == chunk.parent_page_id)


def _chunk_embedding_text(chunk: RetrievalChunk) -> str:
    metadata = chunk.metadata
    return "\n".join(
        [
            f"doc_id: {chunk.doc_id}",
            f"title: {metadata.get('title', '')}",
            f"page: {chunk.page_number}",
            f"chunk_strategy: {metadata.get('chunk_strategy', '')}",
            "",
            chunk.text,
        ]
    )


def _batches(items: list[DocumentPage]) -> list[list[DocumentPage]]:
    settings = get_settings()
    max_count = max(1, settings.cohere_embed_page_batch_size)
    max_bytes = max(1, settings.cohere_embed_page_batch_max_bytes)
    batches: list[list[DocumentPage]] = []
    index = 0
    while index < len(items):
        batch = [items[index]]
        batch_bytes = _page_input_size(items[index])
        index += 1
        while index < len(items) and len(batch) < max_count:
            next_page = items[index]
            next_bytes = _page_input_size(next_page)
            if batch_bytes + next_bytes > max_bytes:
                break
            batch.append(next_page)
            batch_bytes += next_bytes
            index += 1
        batches.append(batch)
    return batches


def _page_input_size(page: DocumentPage) -> int:
    label = f"{page.doc_id} | page {page.page_number} | {page.metadata.get('title', '')}"
    return len(label.encode("utf-8")) + len(page.image_data_url.encode("utf-8"))
