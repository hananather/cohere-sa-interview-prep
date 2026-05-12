from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.config import get_settings
from defence_agent.retrieval.document_pages import DocumentPage


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


def _cohere_page_input(page: DocumentPage) -> dict[str, Any]:
    label = f"{page.doc_id} | page {page.page_number} | {page.metadata.get('title', '')}"
    return {
        "content": [
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {"url": page.image_data_url}},
        ]
    }


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
