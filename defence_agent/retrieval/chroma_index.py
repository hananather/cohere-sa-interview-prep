from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import chromadb
import yaml

from defence_agent.auth.context import DEFAULT_PERSONA_ID, DEMO_USERS
from defence_agent.auth.policy import policy_engine
from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.config import get_settings
from defence_agent.retrieval.bm25_index import bm25_search
from defence_agent.retrieval.chunks import (
    RetrievalChunk,
    chunk_text_sha256,
    pages_by_id,
    retrieval_chunks_for_pages,
)
from defence_agent.retrieval.document_pages import DocumentPage, load_document_pages, metadata_for_vector_store
from defence_agent.retrieval.embeddings import (
    embed_query,
    embed_retrieval_chunk_batches,
    embedding_backend_name,
    embedding_dimension,
)
from defence_agent.retrieval.index_metadata import COLLECTION_PREFIX, INDEX_VERSION


MULTILINGUAL_RERANK_MARGIN = 0.08

_ANSWERABILITY_STOPWORDS = {
    "about",
    "after",
    "also",
    "before",
    "does",
    "from",
    "have",
    "into",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "without",
}


def build_index(force: bool = False) -> dict[str, Any]:
    """Build or refresh the local Chroma index from rendered PDF pages.

    The build is idempotent by default. Existing pages are reused when their
    source page hashes and Cohere embedding metadata still match the corpus.
    """

    with _index_build_lock():
        return _build_index_locked(force=force)


def _build_index_locked(force: bool = False) -> dict[str, Any]:
    """Build the index while holding the process-level index lock."""

    client = _client()
    collection_name = _collection_name()
    pages = load_document_pages()
    chunks = retrieval_chunks_for_pages(pages, get_settings().chunk_strategy)
    page_lookup = pages_by_id(pages)
    collection_metadata = _collection_metadata(pages, chunks)

    if force:
        _delete_collection_if_exists(client, collection_name)

    collection = client.get_or_create_collection(collection_name, metadata=collection_metadata)
    existing_by_id = _existing_metadata_by_id(collection)
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    orphan_ids = sorted(set(existing_by_id) - chunk_ids)
    if orphan_ids:
        collection.delete(ids=orphan_ids)
        existing_by_id = {key: value for key, value in existing_by_id.items() if key not in orphan_ids}

    chunks_to_embed = [chunk for chunk in chunks if _chunk_needs_embedding(chunk, existing_by_id.get(chunk.chunk_id))]
    if not chunks_to_embed:
        _update_collection_metadata(collection, collection_metadata)
        return {
            "collection": collection_name,
            "embedding_backend": embedding_backend_name(),
            "embedding_dimension": embedding_dimension(),
            "indexed_chunks": collection.count(),
            "parsed_pages": len(pages),
            "retrieval_chunks": len(chunks),
            "chunk_strategy": get_settings().chunk_strategy,
            "embedded_pages": 0,
            "embedded_chunks": 0,
            "orphaned_pages_deleted": len(orphan_ids),
            "orphaned_chunks_deleted": len(orphan_ids),
            "skipped": True,
        }

    embedded_chunks = 0
    for chunk_batch, embeddings in embed_retrieval_chunk_batches(chunks_to_embed, page_lookup):
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunk_batch],
            documents=[chunk.text for chunk in chunk_batch],
            embeddings=embeddings,
            metadatas=[_metadata_for_index(chunk) for chunk in chunk_batch],
        )
        embedded_chunks += len(chunk_batch)
    _update_collection_metadata(collection, collection_metadata)
    return {
        "collection": collection_name,
        "embedding_backend": embedding_backend_name(),
        "embedding_dimension": embedding_dimension(),
        "indexed_chunks": collection.count(),
        "parsed_pages": len(pages),
        "retrieval_chunks": len(chunks),
        "chunk_strategy": get_settings().chunk_strategy,
        "embedded_pages": embedded_chunks,
        "embedded_chunks": embedded_chunks,
        "orphaned_pages_deleted": len(orphan_ids),
        "orphaned_chunks_deleted": len(orphan_ids),
        "skipped": False,
    }


def search_index(
    query: str,
    persona_id: str = DEFAULT_PERSONA_ID,
    top_k: int = 5,
    status_filter: str = "approved",
    language: str = "any",
) -> dict[str, Any]:
    """Search local Chroma, then enforce persona, status, and language filters."""

    settings = get_settings()
    retrieval_mode = settings.retrieval_mode
    if retrieval_mode in {"vector", "hybrid"}:
        build_index(force=False)
    auth = DEMO_USERS.get(persona_id, DEMO_USERS[DEFAULT_PERSONA_ID])
    allowed_access = list(policy_engine.acl_filter(auth).allowed_classifications)
    normalized_status = _normalize_status_filter(status_filter)
    normalized_language = _normalize_language_filter(language)
    pages = load_document_pages()
    page_lookup = pages_by_id(pages)
    chunks = retrieval_chunks_for_pages(pages, settings.chunk_strategy)

    vector_candidates: list[dict[str, Any]] = []
    bm25_candidates: list[dict[str, Any]] = []
    excluded_sources: list[dict[str, Any]]

    if retrieval_mode in {"vector", "hybrid"}:
        collection = _client().get_collection(_collection_name())
        vector_candidates, excluded_sources = _vector_candidates(
            query=query,
            auth=auth,
            collection=collection,
            top_k=top_k,
            allowed_access=allowed_access,
            status_filter=normalized_status,
            language=normalized_language,
        )
    else:
        excluded_sources = _excluded_sources_from_pages(query, pages, auth)

    if retrieval_mode in {"bm25", "hybrid"}:
        bm25_candidates = _bm25_candidates(
            query=query,
            chunks=chunks,
            auth=auth,
            top_k=top_k,
            status_filter=normalized_status,
            language=normalized_language,
        )

    merged_candidates = _merge_retrieval_candidates(vector_candidates + bm25_candidates)
    reranked = _rerank(query, merged_candidates)
    selected_chunks = _select_sources(reranked, top_k, language=normalized_language)
    authorized_sources = _promote_sources_to_parent_pages(selected_chunks, page_lookup)
    for index, source in enumerate(authorized_sources, start=1):
        source["citation_id"] = f"C{index}"
        source["citation"] = f"[C{index}]"
    answerability = _answerability(query, authorized_sources, excluded_sources)

    return {
        "index": _index_label(retrieval_mode),
        "collection": _collection_name(),
        "embedding_backend": embedding_backend_name(),
        "retrieval_mode": retrieval_mode,
        "chunk_strategy": settings.chunk_strategy,
        "rerank_backend": get_settings().cohere_rerank_model,
        "persona_id": auth.user_id,
        "allowed_access": allowed_access,
        "filters_applied": {
            "access_level": allowed_access,
            "status": normalized_status,
            "language": normalized_language,
        },
        "retrieval_metrics": {
            "vector_candidate_count": len(vector_candidates),
            "bm25_candidate_count": len(bm25_candidates),
            "merged_candidate_count": len(merged_candidates),
            "reranked_candidate_count": len(reranked),
            "selected_parent_page_count": len(authorized_sources),
        },
        "policy_decision": _policy_decision(answerability),
        "answerability": answerability,
        "authorized_sources": authorized_sources,
        "citation_guide": [
            f"{source['citation']} = {source['doc_id']} page {source['page']} ({source['title']})"
            for source in authorized_sources
        ],
        "excluded_sources": excluded_sources,
        "answering_rule": (
            "Answer only from authorized_sources. Every sentence with source-backed facts must end with "
            "the bracket citation value, for example: 'The initial acknowledgement is due within 15 minutes [C1].' "
            "Never reveal excluded source text."
        ),
    }


def parse_only() -> list[DocumentPage]:
    """Expose parsed page objects for tests and one-off inspection."""

    return load_document_pages()


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(get_settings().data_dir / "chroma"))


def _collection_name() -> str:
    strategy = get_settings().chunk_strategy
    if strategy == "page":
        return f"{COLLECTION_PREFIX}_{embedding_dimension()}"
    return f"{COLLECTION_PREFIX}_{strategy}_{embedding_dimension()}"


@contextmanager
def _index_build_lock() -> Any:
    """Serialize live index builds so parallel agents do not duplicate API calls."""

    import fcntl

    lock_path = get_settings().data_dir / "index_build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _collection_metadata(
    pages: list[DocumentPage],
    chunks: list[RetrievalChunk],
) -> dict[str, str | int | float]:
    settings = get_settings()
    return {
        "index_version": INDEX_VERSION,
        "embedding_model": settings.cohere_embed_model,
        "embedding_dimension": settings.cohere_embed_output_dimension,
        "parser_backend": settings.parser_backend,
        "retrieval_mode": settings.retrieval_mode,
        "chunk_strategy": settings.chunk_strategy,
        "corpus_manifest_sha256": _manifest_sha256(settings.corpus_dir / "manifest.yaml"),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "page_hash_manifest_sha256": _page_hash_manifest_sha256(pages),
        "chunk_hash_manifest_sha256": _chunk_hash_manifest_sha256(chunks),
        "updated_at_epoch": time(),
    }


def _metadata_for_index(chunk: RetrievalChunk) -> dict[str, str | int | float | bool]:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "chunk_id": chunk.chunk_id,
            "parent_page_id": chunk.parent_page_id,
            "index_version": INDEX_VERSION,
            "embedding_model": get_settings().cohere_embed_model,
            "embedding_dimension": embedding_dimension(),
            "chunk_text_sha256": chunk_text_sha256(chunk),
            "indexed_at_epoch": time(),
        }
    )
    return metadata_for_vector_store(metadata)


def _existing_metadata_by_id(collection: Any) -> dict[str, dict[str, Any]]:
    existing = collection.get(include=["metadatas"])
    ids = existing.get("ids", []) or []
    metadatas = existing.get("metadatas", []) or []
    return {
        str(page_id): metadata
        for page_id, metadata in zip(ids, metadatas)
        if isinstance(metadata, dict)
    }


def _chunk_needs_embedding(chunk: RetrievalChunk, existing: dict[str, Any] | None) -> bool:
    if not existing:
        return True
    return any(
        [
            str(existing.get("index_version", "")) != INDEX_VERSION,
            str(existing.get("embedding_model", "")) != get_settings().cohere_embed_model,
            int(existing.get("embedding_dimension") or 0) != embedding_dimension(),
            str(existing.get("page_image_sha256", "")) != str(chunk.metadata.get("page_image_sha256", "")),
            str(existing.get("chunk_text_sha256") or existing.get("page_text_sha256", ""))
            != chunk_text_sha256(chunk),
        ]
    )


def _update_collection_metadata(collection: Any, metadata: dict[str, str | int | float]) -> None:
    try:
        collection.modify(metadata=metadata)
    except Exception:
        return


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _page_hash_manifest_sha256(pages: list[DocumentPage]) -> str:
    payload = "\n".join(
        f"{page.page_id}:{page.metadata.get('page_image_sha256', '')}:{_sha256_text(page.text)}"
        for page in pages
    )
    return _sha256_text(payload)


def _chunk_hash_manifest_sha256(chunks: list[RetrievalChunk]) -> str:
    payload = "\n".join(
        f"{chunk.chunk_id}:{chunk.parent_page_id}:{chunk_text_sha256(chunk)}"
        for chunk in chunks
    )
    return _sha256_text(payload)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _delete_collection_if_exists(client: chromadb.PersistentClient, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name)
    except Exception:
        return


def _normalize_status_filter(status_filter: str) -> str:
    normalized = str(status_filter or "approved").strip().lower()
    aliases = {
        "current": "approved",
        "current_approved": "approved",
        "old": "superseded",
        "older": "superseded",
        "all": "any",
        "*": "any",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"approved", "draft", "superseded", "any"}:
        return "approved"
    return normalized


def _normalize_language_filter(language: str) -> str:
    normalized = str(language or "any").strip().lower()
    aliases = {
        "english": "en",
        "eng": "en",
        "french": "fr",
        "français": "fr",
        "francais": "fr",
        "all": "any",
        "*": "any",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"en", "fr", "any"}:
        return "any"
    return normalized


def _where_filter(allowed_access: list[str], status_filter: str, language: str) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"access_level": {"$in": allowed_access}}]
    if status_filter != "any":
        filters.append({"status": status_filter})
    if language != "any":
        filters.append({"language": language})
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _vector_candidates(
    *,
    query: str,
    auth: Any,
    collection: Any,
    top_k: int,
    allowed_access: list[str],
    status_filter: str,
    language: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_embedding = embed_query(query)
    where = _where_filter(allowed_access, status_filter, language)
    candidate_count = min(collection.count(), max(top_k * 8, 80))
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    raw_for_exclusions = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["metadatas", "distances"],
    )
    candidates: list[dict[str, Any]] = []
    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        source = _authorized_source(chunk_id, text, metadata, distance, len(candidates) + 1)
        source["retrieval_modes"] = ["vector"]
        source["pre_rerank_score"] = source.get("vector_score")
        candidates.append(source)
    return candidates, _excluded_sources(raw_for_exclusions, auth)


def _bm25_candidates(
    *,
    query: str,
    chunks: list[RetrievalChunk],
    auth: Any,
    top_k: int,
    status_filter: str,
    language: str,
) -> list[dict[str, Any]]:
    hits = bm25_search(
        query=query,
        chunks=chunks,
        auth=auth,
        top_n=max(top_k * 8, 80),
        status_filter=status_filter,
        language=language,
    )
    max_score = max((hit.score for hit in hits), default=0.0)
    candidates: list[dict[str, Any]] = []
    for hit in hits:
        score = round(hit.score / max_score, 4) if max_score > 0 else 0.0
        metadata = dict(hit.chunk.metadata)
        source = _authorized_source(
            hit.chunk.chunk_id,
            hit.chunk.text,
            metadata,
            distance=(1.0 / max(score, 0.0001)) - 1.0,
            citation_index=hit.rank,
        )
        source["vector_score"] = None
        source["bm25_score"] = score
        source["bm25_raw_score"] = round(hit.score, 4)
        source["bm25_rank"] = hit.rank
        source["retrieval_modes"] = ["bm25"]
        source["pre_rerank_score"] = score
        candidates.append(source)
    return candidates


def _merge_retrieval_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = _source_selection_key(candidate)
        if not key:
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(candidate)
            continue
        existing_modes = set(existing.get("retrieval_modes", []) or [])
        existing_modes.update(candidate.get("retrieval_modes", []) or [])
        existing["retrieval_modes"] = sorted(existing_modes)
        existing_pre_score = _candidate_pre_score(existing)
        candidate_pre_score = _candidate_pre_score(candidate)
        for field in ("vector_score", "bm25_score", "bm25_raw_score"):
            existing[field] = _max_numeric(existing.get(field), candidate.get(field))
        existing["pre_rerank_score"] = _max_numeric(
            existing.get("pre_rerank_score"),
            candidate.get("pre_rerank_score"),
        )
        if candidate_pre_score > existing_pre_score:
            existing["text"] = candidate.get("text", existing.get("text", ""))
            existing["retrieval_chunk_id"] = candidate.get("retrieval_chunk_id", candidate.get("chunk_id", ""))
            existing["retrieval_chunk_text"] = candidate.get("retrieval_chunk_text", candidate.get("text", ""))
    return sorted(merged.values(), key=_candidate_pre_score, reverse=True)


def _promote_sources_to_parent_pages(
    sources: list[dict[str, Any]],
    page_lookup: dict[str, DocumentPage],
) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for source in sources:
        parent_id = str(source.get("parent_page_id") or source.get("chunk_id") or "")
        page = page_lookup.get(parent_id)
        if page is None:
            promoted.append(source)
            continue
        merged = dict(source)
        retrieval_chunk_id = str(source.get("retrieval_chunk_id") or source.get("chunk_id") or "")
        retrieval_chunk_text = str(source.get("retrieval_chunk_text") or source.get("text") or "")
        merged.update(page.metadata)
        merged.update(
            {
                "chunk_id": page.page_id,
                "parent_page_id": page.page_id,
                "retrieval_chunk_id": retrieval_chunk_id,
                "retrieval_chunk_text": retrieval_chunk_text,
                "text": page.text,
                "page": page.page_number,
                "doc_id": page.doc_id,
            }
        )
        for field in (
            "vector_score",
            "bm25_score",
            "bm25_raw_score",
            "bm25_rank",
            "pre_rerank_score",
            "rerank_score",
            "retrieval_modes",
        ):
            if field in source:
                merged[field] = source[field]
        promoted.append(merged)
    return promoted


def _excluded_sources_from_pages(
    query: str,
    pages: list[DocumentPage],
    auth: Any,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in pages:
        metadata = page.metadata
        doc_id = str(metadata.get("doc_id", ""))
        if not doc_id or doc_id in seen:
            continue
        if policy_engine.can_access_chunk(auth, _ChunkForPolicy(metadata)):
            continue
        source = _excluded_source(metadata, "access_denied")
        source["metadata_overlap"] = _metadata_overlap(query, source)
        excluded.append(source)
        seen.add(doc_id)
    return sorted(excluded, key=lambda item: int(item.get("metadata_overlap", 0) or 0), reverse=True)[:limit]


def _index_label(retrieval_mode: str) -> str:
    if retrieval_mode == "hybrid":
        return "chroma+bm25"
    if retrieval_mode == "bm25":
        return "bm25"
    return "chroma"


def _max_numeric(left: Any, right: Any) -> float | None:
    values = []
    for value in (left, right):
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return round(max(values), 4) if values else None


def _candidate_pre_score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("pre_rerank_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _authorized_source(
    chunk_id: str,
    text: str,
    metadata: dict[str, Any],
    distance: float,
    citation_index: int,
) -> dict[str, Any]:
    return {
        "citation_id": f"C{citation_index}",
        "chunk_id": chunk_id,
        "parent_page_id": metadata.get("parent_page_id", chunk_id),
        "retrieval_chunk_id": metadata.get("chunk_id", chunk_id),
        "retrieval_chunk_text": text,
        "chunk_strategy": metadata.get("chunk_strategy", "page"),
        "doc_id": metadata.get("doc_id", ""),
        "title": metadata.get("title", ""),
        "section": metadata.get("section", ""),
        "page": metadata.get("page", 1),
        "row_id": metadata.get("row_id", ""),
        "status": metadata.get("status", ""),
        "version": metadata.get("version", ""),
        "effective_date": metadata.get("effective_date", ""),
        "access_level": metadata.get("access_level", ""),
        "language": metadata.get("language", ""),
        "source_type": metadata.get("source_type", ""),
        "source_format": metadata.get("source_format", ""),
        "normalized_format": metadata.get("normalized_format", ""),
        "normalization_method": metadata.get("normalization_method", ""),
        "canonical_url": metadata.get("canonical_url", ""),
        "source_url": metadata.get("source_url", ""),
        "retrieved_date": metadata.get("retrieved_date", ""),
        "source_organization": metadata.get("source_organization", ""),
        "owner": metadata.get("owner", ""),
        "source_docx_url": metadata.get("source_docx_url", ""),
        "source_pdf_url": metadata.get("source_pdf_url", ""),
        "provenance_note": metadata.get("provenance_note", ""),
        "source_pdf_path": metadata.get("source_pdf_path", ""),
        "manifest_path": metadata.get("manifest_path", ""),
        "page_image_sha256": metadata.get("page_image_sha256", ""),
        "vector_score": round(1.0 / (1.0 + float(distance)), 4),
        "rerank_score": None,
        "text": text,
    }


def _excluded_source(metadata: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "doc_id": str(metadata.get("doc_id", "")),
        "title": str(metadata.get("title", "")),
        "access_level": str(metadata.get("access_level", "")),
        "doc_family": str(metadata.get("doc_family", "")),
        "language": str(metadata.get("language", "")),
        "reason": reason,
    }


@dataclass(frozen=True)
class _ChunkForPolicy:
    metadata: dict[str, Any]

    @property
    def tenant_id(self) -> str:
        return "deftech"

    @property
    def classification(self) -> str:
        return str(self.metadata.get("access_level", "top_secret"))

    @property
    def allowed_roles_json(self) -> str:
        raw = self.metadata.get("allowed_roles", "[]")
        if isinstance(raw, str):
            return raw
        return json.dumps(raw)


def _excluded_sources(raw: dict[str, Any], auth: Any) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for metadata, distance in zip(metadatas, distances):
        doc_id = str(metadata.get("doc_id", ""))
        if not doc_id or doc_id in seen:
            continue
        if policy_engine.can_access_chunk(auth, _ChunkForPolicy(metadata)):
            continue
        source = _excluded_source(metadata, "access_denied")
        source["vector_score"] = round(1.0 / (1.0 + float(distance)), 4)
        excluded.append(source)
        seen.add(doc_id)
    return excluded


def _answerability(
    query: str,
    authorized_sources: list[dict[str, Any]],
    excluded_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record evidence sufficiency signals for grounded generation.

    This is intentionally metadata/text-light. It prevents the final answer
    layer from using unrelated public pages when the best match is denied, while
    allowing authorized evidence gaps to reach model-grounded abstention.
    """

    if not authorized_sources:
        best_denied = _best_denied_overlap(query, excluded_sources)
        return {
            "answerable": False,
            "reason": "no_authorized_sources",
            "best_authorized_overlap": 0,
            "best_denied_metadata_overlap": best_denied,
            "evidence_quality": _evidence_quality(
                [],
                best_authorized_overlap=0,
                best_denied_metadata_overlap=best_denied,
                hard_refusal=True,
                status="no_authorized_sources",
            ),
        }

    best_authorized = max(_source_overlap(query, source) for source in authorized_sources)
    best_denied = _best_denied_overlap(query, excluded_sources)
    denied_match = best_denied >= 2 and best_authorized < max(4, best_denied + 1)
    if denied_match:
        return {
            "answerable": False,
            "reason": "denied_source_matches_query",
            "best_authorized_overlap": best_authorized,
            "best_denied_metadata_overlap": best_denied,
            "evidence_quality": _evidence_quality(
                authorized_sources,
                best_authorized_overlap=best_authorized,
                best_denied_metadata_overlap=best_denied,
                hard_refusal=True,
                status="denied_source_matches_query",
            ),
        }

    unsupported_specificity = _unsupported_specificity(query, authorized_sources)
    if unsupported_specificity["evidence_gap"]:
        return {
            "answerable": False,
            "reason": "insufficient_authorized_evidence",
            "best_authorized_overlap": best_authorized,
            "best_denied_metadata_overlap": best_denied,
            "unsupported_specificity": unsupported_specificity,
            "evidence_quality": _evidence_quality(
                authorized_sources,
                best_authorized_overlap=best_authorized,
                best_denied_metadata_overlap=best_denied,
                hard_refusal=False,
                status="insufficient_authorized_evidence",
            ),
        }

    return {
        "answerable": True,
        "reason": "authorized_sources_available",
        "best_authorized_overlap": best_authorized,
        "best_denied_metadata_overlap": best_denied,
        "evidence_quality": _evidence_quality(
            authorized_sources,
            best_authorized_overlap=best_authorized,
            best_denied_metadata_overlap=best_denied,
            hard_refusal=False,
            status="authorized_sources_available",
        ),
    }


def _evidence_quality(
    authorized_sources: list[dict[str, Any]],
    *,
    best_authorized_overlap: int,
    best_denied_metadata_overlap: int,
    hard_refusal: bool,
    status: str,
) -> dict[str, Any]:
    """Expose retrieval-quality signals without turning them into fake certainty."""

    top_source = authorized_sources[0] if authorized_sources else {}
    return {
        "status": status,
        "hard_refusal": hard_refusal,
        "selected_source_count": len(authorized_sources),
        "best_authorized_lexical_overlap": best_authorized_overlap,
        "best_denied_metadata_overlap": best_denied_metadata_overlap,
        "top_authorized_vector_score": top_source.get("vector_score"),
        "top_authorized_rerank_score": top_source.get("rerank_score"),
        "threshold_policy": _threshold_policy(status, hard_refusal),
    }


def _threshold_policy(status: str, hard_refusal: bool) -> str:
    if status in {"no_authorized_sources", "denied_source_matches_query"} or hard_refusal:
        return "zero_doc_refusal"
    if status == "insufficient_authorized_evidence":
        return "model_grounded_abstention"
    return "audit_only_until_calibrated"


def _policy_decision(answerability: dict[str, Any]) -> str:
    reason = str(answerability.get("reason", "") or "")
    if reason in {"no_authorized_sources", "denied_source_matches_query"}:
        return "refuse"
    return "allow"


def _unsupported_specificity(query: str, authorized_sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect unsupported highly specific facts for audit and evaluation.

    This is intentionally conservative. It flags dated or scheduled claims where
    authorized evidence does not contain the specificity anchors. Authorized
    pages still flow to Command A so the final result is model-grounded
    abstention rather than a pre-generation hard stop.
    """

    query_tokens = _tokens(query)
    support_text = _support_text(authorized_sources)
    support_tokens = _tokens(support_text)
    year_terms = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", query)))
    missing_years = [year for year in year_terms if year not in support_text]
    scheduled_terms = _scheduled_fact_terms(query_tokens)
    missing_scheduled_terms = sorted(term for term in scheduled_terms if term not in support_tokens)
    evidence_gap = bool(missing_years) and bool(missing_scheduled_terms)
    return {
        "evidence_gap": evidence_gap,
        "hard_refusal": False,
        "year_terms": year_terms,
        "missing_year_terms": missing_years,
        "scheduled_fact_terms": sorted(scheduled_terms),
        "missing_scheduled_fact_terms": missing_scheduled_terms,
        "policy": "send_authorized_evidence_for_model_grounded_abstention",
    }


def _scheduled_fact_terms(query_tokens: set[str]) -> set[str]:
    schedule_terms = {
        "basing",
        "deadline",
        "milestone",
        "schedule",
        "scheduled",
        "timeline",
        "timetable",
    }
    return query_tokens.intersection(schedule_terms)


def _support_text(sources: list[dict[str, Any]]) -> str:
    fields = [
        "doc_id",
        "title",
        "section",
        "status",
        "version",
        "effective_date",
        "text",
    ]
    return " ".join(
        str(source.get(field, ""))
        for source in sources
        for field in fields
    ).lower()


def _best_denied_overlap(query: str, excluded_sources: list[dict[str, Any]]) -> int:
    if not excluded_sources:
        return 0
    return max(_metadata_overlap(query, source) for source in excluded_sources)


def _metadata_overlap(query: str, source: dict[str, Any]) -> int:
    return _overlap(
        query,
        " ".join(
            [
                str(source.get("doc_id", "")),
                str(source.get("title", "")),
                str(source.get("doc_family", "")),
            ]
        ),
    )


def _source_overlap(query: str, source: dict[str, Any]) -> int:
    return _overlap(
        query,
        " ".join(
            [
                str(source.get("doc_id", "")),
                str(source.get("title", "")),
                str(source.get("section", "")),
                str(source.get("text", ""))[:1800],
            ]
        ),
    )


def _overlap(query: str, text: str) -> int:
    return len(_tokens(query).intersection(_tokens(text)))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace("-", " "))
        if len(token) >= 4 and token not in _ANSWERABILITY_STOPWORDS
    }


def _rerank(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    scores = cohere_gateway.rerank(query, [_rerank_document(candidate) for candidate in candidates])
    ranked: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        copy = dict(candidate)
        copy["rerank_score"] = round(float(score), 4)
        ranked.append(copy)
    return sorted(ranked, key=lambda item: item["rerank_score"] or 0.0, reverse=True)


def _select_sources(
    ranked: list[dict[str, Any]],
    top_k: int,
    *,
    language: str = "any",
) -> list[dict[str, Any]]:
    """Return high-ranked pages, preserving close English/French coverage."""

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    for source in ranked:
        if len(selected) >= top_k:
            break
        key = _source_selection_key(source)
        if key in selected_keys:
            continue
        selected.append(source)
        selected_keys.add(key)

    if _normalize_language_filter(language) != "any":
        return selected
    return _rebalance_multilingual_sources(selected, ranked, top_k)


def _rebalance_multilingual_sources(
    selected: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k < 2 or len(selected) < 2:
        return selected
    available_languages = {language for source in ranked if (language := _source_language(source)) in {"en", "fr"}}
    if not {"en", "fr"}.issubset(available_languages):
        return selected
    selected_languages = {language for source in selected if (language := _source_language(source)) in {"en", "fr"}}
    missing_languages = [language for language in ("en", "fr") if language not in selected_languages]
    if not missing_languages:
        return selected

    selected_by_key = {_source_selection_key(source): source for source in selected}
    top_score = _source_rank_score(ranked[0]) if ranked else 0.0
    if top_score <= 0.0:
        return selected
    score_floor = top_score - MULTILINGUAL_RERANK_MARGIN if top_score >= 0.5 else top_score * 0.8

    for language in missing_languages:
        candidate = next(
            (
                source
                for source in ranked
                if _source_language(source) == language
                and _source_selection_key(source) not in selected_by_key
                and _source_rank_score(source) >= score_floor
            ),
            None,
        )
        if candidate is None:
            continue
        replacement_key = _replacement_source_key(selected_by_key.values())
        if replacement_key is None:
            continue
        del selected_by_key[replacement_key]
        selected_by_key[_source_selection_key(candidate)] = candidate

    return [source for source in ranked if _source_selection_key(source) in selected_by_key][:top_k]


def _replacement_source_key(selected: Any) -> str | None:
    counts: dict[str, int] = {}
    for source in selected:
        language = _source_language(source)
        counts[language] = counts.get(language, 0) + 1
    replaceable = [source for source in selected if counts.get(_source_language(source), 0) > 1]
    if not replaceable:
        return None
    return _source_selection_key(min(replaceable, key=_source_rank_score))


def _source_selection_key(source: dict[str, Any]) -> str:
    return str(source.get("parent_page_id") or source.get("chunk_id") or source.get("doc_id", ""))


def _source_language(source: dict[str, Any]) -> str:
    return str(source.get("language", "") or "").strip().lower()


def _source_rank_score(source: dict[str, Any]) -> float:
    try:
        return float(source.get("rerank_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rerank_document(candidate: dict[str, Any]) -> str:
    """Format a page as ordered YAML for Cohere Rerank structured-data support."""

    record = {
        "title": str(candidate.get("title", "")),
        "doc_id": str(candidate.get("doc_id", "")),
        "page": candidate.get("page", ""),
        "section": str(candidate.get("section", "")),
        "language": str(candidate.get("language", "")),
        "status": str(candidate.get("status", "")),
        "version": str(candidate.get("version", "")),
        "effective_date": str(candidate.get("effective_date", "")),
        "access_level": str(candidate.get("access_level", "")),
        "source_format": str(candidate.get("source_format", "")),
        "normalized_format": str(candidate.get("normalized_format", "")),
        "retrieval_chunk_id": str(candidate.get("retrieval_chunk_id", "")),
        "retrieval_modes": ", ".join(str(mode) for mode in candidate.get("retrieval_modes", []) or []),
        "content": str(candidate.get("retrieval_chunk_text") or candidate.get("text", "")),
    }
    return yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=4096)
