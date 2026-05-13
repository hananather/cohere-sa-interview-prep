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
from defence_agent.retrieval.document_pages import DocumentPage, load_document_pages, metadata_for_vector_store
from defence_agent.retrieval.embeddings import embed_page_batches, embed_query, embedding_backend_name, embedding_dimension
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
    collection_metadata = _collection_metadata(pages)

    if force:
        _delete_collection_if_exists(client, collection_name)

    collection = client.get_or_create_collection(collection_name, metadata=collection_metadata)
    existing_by_id = _existing_metadata_by_id(collection)
    page_ids = {page.page_id for page in pages}
    orphan_ids = sorted(set(existing_by_id) - page_ids)
    if orphan_ids:
        collection.delete(ids=orphan_ids)
        existing_by_id = {key: value for key, value in existing_by_id.items() if key not in orphan_ids}

    pages_to_embed = [page for page in pages if _page_needs_embedding(page, existing_by_id.get(page.page_id))]
    if not pages_to_embed:
        _update_collection_metadata(collection, collection_metadata)
        return {
            "collection": collection_name,
            "embedding_backend": embedding_backend_name(),
            "embedding_dimension": embedding_dimension(),
            "indexed_chunks": collection.count(),
            "parsed_pages": len(pages),
            "embedded_pages": 0,
            "orphaned_pages_deleted": len(orphan_ids),
            "skipped": True,
        }

    embedded_pages = 0
    for page in pages_to_embed:
        [(page_batch, embeddings)] = list(embed_page_batches([page]))
        collection.upsert(
            ids=[page.page_id for page in page_batch],
            documents=[page.text for page in page_batch],
            embeddings=embeddings,
            metadatas=[_metadata_for_index(page) for page in page_batch],
        )
        embedded_pages += len(page_batch)
    _update_collection_metadata(collection, collection_metadata)
    return {
        "collection": collection_name,
        "embedding_backend": embedding_backend_name(),
        "embedding_dimension": embedding_dimension(),
        "indexed_chunks": collection.count(),
        "parsed_pages": len(pages),
        "embedded_pages": embedded_pages,
        "orphaned_pages_deleted": len(orphan_ids),
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

    build_index(force=False)
    auth = DEMO_USERS.get(persona_id, DEMO_USERS[DEFAULT_PERSONA_ID])
    collection = _client().get_collection(_collection_name())
    query_embedding = embed_query(query)
    allowed_access = list(policy_engine.acl_filter(auth).allowed_classifications)
    normalized_status = _normalize_status_filter(status_filter)
    normalized_language = _normalize_language_filter(language)
    where = _where_filter(allowed_access, normalized_status, normalized_language)
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
    excluded_sources = _excluded_sources(raw_for_exclusions, auth)

    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        candidates.append(_authorized_source(chunk_id, text, metadata, distance, len(candidates) + 1))

    authorized_sources = _select_sources(_rerank(query, candidates), top_k, language=normalized_language)
    for index, source in enumerate(authorized_sources, start=1):
        source["citation_id"] = f"C{index}"
        source["citation"] = f"[C{index}]"
    answerability = _answerability(query, authorized_sources, excluded_sources)

    return {
        "index": "chroma",
        "collection": _collection_name(),
        "embedding_backend": embedding_backend_name(),
        "rerank_backend": get_settings().cohere_rerank_model,
        "persona_id": auth.user_id,
        "allowed_access": allowed_access,
        "filters_applied": {
            "access_level": allowed_access,
            "status": normalized_status,
            "language": normalized_language,
        },
        "policy_decision": "allow" if answerability["answerable"] else "refuse",
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
    return f"{COLLECTION_PREFIX}_{embedding_dimension()}"


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


def _collection_metadata(pages: list[DocumentPage]) -> dict[str, str | int | float]:
    settings = get_settings()
    return {
        "index_version": INDEX_VERSION,
        "embedding_model": settings.cohere_embed_model,
        "embedding_dimension": settings.cohere_embed_output_dimension,
        "corpus_manifest_sha256": _manifest_sha256(settings.corpus_dir / "manifest.yaml"),
        "page_count": len(pages),
        "page_hash_manifest_sha256": _page_hash_manifest_sha256(pages),
        "updated_at_epoch": time(),
    }


def _metadata_for_index(page: DocumentPage) -> dict[str, str | int | float | bool]:
    metadata = dict(page.metadata)
    metadata.update(
        {
            "page_id": page.page_id,
            "index_version": INDEX_VERSION,
            "embedding_model": get_settings().cohere_embed_model,
            "embedding_dimension": embedding_dimension(),
            "page_text_sha256": _sha256_text(page.text),
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


def _page_needs_embedding(page: DocumentPage, existing: dict[str, Any] | None) -> bool:
    if not existing:
        return True
    return any(
        [
            str(existing.get("index_version", "")) != INDEX_VERSION,
            str(existing.get("embedding_model", "")) != get_settings().cohere_embed_model,
            int(existing.get("embedding_dimension") or 0) != embedding_dimension(),
            str(existing.get("page_image_sha256", "")) != str(page.metadata.get("page_image_sha256", "")),
            str(existing.get("page_text_sha256", "")) != _sha256_text(page.text),
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
    """Decide whether authorized evidence is strong enough to answer.

    This is intentionally metadata/text-light. It prevents the final answer
    layer from using unrelated public pages when the best match is denied.
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
    if unsupported_specificity["hard_refusal"]:
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
                hard_refusal=True,
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
        "threshold_policy": "hard_refusal" if hard_refusal else "audit_only_until_calibrated",
    }


def _unsupported_specificity(query: str, authorized_sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect unsupported highly specific facts before final generation.

    This is intentionally conservative. It only hard-refuses when the user asks
    for a dated or scheduled fact and the authorized evidence does not contain
    those specificity anchors. It avoids treating weak but generic lexical
    overlap as a calibrated relevance score.
    """

    query_tokens = _tokens(query)
    support_text = _support_text(authorized_sources)
    support_tokens = _tokens(support_text)
    year_terms = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", query)))
    missing_years = [year for year in year_terms if year not in support_text]
    scheduled_terms = _scheduled_fact_terms(query_tokens)
    missing_scheduled_terms = sorted(term for term in scheduled_terms if term not in support_tokens)
    hard_refusal = bool(missing_years) and bool(missing_scheduled_terms)
    return {
        "hard_refusal": hard_refusal,
        "year_terms": year_terms,
        "missing_year_terms": missing_years,
        "scheduled_fact_terms": sorted(scheduled_terms),
        "missing_scheduled_fact_terms": missing_scheduled_terms,
        "policy": "hard_refuse_when_dated_scheduled_fact_lacks_authorized_support",
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
        key = str(source.get("chunk_id") or source.get("doc_id", ""))
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
    return str(source.get("chunk_id") or source.get("doc_id", ""))


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
        "content": str(candidate.get("text", "")),
    }
    return yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=4096)
