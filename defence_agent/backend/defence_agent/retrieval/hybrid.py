from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlmodel import Session, select

from defence_agent.auth.context import AuthContext
from defence_agent.auth.policy import policy_engine
from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.db import engine
from defence_agent.models import Chunk, SourceChunk
from defence_agent.observability.tracing import trace_manager
from defence_agent.retrieval.vector_store import vector_store
from defence_agent.safety import sanitize_retrieved_text


@dataclass
class Candidate:
    chunk: Chunk
    lexical_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    chunks: list[SourceChunk]
    trace: dict[str, Any] = field(default_factory=dict)
    degradations: list[str] = field(default_factory=list)


class HybridRetriever:
    def search(
        self,
        query: str,
        auth: AuthContext,
        trace_id: str,
        top_k: int = 6,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        filters = filters or {}
        degradations: list[str] = []
        acl = policy_engine.acl_filter(auth)
        trace_manager.add_span(
            trace_id,
            "retrieval_acl_filter_applied",
            {
                "tenant_id": acl.tenant_id,
                "role": acl.role,
                "clearance": acl.clearance,
                "allowed_classifications": acl.allowed_classifications,
                "filters": filters,
            },
        )

        with trace_manager.span(trace_id, "lexical_search_completed", {"query": query}) as span:
            lexical = self._lexical_search(query, auth, limit=top_k * 4, filters=filters)
            span.attributes_json = json.dumps({"candidate_count": len(lexical), "chunk_ids": [item[0] for item in lexical]})

        with trace_manager.span(trace_id, "vector_search_completed", {"query": query}) as span:
            try:
                query_vector = cohere_gateway.embed_query(query)
                vector, backend = vector_store.search(query_vector, auth, limit=top_k * 4)
                if backend != "qdrant":
                    degradations.append("vector_search_used_sqlite_fallback")
                span.attributes_json = json.dumps({"candidate_count": len(vector), "backend": backend, "chunk_ids": [item[0] for item in vector]})
            except Exception as exc:
                vector = []
                degradations.append("vector_search_failed_lexical_fallback")
                span.status = "error"
                span.error = str(exc)

        candidates = self._merge_candidates(lexical, vector, auth, filters=filters)
        if not candidates:
            return RetrievalResult(chunks=[], trace={"candidate_count": 0}, degradations=degradations)

        with trace_manager.span(trace_id, "rerank_completed", {"candidate_count": len(candidates)}) as span:
            try:
                rerank_inputs = [sanitize_retrieved_text(candidate.chunk.text) for candidate in candidates]
                scores = cohere_gateway.rerank(query, rerank_inputs)
                for candidate, score in zip(candidates, scores):
                    candidate.rerank_score = score
                candidates.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else item.hybrid_score, reverse=True)
                span.attributes_json = json.dumps(
                    {
                        "scores": [
                            {"chunk_id": candidate.chunk.id, "rerank_score": candidate.rerank_score}
                            for candidate in candidates
                        ]
                    }
                )
            except Exception as exc:
                degradations.append("rerank_failed_used_hybrid_score")
                candidates.sort(key=lambda item: item.hybrid_score, reverse=True)
                span.status = "error"
                span.error = str(exc)

        final = [self._to_source_chunk(candidate) for candidate in candidates[:top_k]]
        trace_manager.add_span(
            trace_id,
            "context_assembled",
            {
                "final_chunk_ids": [chunk.chunk_id for chunk in final],
                "source_count": len(final),
                "degradations": degradations,
            },
        )
        return RetrievalResult(
            chunks=final,
            trace={
                "candidate_count": len(candidates),
                "final_chunk_ids": [chunk.chunk_id for chunk in final],
            },
            degradations=degradations,
        )

    def _lexical_search(
        self,
        query: str,
        auth: AuthContext,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        filters = filters or {}
        match_query = self._fts_query(query)
        results: list[tuple[str, float]] = []
        with Session(engine) as session:
            try:
                rows = session.exec(
                    text(
                        """
                        SELECT chunk_id, bm25(chunk_fts) AS score
                        FROM chunk_fts
                        WHERE chunk_fts MATCH :query
                        ORDER BY score
                        LIMIT :limit
                        """
                    ),
                    params={"query": match_query, "limit": limit * 3},
                ).all()
                scored = [(str(row[0]), float(-row[1])) for row in rows]
            except Exception:
                like = f"%{query[:80]}%"
                rows = session.exec(
                    text("SELECT id FROM chunk WHERE text LIKE :like OR title LIKE :like OR section LIKE :like LIMIT :limit"),
                    params={"like": like, "limit": limit * 3},
                ).all()
                scored = [(str(row[0]), 0.1) for row in rows]
            for chunk_id, score in scored:
                chunk = session.get(Chunk, chunk_id)
                if chunk and policy_engine.can_access_chunk(auth, chunk) and self._matches_filters(chunk, filters):
                    results.append((chunk_id, max(score, 0.01)))
                if len(results) >= limit:
                    break
        return results

    def _merge_candidates(
        self,
        lexical: list[tuple[str, float]],
        vector: list[tuple[str, float]],
        auth: AuthContext,
        filters: dict[str, Any],
    ) -> list[Candidate]:
        by_id: dict[str, Candidate] = {}
        max_lexical = max([score for _, score in lexical], default=1.0)
        with Session(engine) as session:
            for chunk_id, score in lexical:
                chunk = session.get(Chunk, chunk_id)
                if not chunk or not policy_engine.can_access_chunk(auth, chunk) or not self._matches_filters(chunk, filters):
                    continue
                by_id[chunk_id] = Candidate(chunk=chunk, lexical_score=score / max_lexical)
            for chunk_id, score in vector:
                chunk = session.get(Chunk, chunk_id)
                if not chunk or not policy_engine.can_access_chunk(auth, chunk) or not self._matches_filters(chunk, filters):
                    continue
                candidate = by_id.get(chunk_id) or Candidate(chunk=chunk)
                candidate.vector_score = max(score, 0.0)
                by_id[chunk_id] = candidate
        for candidate in by_id.values():
            candidate.hybrid_score = (0.45 * candidate.lexical_score) + (0.55 * candidate.vector_score)
        return sorted(by_id.values(), key=lambda item: item.hybrid_score, reverse=True)

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        version = filters.get("version")
        if version and str(chunk.version) != str(version):
            return False
        doc_type = filters.get("doc_type")
        if doc_type and chunk.doc_type != doc_type:
            return False
        doc_types = filters.get("doc_types")
        if doc_types and chunk.doc_type not in set(doc_types):
            return False
        exclude_doc_types = filters.get("exclude_doc_types")
        if exclude_doc_types and chunk.doc_type in set(exclude_doc_types):
            return False
        title_contains = filters.get("title_contains")
        if title_contains and title_contains.lower() not in chunk.title.lower():
            return False
        return True

    def _to_source_chunk(self, candidate: Candidate) -> SourceChunk:
        chunk = candidate.chunk
        return SourceChunk(
            chunk_id=chunk.id,
            title=chunk.title,
            section=chunk.section,
            page=chunk.page,
            text=sanitize_retrieved_text(chunk.text),
            summary=sanitize_retrieved_text(chunk.summary),
            classification=chunk.classification,
            version=chunk.version,
            effective_date=chunk.effective_date,
            doc_type=chunk.doc_type,
            source_uri=chunk.source_uri,
            lexical_score=round(candidate.lexical_score, 4),
            vector_score=round(candidate.vector_score, 4),
            hybrid_score=round(candidate.hybrid_score, 4),
            rerank_score=round(candidate.rerank_score, 4) if candidate.rerank_score is not None else None,
            table_markdown=chunk.table_markdown,
        )

    def _fts_query(self, query: str) -> str:
        tokens = [token for token in re.findall(r"[a-zA-Z0-9_]+", query) if len(token) > 2]
        if not tokens:
            return query
        return " OR ".join(tokens[:12])


hybrid_retriever = HybridRetriever()
