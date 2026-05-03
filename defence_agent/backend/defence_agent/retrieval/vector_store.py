from __future__ import annotations

import json
import math
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlmodel import Session, select

from defence_agent.auth.context import AuthContext
from defence_agent.auth.policy import policy_engine
from defence_agent.config import get_settings
from defence_agent.db import engine
from defence_agent.models import Chunk
from defence_agent.resilience import qdrant_breaker


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: QdrantClient | None = None
        self._ensure_client()

    def _ensure_client(self) -> bool:
        if self.client:
            return True
        try:
            candidate = QdrantClient(url=self.settings.qdrant_url, timeout=4)
            candidate.get_collections()
            self.client = candidate
            return True
        except Exception:
            self.client = None
            return False

    def recreate_collection(self, vector_size: int) -> bool:
        if not self._ensure_client():
            return False
        try:
            qdrant_breaker.call(
                lambda: self.client.recreate_collection(
                    collection_name=self.settings.qdrant_collection,
                    vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
                )
            )
            return True
        except Exception:
            self.client = None
            return False

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> bool:
        if not self._ensure_client() or not chunks or not vectors:
            return False
        points: list[qmodels.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                qmodels.PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "section": chunk.section,
                        "classification": chunk.classification,
                        "allowed_roles": json.loads(chunk.allowed_roles_json),
                        "tenant_id": chunk.tenant_id,
                        "version": chunk.version,
                        "doc_type": chunk.doc_type,
                    },
                )
            )
        try:
            qdrant_breaker.call(
                lambda: self.client.upsert(
                    collection_name=self.settings.qdrant_collection,
                    points=points,
                    wait=True,
                )
            )
            return True
        except Exception:
            self.client = None
            return False

    def search(self, query_vector: list[float], auth: AuthContext, limit: int = 20) -> tuple[list[tuple[str, float]], str]:
        self._ensure_client()
        if self.client:
            try:
                return self._search_qdrant(query_vector, auth, limit), "qdrant"
            except Exception:
                self.client = None
        return self._search_sqlite(query_vector, auth, limit), "sqlite_vector_fallback"

    def _search_qdrant(self, query_vector: list[float], auth: AuthContext, limit: int) -> list[tuple[str, float]]:
        acl = policy_engine.acl_filter(auth)
        qfilter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=acl.tenant_id)),
                qmodels.FieldCondition(key="classification", match=qmodels.MatchAny(any=list(acl.allowed_classifications))),
                qmodels.FieldCondition(key="allowed_roles", match=qmodels.MatchAny(any=[auth.role, "all", *[f"group:{g}" for g in auth.groups]])),
            ]
        )
        results = qdrant_breaker.call(
            lambda: self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vector,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
            )
        )
        return [(str(item.payload.get("chunk_id")), float(item.score)) for item in results]

    def _search_sqlite(self, query_vector: list[float], auth: AuthContext, limit: int) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        with Session(engine) as session:
            chunks = session.exec(select(Chunk)).all()
        for chunk in chunks:
            if not chunk.embedding_json or not policy_engine.can_access_chunk(auth, chunk):
                continue
            vector = json.loads(chunk.embedding_json)
            scored.append((chunk.id, cosine_similarity(query_vector, vector)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


vector_store = VectorStore()
