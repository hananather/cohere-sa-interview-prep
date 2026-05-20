"""Local BM25 lexical retrieval over normalized retrieval chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from defence_agent.auth.policy import policy_engine
from defence_agent.retrieval.chunks import RetrievalChunk


@dataclass(frozen=True)
class Bm25Hit:
    chunk: RetrievalChunk
    score: float
    rank: int


def bm25_search(
    *,
    query: str,
    chunks: list[RetrievalChunk],
    auth: Any,
    top_n: int,
    status_filter: str,
    language: str,
) -> list[Bm25Hit]:
    """Return authorized lexical hits without exposing filtered chunk text."""

    filtered = [
        chunk
        for chunk in chunks
        if _chunk_allowed(chunk, auth=auth, status_filter=status_filter, language=language)
    ]
    if not filtered:
        return []
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    tokenized = [_tokens(_chunk_index_text(chunk)) for chunk in filtered]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        (
            Bm25Hit(chunk=chunk, score=_score_with_overlap_floor(query_tokens, chunk, float(score)), rank=index + 1)
            for index, (chunk, score) in enumerate(zip(filtered, scores))
            if _score_with_overlap_floor(query_tokens, chunk, float(score)) > 0.0
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return [Bm25Hit(chunk=hit.chunk, score=hit.score, rank=index + 1) for index, hit in enumerate(ranked[:top_n])]


def _chunk_allowed(
    chunk: RetrievalChunk,
    *,
    auth: Any,
    status_filter: str,
    language: str,
) -> bool:
    metadata = chunk.metadata
    if not policy_engine.can_access_chunk(auth, _ChunkForPolicy(metadata)):
        return False
    if status_filter != "any" and str(metadata.get("status", "")) != status_filter:
        return False
    if language != "any" and str(metadata.get("language", "")) != language:
        return False
    return True


def _chunk_index_text(chunk: RetrievalChunk) -> str:
    metadata = chunk.metadata
    return " ".join(
        [
            str(metadata.get("doc_id", "")),
            str(metadata.get("title", "")),
            str(metadata.get("section", "")),
            str(metadata.get("doc_family", "")),
            chunk.text,
        ]
    )


def _score_with_overlap_floor(query_tokens: list[str], chunk: RetrievalChunk, score: float) -> float:
    if score > 0.0:
        return score
    overlap = len(set(query_tokens).intersection(_tokens(_chunk_index_text(chunk))))
    return float(overlap) if overlap else 0.0


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace("-", " "))
        if len(token) >= 2
    ]


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
        import json

        return json.dumps(raw)
