from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from pathlib import Path

from sqlalchemy import delete, text
from sqlmodel import Session, func, select

from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.db import engine, init_db
from defence_agent.ingestion.parser import parse_document
from defence_agent.ingestion.synthetic_docs import SPECS, generate_synthetic_documents, spec_by_filename
from defence_agent.models import Chunk, Document
from defence_agent.retrieval.vector_store import vector_store


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "before",
    "from",
    "must",
    "that",
    "this",
    "staff",
    "planning",
    "request",
}


def corpus_has_chunks() -> bool:
    init_db()
    with Session(engine) as session:
        count = session.exec(select(func.count()).select_from(Chunk)).one()
    return count > 0


def reindex_corpus(force_generate: bool = False) -> dict[str, object]:
    init_db()
    paths = generate_synthetic_documents(force=force_generate)
    chunks: list[Chunk] = []
    documents: list[Document] = []

    for path in paths:
        spec = spec_by_filename(path.name)
        document_id = _stable_id(spec.filename)
        documents.append(
            Document(
                id=document_id,
                title=spec.title,
                filename=spec.filename,
                doc_type=spec.doc_type,
                classification=spec.classification,
                allowed_roles_json=json.dumps(list(spec.allowed_roles)),
                version=spec.version,
                effective_date=spec.effective_date,
            )
        )
        parsed_blocks = parse_document(path)
        for index, block in enumerate(parsed_blocks):
            text_parts = [block.text]
            if block.table_markdown:
                text_parts.append(block.table_markdown)
            full_text = "\n\n".join(part for part in text_parts if part).strip()
            chunk_id = f"{document_id}_{index:03d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=index,
                    title=spec.title,
                    section=block.section,
                    page=block.page,
                    text=full_text,
                    table_markdown=block.table_markdown,
                    summary=_summary(full_text),
                    keywords_json=json.dumps(_keywords(full_text)),
                    classification=spec.classification,
                    allowed_roles_json=json.dumps(list(spec.allowed_roles)),
                    tenant_id="deftech",
                    version=spec.version,
                    effective_date=spec.effective_date,
                    doc_type=spec.doc_type,
                    source_uri=str(path),
                    parser_status=block.parser_status,
                    parser_confidence=block.parser_confidence,
                )
            )

    embeddings = cohere_gateway.embed_texts([chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding_json = json.dumps(embedding)

    with Session(engine) as session:
        session.exec(text("DELETE FROM chunk_fts"))
        session.exec(delete(Chunk))
        session.exec(delete(Document))
        for document in documents:
            session.add(document)
        for chunk in chunks:
            session.add(chunk)
        session.commit()
        for chunk in chunks:
            session.exec(
                text(
                    """
                    INSERT INTO chunk_fts(chunk_id, title, section, text, summary, keywords)
                    VALUES (:chunk_id, :title, :section, :text, :summary, :keywords)
                    """
                ),
                params={
                    "chunk_id": chunk.id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "text": chunk.text,
                    "summary": chunk.summary,
                    "keywords": " ".join(json.loads(chunk.keywords_json)),
                },
            )
        session.commit()

    qdrant_indexed = False
    if chunks and embeddings:
        qdrant_indexed = vector_store.recreate_collection(vector_size=len(embeddings[0]))
        if qdrant_indexed:
            qdrant_indexed = vector_store.upsert_chunks(chunks, embeddings)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "qdrant_indexed": qdrant_indexed,
        "generated_files": [str(path) for path in paths],
        "expected_documents": [spec.filename for spec in SPECS],
    }


def document_registry_status() -> dict[str, object]:
    init_db()
    with Session(engine) as session:
        documents = session.exec(select(Document).order_by(Document.title)).all()
        chunks = session.exec(select(Chunk)).all()
    chunks_by_document: dict[str, int] = {}
    classification_counts: dict[str, int] = {}
    parser_status_counts: dict[str, int] = {}
    for chunk in chunks:
        chunks_by_document[chunk.document_id] = chunks_by_document.get(chunk.document_id, 0) + 1
        classification_counts[chunk.classification] = classification_counts.get(chunk.classification, 0) + 1
        parser_status_counts[chunk.parser_status] = parser_status_counts.get(chunk.parser_status, 0) + 1
    return {
        "documents": [
            {
                "id": document.id,
                "title": document.title,
                "filename": document.filename,
                "doc_type": document.doc_type,
                "classification": document.classification,
                "allowed_roles": json.loads(document.allowed_roles_json),
                "version": document.version,
                "effective_date": document.effective_date,
                "parser_status": document.parser_status,
                "parser_confidence": document.parser_confidence,
                "chunk_count": chunks_by_document.get(document.id, 0),
            }
            for document in documents
        ],
        "chunk_count": len(chunks),
        "classification_distribution": classification_counts,
        "parser_status_distribution": parser_status_counts,
    }


def _stable_id(value: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, value).hex


def _summary(text_value: str) -> str:
    first = re.split(r"(?<=[.!?])\s+", text_value.strip())[0]
    return first[:260]


def _keywords(text_value: str) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9_]+", text_value.lower())
        if len(token) > 3 and token not in STOPWORDS
    ]
    return [token for token, _ in Counter(tokens).most_common(8)]
