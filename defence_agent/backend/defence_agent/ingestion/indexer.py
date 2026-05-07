from __future__ import annotations

import json
import re
import csv
from collections import Counter
from pathlib import Path

from sqlalchemy import delete, text
from sqlmodel import Session, func, select

from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.db import engine, init_db
from defence_agent.ingestion.parser import parse_document, parse_markdown_metadata
from defence_agent.ingestion.synthetic_docs import SPECS, generate_synthetic_documents, source_checksum
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

CANONICAL_DOC_IDS = {spec.doc_id for spec in SPECS}

TABLE_DOCUMENT_METADATA: dict[str, dict[str, str]] = {
    "doctrine_review_tracker": {
        "title": "Doctrine Review Tracker",
        "doc_family": "doctrine_review_tracker",
        "owner": "Records Management Office",
    },
    "readiness_review_table": {
        "title": "Readiness Review Table",
        "doc_family": "readiness_review",
        "owner": "Readiness Secretariat",
    },
    "approval_register": {
        "title": "Planning Brief Approval Register",
        "doc_family": "approval_register",
        "owner": "Joint Planning Office",
    },
    "corrective_action_tracker": {
        "title": "Corrective Action Tracker",
        "doc_family": "corrective_action",
        "owner": "Readiness Secretariat",
    },
    "annex_inventory": {
        "title": "Annex Inventory",
        "doc_family": "annex_inventory",
        "owner": "Records and Security Office",
    },
}


def corpus_has_chunks() -> bool:
    init_db()
    with Session(engine) as session:
        count = session.exec(select(func.count()).select_from(Chunk)).one()
        canonical_count = session.exec(select(func.count()).select_from(Document).where(Document.id.in_(CANONICAL_DOC_IDS))).one()
    return count > 0 and canonical_count >= len(CANONICAL_DOC_IDS)


def reindex_corpus(force_generate: bool = False) -> dict[str, object]:
    init_db()
    paths = generate_synthetic_documents(force=force_generate)
    chunks: list[Chunk] = []
    documents: list[Document] = []

    for path in paths:
        if path.suffix.lower() == ".csv":
            table_document, table_chunks = _index_table(path)
            documents.append(table_document)
            chunks.extend(table_chunks)
            continue

        metadata = parse_markdown_metadata(path)
        if not metadata:
            continue
        document_id = str(metadata["doc_id"])
        generated_path = path.parent.parent / str(metadata.get("canonical_source", ""))
        source_uri = str(generated_path if generated_path.exists() else path)
        documents.append(
            Document(
                id=document_id,
                title=str(metadata["title"]),
                filename=Path(source_uri).name,
                doc_type=str(metadata["doc_family"]),
                classification=str(metadata["access_level"]),
                allowed_roles_json=json.dumps(list(metadata["allowed_roles"])),
                version=str(metadata["version"]),
                effective_date=str(metadata["effective_date"]),
                doc_family=str(metadata["doc_family"]),
                status=str(metadata["status"]),
                owner=str(metadata["owner"]),
                review_due=str(metadata["review_due"]),
                language=str(metadata["language"]),
                source_type=str(metadata["source_type"]),
                checksum=source_checksum(path),
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
                    title=str(metadata["title"]),
                    section=block.section,
                    page=block.page,
                    text=full_text,
                    table_markdown=block.table_markdown,
                    summary=_summary(full_text),
                    keywords_json=json.dumps(_keywords(full_text)),
                    classification=str(metadata["access_level"]),
                    allowed_roles_json=json.dumps(list(metadata["allowed_roles"])),
                    tenant_id="deftech",
                    version=str(metadata["version"]),
                    effective_date=str(metadata["effective_date"]),
                    doc_family=str(metadata["doc_family"]),
                    status=str(metadata["status"]),
                    owner=str(metadata["owner"]),
                    review_due=str(metadata["review_due"]),
                    language=str(metadata["language"]),
                    source_type=str(metadata["source_type"]),
                    row_id=block.section_id,
                    doc_type=str(metadata["doc_family"]),
                    source_uri=source_uri,
                    parser_status=block.parser_status,
                    parser_confidence=block.parser_confidence,
                )
            )

    embeddings = cohere_gateway.embed_texts([chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding_json = json.dumps(embedding)

    metadata_issues = _missing_required_metadata(documents, chunks)
    language_distribution = dict(Counter(document.language for document in documents))
    status_distribution = dict(Counter(document.status for document in documents))
    access_level_distribution = dict(Counter(document.classification for document in documents))

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

    report = {
        "documents": len(documents),
        "chunks": len(chunks),
        "qdrant_indexed": qdrant_indexed,
        "generated_files": [str(path) for path in paths],
        "expected_documents": [spec.doc_id for spec in SPECS],
        "missing_required_metadata": metadata_issues,
        "language_distribution": language_distribution,
        "status_distribution": status_distribution,
        "access_level_distribution": access_level_distribution,
    }
    report_path = Path("./defence_agent/data/processed/ingestion_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


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
                "doc_family": document.doc_family,
                "status": document.status,
                "owner": document.owner,
                "review_due": document.review_due,
                "language": document.language,
                "source_type": document.source_type,
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


def _index_table(path: Path) -> tuple[Document, list[Chunk]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    document_id = path.stem
    table_metadata = TABLE_DOCUMENT_METADATA.get(
        document_id,
        {
            "title": path.stem.replace("_", " ").title(),
            "doc_family": document_id,
            "owner": "Records Management Office",
        },
    )
    document = Document(
        id=document_id,
        title=table_metadata["title"],
        filename=path.name,
        doc_type="table",
        classification="public_internal",
        allowed_roles_json=json.dumps(["planning_analyst", "planning_lead", "auditor", "admin"]),
        version="2026-05-06",
        effective_date="2026-05-06",
        doc_family=table_metadata["doc_family"],
        status="approved",
        owner=table_metadata["owner"],
        review_due="2026-12-31",
        language="en",
        source_type="table",
        checksum=source_checksum(path),
        parser_status="table_extracted",
        parser_confidence=1.0,
    )
    table_markdown = _rows_to_markdown(rows)
    chunks: list[Chunk] = []
    for index, row in enumerate(rows, start=1):
        row_id = f"R{index}"
        if document_id == "doctrine_review_tracker":
            text_value = (
                f"Row {row_id}: {row['doc_id']} is {row['title']} owned by {row['owner']}. "
                f"Status {row['status']}. Effective date {row['effective_date']}. "
                f"Next review due {row['next_review_due']}. Access level {row['access_level']}. Language {row['language']}."
            )
        else:
            row_pairs = " ".join(f"{key}={value}." for key, value in row.items())
            text_value = f"Row {row_id} in {table_metadata['title']}: {row_pairs}"
        row_access_level = str(row.get("access_level", "public_internal"))
        row_language = str(row.get("language", "en"))
        row_status = str(row.get("status", "approved"))
        row_owner = str(row.get("owner", table_metadata["owner"]))
        row_doc_family = str(row.get("doc_family", table_metadata["doc_family"]))
        row_effective_date = str(row.get("effective_date", row.get("review_date", row.get("due_date", "2026-05-06"))))
        row_review_due = str(row.get("next_review_due", row.get("due_date", row.get("review_date", "2026-12-31"))))
        chunks.append(
            Chunk(
                id=f"{document_id}_{row_id}",
                document_id=document_id,
                chunk_index=index - 1,
                title=table_metadata["title"],
                section=f"{table_metadata['title']} Row {row_id}",
                page=1,
                text=text_value,
                table_markdown=table_markdown if index == 1 else None,
                summary=_summary(text_value),
                keywords_json=json.dumps(_keywords(text_value)),
                classification=row_access_level,
                allowed_roles_json=json.dumps(_roles_for_access(row_access_level)),
                tenant_id="deftech",
                version="2026-05-06",
                effective_date=row_effective_date,
                doc_family=row_doc_family,
                status=row_status,
                owner=row_owner,
                review_due=row_review_due,
                language=row_language,
                source_type="table",
                row_id=row_id,
                doc_type="table",
                source_uri=str(path),
                parser_status="table_row_extracted",
                parser_confidence=1.0,
            )
        )
    return document, chunks


def _roles_for_access(access_level: str) -> list[str]:
    if access_level == "restricted":
        return ["planning_lead", "admin"]
    return ["planning_analyst", "planning_lead", "auditor", "admin"]


def _rows_to_markdown(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _missing_required_metadata(documents: list[Document], chunks: list[Chunk]) -> list[str]:
    missing: list[str] = []
    for document in documents:
        for field in ("id", "title", "doc_family", "version", "status", "effective_date", "owner", "review_due", "classification", "language"):
            if not getattr(document, field, None):
                missing.append(f"document:{document.id}:{field}")
    for chunk in chunks:
        if not chunk.text.strip():
            missing.append(f"chunk:{chunk.id}:text")
        for field in ("doc_family", "status", "classification", "language"):
            if not getattr(chunk, field, None):
                missing.append(f"chunk:{chunk.id}:{field}")
    return missing
