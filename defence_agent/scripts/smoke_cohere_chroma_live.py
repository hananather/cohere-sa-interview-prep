from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.config import get_settings
from defence_agent.retrieval.document_pages import load_document_pages, metadata_for_vector_store


def main() -> int:
    args = _parse_args()
    settings = get_settings()

    pages = [page for page in load_document_pages() if page.doc_id == args.doc_id]
    if not pages:
        print(f"No pages found for doc_id={args.doc_id}")
        return 2
    page = pages[0]

    collection_name = f"deftech_smoke_{args.doc_id.lower().replace('-', '_')}_{settings.cohere_embed_output_dimension}"
    chroma_client = chromadb.PersistentClient(path=str(settings.data_dir / "chroma"))
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(collection_name)

    started = time.perf_counter()
    document_embedding = _embed_page(page)
    collection.add(
        ids=[page.page_id],
        documents=[page.text],
        embeddings=[document_embedding],
        metadatas=[metadata_for_vector_store(page.metadata)],
    )

    query_embedding = cohere_gateway.embed_query(args.query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    retrieved_text = results["documents"][0][0]
    retrieved_metadata = results["metadatas"][0][0]
    retrieved_distance = float(results["distances"][0][0])

    rerank_score = float(cohere_gateway.rerank(args.query, [retrieved_text])[0])

    chunks = cohere_gateway.chat_stream(
        model=settings.cohere_chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DefTech Doctrine Intelligence Assistant. "
                    "Answer only from the provided document. Cite the source as [C1]."
                ),
            },
            {"role": "user", "content": args.query},
        ],
        documents=[
            {
                "id": "C1",
                "data": {
                    "doc_id": retrieved_metadata.get("doc_id"),
                    "title": retrieved_metadata.get("title"),
                    "page": retrieved_metadata.get("page"),
                    "snippet": retrieved_text,
                },
            }
        ],
        temperature=0.1,
        max_tokens=350,
    )
    answer_text = _message_text(chunks)
    citations = _citations(chunks)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_id": args.doc_id,
        "query": args.query,
        "models": {
            "chat": settings.cohere_chat_model,
            "embed": settings.cohere_embed_model,
            "embed_output_dimension": settings.cohere_embed_output_dimension,
            "rerank": settings.cohere_rerank_model,
        },
        "chroma": {
            "path": str(settings.data_dir / "chroma"),
            "collection": collection_name,
            "stored_ids": collection.get()["ids"],
            "query_top_id": results["ids"][0][0],
            "distance": retrieved_distance,
        },
        "rerank": {"score": rerank_score},
        "answer": answer_text,
        "citations": citations,
        "latency_ms": elapsed_ms,
    }

    output_path = settings.data_dir / "adk_smoke" / "latest_cohere_chroma_smoke.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Doc: {args.doc_id} page {page.page_number}")
    print(f"Embedding model: {settings.cohere_embed_model} ({settings.cohere_embed_output_dimension}d)")
    print(f"Rerank model: {settings.cohere_rerank_model}")
    print(f"Chat model: {settings.cohere_chat_model}")
    print(f"Chroma collection: {collection_name}")
    print(f"Stored IDs: {payload['chroma']['stored_ids']}")
    print(f"Top ID: {payload['chroma']['query_top_id']}")
    print(f"Rerank score: {rerank_score:.4f}")
    print(f"Latency ms: {elapsed_ms}")
    print(f"Saved: {output_path}")
    print("Answer preview:")
    print(answer_text[:700])
    return 0


def _embed_page(page: Any) -> list[float]:
    vectors = cohere_gateway.embed_inputs(
        [
            {
                "content": [
                    {"type": "text", "text": f"{page.doc_id} page {page.page_number}"},
                    {"type": "image_url", "image_url": {"url": page.image_data_url}},
                ]
            }
        ],
        input_type="search_document",
    )
    return list(vectors[0])


def _message_text(chunks: list[Any]) -> str:
    text = ""
    for chunk in chunks:
        if getattr(chunk, "type", "") != "content-delta":
            continue
        delta = getattr(chunk, "delta", None)
        message = getattr(delta, "message", None)
        content = getattr(message, "content", None)
        text += str(getattr(content, "text", "") or "")
    return text.strip()


def _citations(chunks: list[Any]) -> list[str]:
    output: list[str] = []
    for chunk in chunks:
        if getattr(chunk, "type", "") != "citation-start":
            continue
        delta = getattr(chunk, "delta", None)
        message = getattr(delta, "message", None)
        citations = getattr(message, "citations", None)
        if isinstance(citations, list):
            output.extend(str(citation) for citation in citations)
        elif citations is not None:
            output.append(str(citations))
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Cohere Embed v4 + Chroma + Rerank v4 smoke test.")
    parser.add_argument("--doc-id", default="CA-AI-STRAT-2024-EN")
    parser.add_argument(
        "--query",
        default="What are the DND CAF AI Strategy lines of effort?",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
