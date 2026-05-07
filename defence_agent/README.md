# Defence Agent

Production-shaped interview prototype for a fictional DefTech Doctrine Intelligence Assistant.

## What It Shows

- Cohere `ClientV2` Embed, Rerank, Chat, and Chat streaming.
- FastAPI backend with auth, policy, tools, traces, metrics, and evals.
- Streamlit frontend with Ask, Trace, Evaluation, Security, and Corpus tabs.
- Qdrant vector retrieval with SQLite fallback.
- SQLite metadata, FTS lexical search, traces, evals, feedback, and sandbox logs.
- Identity-aware retrieval. Unauthorized chunks never enter model context.
- Sandboxed Python table analysis.
- Synthetic PDF/DOCX/Markdown corpus plus a doctrine review tracker table.
- Golden eval suite with 24 cases across lookup, metadata filtering, multi-hop, comparison, structured analysis, permission, bilingual, and refusal behavior.

## Fast Start

```bash
python -m pip install -e ".[dev]"
USE_MOCK_COHERE=true python -c "from defence_agent.ingestion.indexer import reindex_corpus; print(reindex_corpus(force_generate=True))"
defence_agent/scripts/run_demo.sh
```

- Backend: `http://127.0.0.1:8000`
- Streamlit: `http://127.0.0.1:8501`
- Metrics: `http://127.0.0.1:8000/metrics`

Stop local processes:

```bash
defence_agent/scripts/stop_demo.sh
```

Run with the real Cohere key stored in `.env`:

```bash
defence_agent/scripts/run_demo_real.sh
```

## Cohere Modes

- Mock mode: `USE_MOCK_COHERE=true`. No Cohere key required.
- Real mode: set `COHERE_API_KEY` in local `.env` and run `defence_agent/scripts/run_demo_real.sh`.
- `.env` is ignored by git and should be `chmod 600 .env`.

The backend uses Cohere v2 endpoints through `cohere.ClientV2`.

Default model settings:

- Chat: `command-a-03-2025`
- Embed: `embed-v4.0`
- Rerank: `rerank-v4.0-pro`

## Generate And Ingest The Demo Corpus

```bash
USE_MOCK_COHERE=true python -c "from defence_agent.ingestion.indexer import reindex_corpus; print(reindex_corpus(force_generate=True))"
```

This creates:

- canonical Markdown with YAML frontmatter under `defence_agent/data/synthetic_corpus/source_markdown/`
- PDF/DOCX demo artifacts under `defence_agent/data/synthetic_corpus/generated/`
- `doctrine_review_tracker.csv` under `defence_agent/data/synthetic_corpus/tables/`
- `defence_agent/data/processed/ingestion_report.json`

## Docker Compose

```bash
docker compose up --build
```

Compose runs Qdrant, FastAPI, and Streamlit. If Qdrant is not ready, the app degrades to SQLite vector fallback and records the degradation in traces.

## Demo Personas

- `planning_analyst`: protected clearance, no restricted annex access.
- `planning_lead`: restricted clearance, can access Restricted Annex B.
- `auditor`: protected audit user.
- `admin`: restricted admin user.

Use `X-Demo-User` on API calls.

## API Endpoints

- `GET /health`
- `GET /healthz`
- `POST /v1/agent/query`
- `POST /v1/agent/stream`
- `GET /v1/traces/{trace_id}`
- `POST /v1/evals/run`
- `GET /v1/evals/results`
- `POST /v1/feedback`
- `POST /v1/admin/reindex`
- `GET /v1/admin/corpus`
- `GET /metrics`

`POST /v1/ask`, `POST /v1/ingestion/reindex`, and `GET /v1/corpus` are also kept as convenience aliases.

## Checks

```bash
make test
make eval
python defence_agent/scripts/smoke_demo.py
```

The fixture-mode acceptance gate is:

- 100% route accuracy on canonical demo queries.
- 100% permission correctness.
- 100% citation validation.
- Exact structured-analysis result for overdue review counts as of `2026-05-06`.

## Interview Docs

- [Demo guide](README_DEMO.md)
- [Architecture](ARCHITECTURE.md)
- [Security](SECURITY.md)
- [Evaluation](EVALS.md)
- [Demo QA tracker](docs/demo_qa_todo.md)
- [Observability](docs/observability.md)
- [ADRs](docs/adr)
