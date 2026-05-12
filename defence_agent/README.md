# Defence Agent

Defence Agent is the runnable ADK-first backend for the Cohere Solutions
Architect demo. It helps planning staff retrieve authorized doctrine pages and
produce cited answers from a mixed English/French normalized page corpus.

Use `../source-materials/presentation-interview-instructions.md` as the source
of truth for scope: live demo, technical walkthrough, staff efficiency,
security, accuracy, and traceability.

## What It Shows

- Agentic RAG: Google ADK orchestrates one `search_documents` tool and can call
  it multiple times for multi-part questions.
- Cohere retrieval stack: Embed v4 embeds rendered page artifacts, Chroma stores page
  vectors and metadata, and Rerank v4 orders authorized evidence.
- Cited final answers: direct Cohere Chat receives only authorized documents and
  returns native document-RAG citation spans.
- Security control demonstration: persona clearance controls which pages can
  reach the final answer model.
- Traceability: `answer_audit` records filters, source IDs, excluded source
  metadata, rerank scores, Cohere document IDs, and citation spans.

## Main Files

- `agent.py`: ADK `root_agent`.
- `tools.py`: the single model-facing `search_documents` tool.
- `prompts.py`: retrieval-agent instruction.
- `session.py`: ADK runner, session storage, follow-up handling, and audit
  assembly.
- `grounding.py`: direct Cohere final-answer call and citation extraction.
- `retrieval/`: normalized PDF page loading, Embed v4 page embeddings, Chroma, and Rerank.
- `auth/`: fixed demo personas and access policy.
- `data/corpus/`: active normalized page corpus and provenance.
- `data/evals/`: demo query registry and access-control/bilingual eval specs.

## Run

```bash
cd /Users/hananather/Desktop/Cohere
make run
```

Then open the primary Streamlit demo UI at `http://127.0.0.1:8501`.

For technical inspection only, run:

```bash
make run-adk
```

Then open `http://127.0.0.1:8502` and choose `defence_agent` in the ADK Dev UI.
ADK startup can run index checks and can trigger paid Cohere Embed calls when
the local index is stale.

The ADK Dev UI default persona is `clearance_unclassified`. Override it with:

```bash
DEFTECH_ADK_PERSONA=clearance_secret ./defence_agent/scripts/run_adk_agent.sh
```

## CLI Demo

```bash
python defence_agent/scripts/run_agent_session.py \
  "For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?" \
  --persona clearance_secret \
  --show-audit
```

Use `--follow-up` to rehearse source/access follow-ups:

```bash
python defence_agent/scripts/run_agent_session.py \
  "For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?" \
  --follow-up "Which source page supports that, and what access level was required?" \
  --persona clearance_secret \
  --show-audit
```

## Rebuild The Index

```bash
python defence_agent/scripts/build_chroma_index.py
```

The active index build uses live Cohere Embed v4 page embeddings. There is no
mock embedding path in the active prototype.

The Chroma index is a local artifact under `defence_agent/data/chroma/`. It is
ignored by git. Re-running the build without `--force` reuses pages whose
manifest, page-image hash, page-text hash, embedding model, and embedding
dimension have not changed.

The default page embedding batch size is one page. This keeps each live Cohere
call small and lets the builder checkpoint progress page by page.

Index builds take an exclusive local lock at `defence_agent/data/index_build.lock`
so parallel agents or terminals do not duplicate live Cohere embedding calls.

Use `--force` only for an intentional full rebuild:

```bash
python defence_agent/scripts/build_chroma_index.py --force
```

## Verify

Fast checks skip live Cohere/index tests and expensive local PDF-render checks by default:

```bash
python -m pytest defence_agent/tests/
```

Run the full offline suite after corpus or source-preview changes:

```bash
python -m pytest defence_agent/tests/ --run-slow-offline
```

Run live checks deliberately when validating the demo path:

```bash
python -m pytest defence_agent/tests/ --run-live -m live
python defence_agent/scripts/run_demo_query_registry.py --no-fail
```

`make test` runs the fast local suite. `make verify` runs the full offline
suite. `make verify-live` runs the index check, live-marked tests, and demo
registry.

## Active Docs

- `docs/architecture.md`: runtime, ingestion, retrieval, and bilingual behavior.
- `docs/security.md`: persona-aware retrieval and production mapping.
- `docs/observability.md`: ADK runtime traces versus `answer_audit`.
- `docs/evaluation.md`: live evaluation checks and citation review expectations.
- `docs/demo_script.md`: presentation flow and rehearsal commands.
- `docs/cohere_capability_matrix.md`: Cohere source inventory and Q&A
  positioning.
- `docs/adr/`: current architecture decisions.

## Important Boundaries

- Native `.docx` parsing is not implemented. One DOCX-origin source is represented
  through the publisher's official PDF pair, then indexed through the same page
  evidence pipeline.
- The retrieval layer is format-agnostic after verified normalization. Production
  native DOCX ingestion should use Compass or a manifest-compatible parser and
  normalization adapter.
- Synthetic `secret` and `top_secret` labels are illustrative demo tiers only.
- Citation coverage checks whether claim-like answer sentences received
  citations. Citation precision still needs transcript review.
- Excluded source text is not returned from `search_documents` or included in
  `answer_audit`.
- Retired legacy implementations are archived outside the main repo at
  `/Users/hananather/Desktop/Cohere-legacy-archive`.
