# Defence Agent Demo Guide

## Product Framing

DefTech Doctrine Intelligence Assistant is a secure, traceable, evidence-grounded assistant that helps authorized planning staff interrogate approved manuals, procedures, and doctrine while preserving citations, access control, and human accountability.

## Architecture Story

For free-form policy questions, the assistant uses retrieval and reranking. For multi-document workflows, it uses agentic retrieval. For structured tables, it does not ask the model to pretend it is a calculator; it gives the task to a locked-down code sandbox and makes the calculation auditable.

## Demo Sequence

The latest demo selector recommends this sequence:

1. Evidence lookup: `What review steps are required before a planning brief is approved?`
2. Metadata-aware retrieval: `What is the current approved procedure for approving a planning brief? Do not use drafts or old versions.`
3. Cross-source synthesis: `What should I include in a planning brief before it goes for review?`
4. Version comparison: `What changed between the 2024 and 2025 planning-brief review process? Cite both versions.`
5. Structured analysis: `Which planning procedures are overdue for review? Group them by owner and show how many days overdue.`
6. Permission-sensitive access: run `What restricted annex handling steps apply before external distribution?` as `planning_analyst`, then `As a restricted user, what high-level rules apply to restricted annexes?`
7. Bilingual retrieval: `Quels sont les délais dans la procédure de communications d’urgence?`

## Why Not Just Simple RAG

Simple retrieval works for Q1. It is not enough for current-approved filtering, cross-reference following, version comparison, permission-sensitive retrieval, and deterministic date math over tables.

## Security Story

- User identity and access level are passed into retrieval.
- Metadata filters enforce status, language, and access before generation.
- Restricted documents are not retrieved for unauthorized users.
- Code execution receives only authorized temporary table rows.
- The sandbox has no network, no mounted secrets, no source-document writes, and a timeout.
- Every run writes a trace JSON file under `defence_agent/data/runs/`.

## Eval Story

The fixture canonical suite has 24 cases. It checks routing, retrieval source IDs, metadata filters, citation presence, citation validation, permission behavior, refusal behavior, bilingual retrieval, and exact sandbox output.

The broader generated suite has 170 cases. It covers find, answer, summarize, synthesize, compare, verify, refusal, permission, bilingual, structured analysis, scanned/OCR, adversarial, and demo-candidate tasks.

Run it:

```bash
make eval
make eval-advanced
make eval-demo
```

Open `reports/eval/latest/demo_selection_report.md` for the selected sequence and `reports/eval/latest/demo_scorecard.md` for backup queries.

Run a smoke test against the local API:

```bash
python defence_agent/scripts/smoke_demo.py
```

## Troubleshooting

- If old documents appear, reindex with `USE_MOCK_COHERE=true python -c "from defence_agent.ingestion.indexer import reindex_corpus; print(reindex_corpus(force_generate=True))"`.
- If Qdrant is not running, the trace will show SQLite vector fallback. This is expected for the local demo.
- If real Cohere mode fails, switch to `USE_MOCK_COHERE=true` for the interview demo and keep the live key in local `.env` only.
