# Defence Agent Demo Guide

## Product Framing

DefTech Doctrine Intelligence Assistant is a secure, traceable, evidence-grounded assistant that helps authorized planning staff interrogate approved manuals, procedures, and doctrine while preserving citations, access control, and human accountability.

## Architecture Story

For free-form policy questions, the assistant uses retrieval and reranking. For multi-document workflows, it uses agentic retrieval. For structured tables, it does not ask the model to pretend it is a calculator; it gives the task to a locked-down code sandbox and makes the calculation auditable.

## UI Map

Use `Guided Demo` for the live interview story. It has step selectors, expected behavior, speaker notes, run buttons, cited answers, source/exclusion tables, trace summaries, and model/tool/latency summaries.

Use `Persona Compare` when the panel asks about access control. It runs the same query for two personas and shows answer differences, source differences, policy decisions, and trace IDs.

Use `Trace Inspector` for the technical walkthrough. It shows policy, retrieval, tool, model, citation, and eval spans in a timeline table.

Use `Governance / Tool Registry` to explain persona permissions, tool risk classification, and recent policy decisions.

## Demo Sequence

The latest demo selector recommends this sequence:

1. Evidence lookup: `What review steps are required before a planning brief is approved?`
2. Metadata-aware retrieval: `What is the current approved procedure for approving a planning brief? Do not use drafts or old versions.`
3. Cross-source synthesis: `What should I include in a planning brief before it goes for review?`
4. Version comparison: `What changed between the 2024 and 2025 planning-brief review process? Cite both versions.`
5. Structured analysis: `Which planning procedures are overdue for review? Group them by owner and show how many days overdue.`
6. Persona access comparison: `What are the approval steps for a planning brief that includes restricted annexes?` with Alex Chen versus Morgan Singh.
7. Bilingual retrieval: `Quels sont les délais dans la procédure de communications d’urgence?`

Backup security query:

```text
What restricted annex handling steps apply before external distribution?
```

Expected behavior:

- Alex Chen, Planning Analyst: restricted-only answer is refused.
- Morgan Singh, Doctrine Steward: restricted source can be cited.
- Priya Rao, Security Auditor: can inspect policy metadata, not restricted content.

## Why Not Just Simple RAG

Simple retrieval works for Q1. It is not enough for current-approved filtering, cross-reference following, version comparison, permission-sensitive retrieval, and deterministic date math over tables.

## Security Story

- User identity and access level are passed into retrieval.
- Metadata filters enforce status, language, and access before generation.
- Restricted documents are not retrieved for unauthorized users.
- Same query can produce different answers for different personas because source access differs.
- Admin privileges are separated from document-content privileges.
- Code execution receives only authorized temporary table rows.
- The sandbox has no network, no mounted secrets, no source-document writes, and a timeout.
- Every run writes a trace JSON file under `defence_agent/data/runs/`.

## Recovery Plan

- If live Cohere calls are slow, switch to fixture/mock mode and explain that the same traces/evals still prove architecture.
- If the UI state looks stale, click `Reset demo state` in Guided Demo.
- If a source card looks confusing, open `Trace Inspector` and use the trace summary block.
- If a restricted query fails live, use Persona Compare. It is deterministic in fixture mode and clearly shows the policy decision.

## Eval Story

The fixture canonical suite has 24 cases. It checks routing, retrieval source IDs, metadata filters, citation presence, citation validation, permission behavior, refusal behavior, bilingual retrieval, and exact sandbox output.

The broader generated suite has 173 cases. It covers find, answer, summarize, synthesize, compare, verify, refusal, permission, bilingual, structured analysis, scanned/OCR, adversarial, persona-security, and demo-candidate tasks.

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
