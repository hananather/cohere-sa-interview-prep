# Evaluation

## Active Suites

The mixed-corpus evals are lightweight YAML specs under `defence_agent/data/evals/`:

- `bilingual_eval_set.yaml`: EN to EN, FR to FR, EN to FR, and FR to EN retrieval cases.
- `acl_eval_set.yaml`: unclassified, secret, and top secret persona checks.

These files document the demo contract. Unit tests currently enforce the same core behavior through `defence_agent/tests/test_minimal_adk_agent.py`.

## Live Requirement

Use live Cohere for all active evaluation and demo checks:

```bash
python defence_agent/scripts/build_chroma_index.py
```

The active prototype no longer has a mock embedding, rerank, or final-answer
path. Evaluation should exercise real Embed v4, Rerank v4, ADK tool calls, and
Cohere native citations.

## Citation Expectations

- Cite only authorized sources.
- Use Cohere native document-RAG citations in live final-answer mode.
- Resolve every Cohere citation source ID back to a retrieved `doc_id` and page.
- Confirm every citation source ID maps to a source page sent to final
  generation, not merely to a broader retrieval candidate.
- Keep source excerpts in the source language.
- Show `doc_id`, page number, language, and citation marker.
- Require citation span coverage for claim-like answer sentences. The
  final-answer path retries once when uncited claim-like sentences are detected
  and fails closed if they remain.
- Never reveal text from `excluded_sources`.

## Answerability Expectations

- Refuse when the best matching evidence is denied by persona access.
- Refuse when a dated or scheduled claim is not supported by authorized
  evidence, such as an unsupported 2031 basing schedule.
- Send zero documents to final generation for refusal cases.
- Treat low lexical overlap as an audit signal unless the query asks for a
  specific dated or scheduled fact that the evidence does not contain.

## Answer Audit Expectations

- Run demo rehearsals with `--show-audit`.
- Confirm `answer_audit.retrieval.sources_sent_to_answer` contains only authorized evidence.
- Confirm `answer_audit.retrieval.excluded_sources` contains IDs and reasons, never source text.
- Confirm citation source IDs map back to authorized source metadata from
  `answer_audit.retrieval.sources_sent_to_answer`.
- Treat `citation_validation` as citation source resolution plus citation-span
  coverage. It is still not a factual truth score; it verifies that cited spans
  point to authorized source objects and that claim-like answer sentences are
  cited.

## Release Gate

For routine development, run the fast local suite. It skips tests marked
`live` and `slow_offline`, so it does not call Cohere or re-render the full PDF
corpus:

```bash
python -m pytest defence_agent/tests/
```

For a fuller local check after corpus, source-preview, or ingestion changes:

```bash
python -m pytest defence_agent/tests/ --run-slow-offline
```

For presentation readiness, run the live gate deliberately:

```bash
python defence_agent/scripts/build_chroma_index.py
python -m pytest defence_agent/tests/ --run-live -m live
python defence_agent/scripts/run_demo_query_registry.py --no-fail
```

Then run the same query across `clearance_unclassified`, `clearance_secret`, and `clearance_top_secret` to confirm access-filtered evidence changes as expected.

Pytest markers:

- `live`: real Cohere or live Chroma index behavior. Skipped unless
  `--run-live` is passed.
- `streamlit`: Streamlit `AppTest` checks. These use patched or replayed backend
  behavior and stay in the fast suite.
- `slow_offline`: local-only checks that are slower, such as full corpus page
  rendering. Skipped unless `--run-slow-offline` is passed.

For live presentation readiness, also run one bilingual query and one ACL
contrast query with `--show-audit` and save the terminal transcript for
rehearsal.

The Streamlit Eval tab defaults to a presentation readiness bundle that combines
the curated live readiness run with the saved unsupported-evidence refusal run.
That keeps the visible Ask examples and the inspectable proof table aligned.

## Demo QA Checklist

- Confirm the manifest parses 207 pages.
- Confirm French-source retrieval can return `NATO-STRAT-CONCEPT-2022-FR`.
- Confirm the unclassified persona excludes `SYN-FUSION-S-RELEASE-001`.
- Confirm the secret persona retrieves `SYN-FUSION-S-RELEASE-001`.
- Confirm the secret persona excludes `SYN-FUSION-TS-ANNEX-002`.
- Confirm the top-secret persona retrieves `SYN-FUSION-TS-ANNEX-002`.
- Confirm unsupported scheduled claims send zero documents to the final answer
  model and return an insufficient-evidence refusal.
- Confirm public source URLs are visible in `defence_agent/data/corpus/SOURCES.md`.
- Confirm the DOCX-origin source preserves `source_format: docx`,
  `normalized_format: pdf`, and `normalization_method: official_pdf_pair`.
- Confirm native `.docx` parsing is not claimed.
- Confirm citation spans are described as traceability evidence, not proof of
  factual truth by themselves.
