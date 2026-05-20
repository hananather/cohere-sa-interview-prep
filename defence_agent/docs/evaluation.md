# Evaluation

## Active Suites

The mixed-corpus evals are lightweight YAML specs under `defence_agent/data/evals/`:

- `bilingual_eval_set.yaml`: EN to EN, FR to FR, EN to FR, and FR to EN retrieval cases.
- `acl_eval_set.yaml`: unclassified, secret, and top secret persona checks.

These files document the demo contract. Unit tests currently enforce the same core behavior through `defence_agent/tests/test_minimal_adk_agent.py`.

## Layered Offline Harness

Use the offline harness when you want Cohere-aligned evaluation statistics
without spending live Cohere calls:

```bash
python defence_agent/scripts/summarize_model_quality_eval.py
python defence_agent/scripts/summarize_model_quality_eval.py --rerank-uplift
python defence_agent/scripts/summarize_eval_harness.py
python defence_agent/scripts/summarize_eval_harness.py --all-runs
python defence_agent/scripts/summarize_eval_harness.py --all-runs --citation-support
python defence_agent/scripts/export_citation_review_queue.py --source-text
python defence_agent/scripts/build_pilot_eval_bank.py
```

The primary presentation scorecard is now the curated model-quality report:

- `defence_agent/data/evals/model_quality_eval_bank.yaml`
- `defence_agent/data/evals/reports/model_quality_eval_report.md`
- `defence_agent/data/evals/reports/model_quality_eval_report.json`

For a client- or panel-facing trust story, use the defensible eval report
instead of the smaller model-quality debug slice:

- `defence_agent/docs/defensible_eval_set.md`
- `defence_agent/data/evals/pilot_eval_bank.yaml`
- `defence_agent/data/evals/ambiguous_clarification_eval_bank.yaml`
- `defence_agent/data/evals/reports/defensible_eval_report.md`
- `defence_agent/data/evals/reports/defensible_eval_report.json`

Generate it with:

```bash
python defence_agent/scripts/summarize_defensible_eval.py
```

This report intentionally excludes ACL enforcement from model-quality metrics.
ACL is a deterministic metadata boundary: restricted evidence should not enter
model context. Track it as a security invariant and regression test, not as a
model-quality score. Model-quality metrics should focus on the stochastic or
ranking-dependent surfaces: retrieval, reranking, answerability, generation,
citations, and reviewer-agent behavior.

The harness scores saved transcript JSON against
`defence_agent/data/evals/demo_query_registry.yaml` and reports:

- retrieval recall, retrieval precision, and facet recall;
- expected fact recall and forbidden fact absence;
- answerability/refusal accuracy;
- citation recall proxy, citation precision from support labels,
  cited-document recall/precision, and citation source precision proxy;
- optional lexical citation-support precision by reloading corpus page text from
  citation source IDs;
- end-to-end pass rate, case consistency, failure examples, and sample coverage
  against the 60-case target bank.

The curated model-quality report adds live rerank-uplift diagnostics when
called with `--rerank-uplift`:

- BM25 pre-rerank: lexical baseline, no embedding model.
- Vector pre-rerank: embedding-only baseline.
- Hybrid pre-rerank: lexical plus embedding before reranking.
- Hybrid post-rerank: current production-style path after Cohere Rerank.

For presentation, lead with precision@K and recall@K across those stages. MRR
and nDCG remain useful diagnostics in the JSON report, but they are not the
core trust metrics. The uplift is the difference between hybrid post-rerank and
hybrid pre-rerank.

The model-quality report also includes a bilingual citation section. Use it to
answer whether bilingual questions are worse than the rest of the bank. It
compares bilingual and non-bilingual cohorts on retrieval precision/recall,
citation recall, semantic citation precision, and supported or unsupported
French/English citation labels.

For the presentation trust-layer story, read
`defence_agent/docs/trust_layer_eval_story.md`.

Generated baseline reports live under `defence_agent/data/evals/reports/`.
The clean notebook surface is `notebooks/defence_agent_eval_harness.ipynb`.
The balanced 60-case target bank is
`defence_agent/data/evals/pilot_eval_bank.yaml`, built from registry variants
and `pilot_eval_supplemental_cases.yaml`.

The current full pilot run is saved at
`defence_agent/data/transcripts/eval_full_registry_20260514_020915` and scored
in `defence_agent/data/evals/reports/eval_full_registry_report.*`. A
35-pair full-pilot citation label sample is scored in
`eval_full_registry_report_with_pilot_labels.*`.

Treat citation source precision and lexical citation-support precision as
proxies, not as full semantic citation precision. `--citation-support` reloads
the page text and checks whether the cited span has enough lexical overlap with
the cited page. That is deterministic and useful for triage, but translated or
paraphrased evidence can under-score.

For full semantic citation precision, export `(answer span, cited source)` rows
with `export_citation_review_queue.py`, label support decisions in the schema
shown by `citation_precision_labels.example.yaml`, then rerun
`summarize_eval_harness.py --citation-labels <labels.yaml>`.

The current repo includes `citation_precision_pilot_labels.yaml` as a small
reviewed sample. It demonstrates the metric and currently yields a pilot
semantic citation precision score in
`presentation_readiness_eval_report_with_pilot_labels.*`; it is not the full
calibrated citation study.

The full 60-case run also includes
`citation_precision_full_pilot_labels.yaml`, which samples citation/source
support across citation-producing slices and populates citation precision in
`eval_full_registry_report_with_pilot_labels.*`.

## Reviewer Agent Value Harness

Use the critic trust harness as the primary proof for the Reviewer Agent claim:

```bash
python defence_agent/scripts/summarize_critic_trust_eval.py
```

Inputs and outputs:

- `defence_agent/data/evals/critic_citation_trust_eval.yaml`
- `defence_agent/data/evals/reports/critic_trust_eval_report.md`
- `defence_agent/data/evals/reports/critic_trust_eval_report.json`

This harness isolates the reviewer from retrieval and generation. Each case has
synthetic cited evidence, an answer, expected release gate, and gold citation
support labels. That makes precision and recall meaningful:

- `invalid_citation_recall`: of gold-invalid citations, how often the reviewer
  marks them not verified.
- `trusted_citation_precision`: of citations the reviewer marks verified, how
  often the gold label says they are supported.
- `release_gate_accuracy`: whether the reviewer chose release, revise,
  clarification, or human review correctly.
- `unsafe_release_rate`: cases that should not release but did release.
- `traffic_light_accuracy`: whether green, yellow, and red citation trust
  labels match the gold labels.

Use the reviewer challenge harness when the architecture question is whether the
full Research Agent plus Reviewer Agent path earns its latency and cost:

```bash
python defence_agent/scripts/run_reviewed_answer_challenges.py
```

The harness runs hard live cases through the Research Agent plus Reviewer Agent
path and records:

- whether the Research Agent found the expected evidence;
- whether the answer met citation and refusal expectations;
- the Reviewer Agent citation credibility score;
- whether the score confirmed release, added review pressure, caught a failure,
  or missed a failure.

Read the verdict narrowly. The Reviewer Agent adds value when it catches weak
citation support or flags review pressure on high-stakes synthesis. It does not
replace retrieval metrics, citation-label studies, or SME review.

The product contract is in `defence_agent/docs/trust_layer_eval_story.md`:
unsupported citations should be quarantined as reviewer findings, not presented
as trusted final-answer citations.

The current critic trust eval is still synthetic. Treat it as a prototype gold
set, not as a production SME study. Its value is that it proves the measurement
shape: citation-level gold labels, trust score, release gate, terminal human
review behavior, latency, and estimated cost per query.

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
- Use the Reviewer Agent score as citation-support evidence, not as proof of
  factual truth outside the cited pages.
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
- Send zero documents to final generation for access failures or no authorized
  source text.
- For insufficient-evidence cases with authorized pages, send the authorized
  pages to Command A and evaluate whether grounded generation abstains instead
  of inventing unsupported facts.
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

## Agent Value Route Comparison

Use the route comparison harness when the presentation claim is specifically
about how much value the agentic system adds over plain RAG:

```bash
python defence_agent/scripts/summarize_route_comparison_eval.py
```

The current report is written to:

- `defence_agent/data/evals/reports/agent_value_route_comparison_report.md`
- `defence_agent/data/evals/reports/agent_value_route_comparison_report.json`

Interpretation rules:

- Treat Research Agent lift as raw task-success and evidence-recall lift.
- Treat Reviewer Agent lift as release-control and traceability lift.
- Do not claim the Reviewer Agent improves retrieval recall unless the same run
  shows expected documents being recovered.
- Do not call lexical citation support semantic citation precision.
- Check citation granularity before using reviewer citation precision. The
  current saved route run is page-level throughout, but `windowed` retrieval can
  rank child chunks and then promote them back to parent-page evidence for
  generation and review.
- If child chunks are promoted to page-level citations, citation doc recall can
  stay valid while citation precision becomes less strict because the reviewer
  sees a broader page than the retrieved child chunk.
- Treat captured generation cost as partial telemetry because ADK planner,
  reviewer, Embed, and Rerank billing are not fully recorded in the route
  transcripts.

The Streamlit Eval tab defaults to a presentation readiness bundle that combines
the curated live readiness run with the saved unsupported-evidence refusal run.
That keeps the visible Ask examples and the inspectable proof table aligned.

## Demo QA Checklist

- Confirm the manifest parses 217 pages.
- Confirm French-source retrieval can return `NATO-STRAT-CONCEPT-2022-FR`.
- Confirm the unclassified persona excludes `SYN-FUSION-S-RELEASE-001`.
- Confirm the secret persona retrieves `SYN-FUSION-S-RELEASE-001`.
- Confirm the secret persona excludes `SYN-FUSION-TS-ANNEX-002`.
- Confirm the top-secret persona retrieves `SYN-FUSION-TS-ANNEX-002`.
- Confirm unsupported scheduled claims send authorized retrieved pages to the
  final answer model and return an insufficient-evidence refusal.
- Confirm public source URLs are visible in `defence_agent/data/corpus/SOURCES.md`.
- Confirm the scanned-manual source preserves `source_format: scanned_pdf`,
  `normalized_format: pdf`, and `normalization_method:
  digitized_scan_with_ocr_text_layer`.
- Confirm the DOCX-origin source preserves `source_format: docx`,
  `normalized_format: pdf`, and `normalization_method: official_pdf_pair`.
- Confirm native `.docx` parsing is not claimed.
- Confirm citation spans are described as traceability evidence, not proof of
  factual truth by themselves.
