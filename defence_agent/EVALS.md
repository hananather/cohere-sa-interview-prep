# Evaluation

## What Is Evaluated

The demo now has two eval layers:

- Legacy golden suite: `defence_agent/evals/golden_dataset.yaml`.
- Advanced harness: `defence_agent/data/evals/*.yaml`.

The advanced harness grades the pipeline by component instead of producing one vague score.

It checks:

- Corpus readiness.
- Route and tool selection.
- Metadata and access filters.
- Retrieval recall, precision, and mean reciprocal rank.
- Rerank score capture.
- Tool execution.
- Structured table/code execution.
- Answer behavior.
- Citation validation.
- Safety behavior.
- Persona policy behavior.
- Excluded source behavior.
- Trace completeness.
- Tool policy enforcement.
- Trace redaction.
- Latency, trace ID, and token/cost estimate presence.

## Eval Suites

- `canonical_eval_set.yaml`: stable 24-case acceptance gate.
- `generated_eval_set.yaml`: 173 broad coverage cases.
- `heldout_eval_set.yaml`: generated cases reserved for regression review.
- `regression_eval_set.yaml`: canonical cases that should keep passing.
- `adversarial_eval_set.yaml`: prompt-injection, metadata-bypass, and access-bypass cases.
- `demo_candidates.yaml`: candidate live-demo queries.

Persona/security fields are optional per case:

- `persona_id`
- `security_scenario`
- `expected_policy_decision`
- `expected_excluded_sources`
- `expected_visible_sources`
- `expected_hidden_sources`
- `expected_trace_assertions`

## Task Complexity

Each case has:

- `task_type`
- `capability_tags`
- `complexity_level`
- `complexity_score`
- `complexity_dimensions`

Complexity is based on concrete dimensions such as access control, multi-hop retrieval, metadata filters, bilingual retrieval, deterministic calculation, and adversarial wording.

## Commands

Validate schema and source references:

```bash
PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py validate
```

Run canonical fixture eval:

```bash
PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py run --suite canonical --mode fixture
```

Compare retrieval/agent variants:

```bash
PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py compare --suite canonical --limit 24
```

Select reliable demo queries:

```bash
PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py select-demo --mode fixture --runs 3
```

Run everything local:

```bash
make verify
```

## Reports

Reports are written under `reports/eval/latest/`.

Key files:

- `summary.json`
- `summary.md`
- `metrics_by_task_type.csv`
- `metrics_by_complexity.csv`
- `metrics_by_route.csv`
- `metrics_by_tool.csv`
- `retrieval_metrics.csv`
- `citation_metrics.csv`
- `safety_metrics.csv`
- `structured_analysis_metrics.csv`
- `persona_policy_metrics.csv`
- `trace_completeness_metrics.csv`
- `failure_analysis.md`
- `confusion_matrix_route.csv`
- `experiment_comparison.md`
- `demo_scorecard.md`
- `demo_selection_report.md`

## Current Fixture Gate

Latest canonical fixture run:

- Cases: 24.
- Pass rate: 100%.
- Route accuracy: 100%.
- Tool accuracy: 100%.
- Retrieval recall@k: 100%.
- Citation validation: 100%.
- Access-control correctness: 100%.
- Persona-policy pass rate: 100%.
- Tool-policy pass rate: 100%.
- Trace-completeness pass rate: 100%.
- Restricted leakage rate: 0%.
- Structured-analysis exactness: 100%.

Structured-analysis gate uses deterministic `today = 2026-05-06`.

Expected overdue results:

- `PB-SOP-2025`: 52 days overdue.
- `EC-PROC-2025`: 35 days overdue.
- `EC-PROC-2025-FR`: 35 days overdue.
- `LOG-RET-2025`: 96 days overdue.

## Feedback-To-Eval Flywheel

Feedback events are stored in:

```text
defence_agent/data/feedback/feedback_events.jsonl
```

Promote low-rated runs to draft eval cases:

```bash
PYTHONPATH=defence_agent/backend python defence_agent/scripts/promote_feedback_to_eval.py
```

This script creates draft cases only. A human should review them before adding them to the regression set.
