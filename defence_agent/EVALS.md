# Evaluation

## Dataset

Canonical demo queries live in `defence_agent/data/evals/demo_queries.yaml`.

The executable golden dataset lives in `defence_agent/evals/golden_dataset.yaml`.

Each case includes:

- query
- user persona
- expected route
- expected tools
- expected source document IDs
- forbidden sources when relevant
- expected answer substrings
- citation expectation
- refusal expectation

## Metrics

- Route accuracy
- Tool accuracy
- Retrieval recall at K
- Context precision
- Citation presence
- Citation validation
- Permission correctness
- Abstention correctness
- Safety pass rate
- Structured-analysis exactness
- Latency
- Error rate

## Acceptance Gate

Fixture mode must pass:

- 100% route accuracy.
- 100% permission correctness.
- 100% citation validation.
- Exact overdue-review results for Q6:
  - `PB-SOP-2025`: 52 days overdue.
  - `EC-PROC-2025`: 35 days overdue.
  - `EC-PROC-2025-FR`: 35 days overdue.
  - `LOG-RET-2025`: 96 days overdue.

Run:

```bash
make eval
python -m pytest
```
