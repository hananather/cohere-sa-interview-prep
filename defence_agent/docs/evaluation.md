# Evaluation

## Golden Dataset

The eval suite has 24 cases across:

- Direct answerable questions.
- Ambiguous questions.
- Unanswerable questions.
- Version comparison.
- Table analysis.
- Permission-sensitive retrieval.
- Prompt injection.
- Stale or conflicting source handling.
- Human review.

## Metrics

- Route accuracy.
- Tool accuracy.
- Retrieval recall at K.
- Context precision.
- Citation presence.
- Citation validation.
- Permission correctness.
- Abstention correctness.
- Safety pass rate.
- Latency.
- Error rate.

## Demo Use

Run evals from the Evaluation tab or with:

```bash
make eval
```

Treat this as a release gate. A production version would fail deployment if route, permission, safety, or citation metrics regress.
