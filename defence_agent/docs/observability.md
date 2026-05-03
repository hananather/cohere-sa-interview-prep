# Observability

## Trace Spans

Each request stores spans for:

- Request received.
- Auth validated.
- Input safety checked.
- Route selected.
- Plan created.
- Tool call started.
- ACL filter applied.
- Lexical search completed.
- Vector search completed.
- Rerank completed.
- Context assembled.
- Model generation started.
- Citation validation completed.
- Output safety checked.
- Response returned.

## Storage

- SQLite trace tables support UI lookup.
- JSONL trace logs support export and offline review.
- `/metrics` exposes Prometheus-style counters and histograms.

## Demo Use

After asking a question:

- Open the Trace tab.
- Load the trace ID.
- Show route, plan, tool calls, filters, rerank scores, citations, safety, latency, and token estimates.
