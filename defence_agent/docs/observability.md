# Observability

## Position

Use ADK and OpenTelemetry for runtime tracing. Use `answer_audit` for domain
traceability.

These are different:

- Runtime trace: agent invocation, model calls, tool calls, state events, timing,
  and errors.
- Answer audit: persona, filters, authorized sources, excluded source IDs,
  answerability decisions, rerank scores, Cohere document IDs, citation spans,
  and citation source IDs.

## Current Prototype

- ADK events capture the agent turn and `search_documents` tool execution.
- `search_documents` stores sanitized retrieval audit metadata in ADK state.
- The ADK tool cache records cache stores and hits in session state.
- `session.py` returns `AgentTurnResult.answer_audit`.
- `run_agent_session.py --show-audit` prints the audit JSON for demo review.
- Python logging records compact intent/outcome events.

The prototype does not run a custom trace service. If external traces are
needed, use ADK's OpenTelemetry export:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://localhost:4318/v1/traces"
./defence_agent/scripts/run_adk_agent.sh
```

## Demo Use

For a stakeholder traceability question, show:

- ADK event count and tool calls.
- `answer_audit.retrieval.filters_applied`.
- `answer_audit.retrieval.answerability`.
- `answer_audit.retrieval.tool_cache`.
- `answer_audit.retrieval.sources_sent_to_answer`.
- `answer_audit.retrieval.excluded_sources`.
- `answer_audit.generation.citation_mode`.
- `answer_audit.citations`.

Do not enable full prompt or source-text capture in production tracing unless the
customer explicitly approves that data path.
