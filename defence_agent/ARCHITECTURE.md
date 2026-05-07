# Architecture

## Position

Defence Agent is an eval-driven agentic RAG demo for public-sector document intelligence. It augments planning staff; it does not replace human approval.

## Components

- Streamlit: presentation UI with Guided Demo, Ask, Persona Compare, Trace Inspector, Evaluation Harness, Governance / Tool Registry, and Architecture / Production Path.
- FastAPI: auth, routing, retrieval, tools, traces, evals, and metrics.
- Cohere ClientV2: Chat, Embed, Rerank, and streaming paths.
- SQLite: document metadata, chunks, lexical FTS, traces, eval results, feedback, and sandbox logs.
- Qdrant: vector store when available, with SQLite vector fallback.
- SandboxRunner: local safe runner for controlled pandas analysis.
- Advanced eval harness: suite loader, layered graders, experiment comparison, trace replay, and demo selection.

## Presentation Views

- Guided Demo: step-by-step interview path with expected behavior and speaker notes.
- Ask: clean end-user interface for free-form questions.
- Persona Compare: runs one query for two personas and shows source, policy, and answer differences.
- Trace Inspector: ordered span table plus policy, retrieval, generation, and citation panels.
- Evaluation Harness: suite/case selection, experiment variants, and pass/fail breakdown.
- Governance / Tool Registry: persona access matrix, tool risk registry, and policy decision log.
- Architecture / Production Path: Cohere model flow, plugin lifecycle, and production hardening story.

## Request Flow

1. Authenticate demo persona from `X-Demo-User`.
2. Validate input and assign a trace ID.
3. Route the query to the lightest sufficient workflow.
4. Apply ACL and metadata filters before retrieval.
5. Retrieve with lexical plus vector search.
6. Rerank candidates with Cohere Rerank or use hybrid-score fallback.
7. Execute tools only through the policy-gated registry.
8. Generate a cited answer from retrieved evidence or return a safe refusal.
9. Validate citations and output safety.
10. Store spans, response, and a run JSON trace.

## Routes

- `evidence_lookup`
- `grounded_summary`
- `metadata_aware_retrieval`
- `cross_source_synthesis`
- `version_comparison`
- `structured_table_analysis`
- `claim_verification`
- `permission_sensitive_retrieval`
- `bilingual_retrieval`
- `refuse_or_clarify`

## Tool Boundary

Tools are Pydantic-validated functions with authorization checks. The model does not get broad filesystem, network, database, or write access.

Key tools:

- `search_documents`
- `get_document_sections`
- `follow_references`
- `compare_document_versions`
- `get_table`
- `run_table_analysis`
- `validate_answer_citations`

## Observability Plugin Pattern

The local plugin layer is ADK-aligned without replacing the Cohere-native demo.

Implemented hooks:

- `TracePlugin`: records before/after tool spans and errors.
- `PolicyGuardPlugin`: enforces persona tool access before tool execution.
- `MetricsPlugin`: records tool result size and success/failure spans.
- `RedactionPlugin`: keeps the lifecycle location for privacy redaction.
- `FeedbackPlugin`: links feedback events to trace IDs.

These hooks map cleanly to Google ADK-style callbacks such as `before_tool`, `after_tool`, and `on_tool_error`. In this demo, Cohere remains the model layer.

## Trace Schema

Raw trace records are stored in SQLite and `defence_agent/data/runs/`.

The UI consumes redacted structured traces from:

```text
GET /v1/traces/{trace_id}/structured
```

The structured trace includes:

- persona and query metadata
- policy decision and reasons
- ordered spans
- retrieval context and excluded sources
- generation, citations, and citation validation
- eval placeholders and feedback linkage

Restricted-source answers are redacted for personas that can view audit metadata but not restricted content.

## Production Hardening Path

For production, replace demo headers with SSO/RBAC, use private networking, managed databases, encrypted object storage, centralized logs, egress allowlists, and a hardened containerized sandbox.

Production observability should export spans to an OpenTelemetry-compatible collector, forward security events to a SIEM, and store full restricted prompts only under explicit break-glass controls.

## Evaluation Flow

1. Load a typed eval case from `defence_agent/data/evals/`.
2. Run the query through the same agent service used by the UI.
3. Store a trace with route, filters, candidates, tools, citations, sandbox policy, and metrics.
4. Grade each component independently.
5. Aggregate results by task type, complexity, route, tool, citation, safety, and structured analysis.
6. Write reports to `reports/eval/latest/`.

The harness supports fixture mode for CI and live mode when `COHERE_API_KEY` is present. Fixture mode is deterministic and should not be presented as live Cohere performance.
