# Security

## Demo Controls

- No anonymous access.
- Demo personas carry role, groups, clearance, and tenant.
- Personas are explicit:
  - Alex Chen, Planning Analyst: public-internal doctrine only.
  - Morgan Singh, Doctrine Steward: public-internal plus restricted doctrine.
  - Priya Rao, Security Auditor: policy metadata and audit traces without restricted content by default.
  - Sam Rivera, Platform Admin: platform administration without automatic restricted doctrine access.
- Retrieval applies authorization before the model sees context.
- Metadata filters enforce approved/current/language/access behavior.
- Tools are allowlisted and Pydantic-validated.
- Tool outputs are sanitized before generation.
- Restricted sources are not returned to `planning_analyst` or `auditor`.
- Traces record route, filters, candidates, tool calls, citation validation, and sandbox state.
- Default structured trace views redact answers that used restricted sources when the viewer lacks restricted content permission.
- Full prompt/context logging is disabled unless `DEFTECH_DEBUG_FULL_TRACE=true`.

## Why The LLM Is Not The Boundary

- Access control is enforced by the backend policy engine before retrieval and before tool execution.
- Metadata filters exclude draft, superseded, language-mismatched, and unauthorized sources before generation.
- The model can only answer from the evidence and tool results the orchestrator provides.
- Retrieved text is treated as untrusted evidence, not as system instruction.

## Sandbox

The local demo uses a safe runner, not a production isolation boundary.

Policy shown in traces:

```json
{
  "network": "disabled",
  "filesystem": "ephemeral_workspace_only",
  "source_documents": "read_only",
  "secrets": "not_mounted",
  "egress": "blocked",
  "timeout_seconds": 5,
  "allowed_packages": ["pandas", "numpy", "datetime", "json", "math", "statistics"],
  "audit": ["code", "stdout", "stderr", "input_dataset_hash", "row_ids"]
}
```

The runner blocks imports and dangerous names with AST validation, runs in a separate process, and times out.

## Feedback And Audit Logs

- Feedback events are stored locally in `defence_agent/data/feedback/feedback_events.jsonl`.
- Feedback captures trace ID, query ID, route, tools, rating, and selected failure type.
- Feedback is not treated as ground truth. It is promoted to draft eval cases for human review.
- Production logs should scrub sensitive content before long-term retention.
- Production audit events should flow to a SIEM, security information and event management system.

## Tool Risk Classification

- `search_documents`, `get_document_sections`, `follow_references`, `compare_document_versions`, and citation validation are read-only.
- `run_table_analysis` is sandboxed medium risk because it executes controlled analysis over authorized rows.
- `export_brief_draft` is modeled as reversible write and requires approval before production use.
- `admin_reindex` is admin-only and disabled as a model-invoked demo tool.

Tool policy decisions are visible in the Governance view and in trace spans named `before_tool_policy`.

## Prompt Injection Defense

- Prompt-like text inside retrieved documents does not override system policy.
- Draft and superseded documents remain excluded when the route asks for current approved guidance.
- Access-control filters cannot be bypassed by user wording such as "ignore metadata" or "override access control."

## Production Controls

- OIDC or SAML SSO with RBAC/ABAC claims.
- Per-user retrieval filters and optional index namespaces by access domain.
- Private deployment or customer-managed Cohere deployment when required.
- No customer data used for model training.
- Secrets manager instead of local `.env`.
- VPC/private networking and egress allowlists.
- Encrypted object storage and database encryption.
- SIEM export for audit events.
- PII and sensitive-data scrubbing before long-term logging.
- Hardened container sandbox with stronger isolation.
- CI/CD eval gates and canary rollout.
