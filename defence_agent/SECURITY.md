# Security

## Demo Controls

- No anonymous access.
- Demo personas carry role, groups, clearance, and tenant.
- Retrieval applies authorization before the model sees context.
- Metadata filters enforce approved/current/language/access behavior.
- Tools are allowlisted and Pydantic-validated.
- Tool outputs are sanitized before generation.
- Restricted sources are not returned to `planning_analyst` or `auditor`.
- Traces record route, filters, candidates, tool calls, citation validation, and sandbox state.

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
