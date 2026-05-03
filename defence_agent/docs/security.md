# Security

## Controls

- No anonymous access.
- Demo users are fixed personas with role, groups, clearance, and tenant.
- The policy engine checks chunk access and tool access.
- Retrieval applies ACL filters before the model sees any context.
- Tool inputs are Pydantic-validated.
- Tool outputs are sanitized before model context.
- Python table analysis runs in a separate process with AST validation.
- `.env` is ignored and should be mode `600`.

## Demo Personas

- `planning_analyst`: protected clearance, no restricted annex access.
- `planning_lead`: restricted clearance, can access Restricted Annex B.
- `auditor`: protected clearance, can inspect metadata and traces.
- `admin`: restricted admin user.

## Production Mapping

- Replace demo headers with OIDC or SAML JWT validation.
- Replace SQLite with Postgres or managed relational storage.
- Replace local `.env` with a secret manager.
- Export traces and metrics to the customer observability stack.
