# 0002 Secure Code Execution Tool

## Context

The overdue-review query requires deterministic date math and grouping over a table. A language model should not eyeball this result.

## Decision

Use a policy-gated sandbox runner over authorized table rows. Log code, input hash, row IDs, output, errors, and runtime.

## Alternatives Considered

- Ask the model to calculate from table text.
- Expose an unrestricted Python shell.

## Consequences

The local runner is demo-safe but not production-hardened. Production should use a containerized sandbox with stronger isolation.

## Status

Accepted.
