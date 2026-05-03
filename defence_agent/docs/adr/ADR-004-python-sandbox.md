# ADR-004: Use Sandboxed Python For Tables

## Decision

Analyze readiness tables with a constrained Python sandbox.

## Reason

Numerical table analysis should be deterministic and inspectable. Model-only arithmetic is harder to trust in a public-sector workflow.

## Consequence

The sandbox blocks file, network, process, and dynamic code execution and logs every run.
