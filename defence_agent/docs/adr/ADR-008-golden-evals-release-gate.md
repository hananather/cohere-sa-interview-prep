# ADR-008: Use Golden Evals As A Release Gate

## Decision

Maintain a golden dataset and eval runner inside the prototype.

## Reason

The safest way to improve an agent is to track route, retrieval, citation, permission, safety, and latency behavior over time.

## Consequence

The Evaluation tab shows demo-ready quality metrics and creates a natural path to CI gates.
