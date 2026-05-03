# ADR-001: Use Direct Cohere SDK Over ADK

## Decision

Use Cohere `ClientV2` directly for the live demo.

## Reason

The interview is Cohere-focused. Direct SDK calls make Cohere Embed, Rerank, Chat, and Chat streaming visible in the code and easier to defend.

## Consequence

The architecture remains ADK-inspired through routers, tools, workflows, traces, and evals, but ADK is not a runtime dependency.
