# 0001 ADK Runtime With Direct Cohere Final Answer

## Decision

Use Google Agent Development Kit (ADK) for the runtime, session state, and tool
orchestration. Use direct Cohere SDK calls for the final grounded answer.

## Reason

The interview demo needs visible agent runtime behavior and source-grounded
Cohere output. ADK gives the agent loop and session tooling. Cohere Embed v4,
Rerank v4, and Chat document citations stay explicit in the code path.

## Consequence

The final answer call receives only authorized evidence through
`documents=...`, so Cohere native citation spans can be shown and audited.
