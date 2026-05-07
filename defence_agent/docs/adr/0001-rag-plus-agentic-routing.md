# 0001 RAG Plus Agentic Routing

## Context

Simple document questions should stay fast, but comparison, cross-reference following, metadata-sensitive retrieval, and structured analysis need more than one retrieval call.

## Decision

Use a router plus tool workflows. Default to evidence lookup, then activate agentic workflows only when the query requires them.

## Alternatives Considered

- Single RAG path for every query.
- Full multi-agent system from the start.

## Consequences

The demo shows judgement: more agency is used only when it improves task success or trust.

## Status

Accepted.
