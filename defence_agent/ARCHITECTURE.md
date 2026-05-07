# Architecture

## Position

Defence Agent is an eval-driven agentic RAG demo for public-sector document intelligence. It augments planning staff; it does not replace human approval.

## Components

- Streamlit: presentation UI with User View and Demo Console.
- FastAPI: auth, routing, retrieval, tools, traces, evals, and metrics.
- Cohere ClientV2: Chat, Embed, Rerank, and streaming paths.
- SQLite: document metadata, chunks, lexical FTS, traces, eval results, feedback, and sandbox logs.
- Qdrant: vector store when available, with SQLite vector fallback.
- SandboxRunner: local safe runner for controlled pandas analysis.

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

## Production Hardening Path

For production, replace demo headers with SSO/RBAC, use private networking, managed databases, encrypted object storage, centralized logs, egress allowlists, and a hardened containerized sandbox.
