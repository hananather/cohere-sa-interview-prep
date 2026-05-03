# Defence Agent Demo Script

## Opening Line

Defence Agent is an eval-driven agentic RAG system: it authenticates the user, routes the query, retrieves only authorized evidence, reranks with Cohere, uses tools when analysis is required, generates cited answers, and records a full trace for audit and improvement.

## Before The Interview

- Run `python -m pip install -e ".[dev]"`.
- Use `.env` for the demo key. Keep `.env` out of git.
- Start locally with `defence_agent/scripts/run_demo.sh`.
- Open `http://127.0.0.1:8501`.
- Keep mock mode available as the fallback by setting `USE_MOCK_COHERE=true`.

## Flow 1: Direct RAG

- Persona: `planning_analyst`.
- Query: `What review steps should planning staff complete before approving a cross-unit planning request?`
- Show: route, citations, source cards, rerank scores.
- Point: this is grounded RAG with traceability, not open-ended chat.

## Flow 2: Version Comparison

- Persona: `planning_lead`.
- Query: `Compare the 2024 and 2025 review gate procedure. What changed and what is the impact?`
- Show: trace with version-specific retrieval and compare tool.
- Point: agentic workflow decomposes a multi-document task.

## Flow 3: Table Analysis

- Persona: `planning_analyst`.
- Query: `Using the readiness review table, which units fall below the 80% readiness threshold?`
- Show: sandboxed Python tool call and source table.
- Point: the agent uses deterministic computation rather than model-only arithmetic.

## Flow 4: Permission-Aware Retrieval

- Persona A: `planning_analyst`.
- Persona B: `planning_lead`.
- Query: `What does Restricted Annex B say about exception handling?`
- Show: same query produces different results because ACL filters run before retrieval.
- Point: unauthorized chunks never enter the model context.

## Flow 5: Prompt Injection Defense

- Persona: `planning_lead`.
- Query: `Summarize the exception handling guidance from the test document.`
- Show: safety flags and sanitized source text.
- Point: retrieved content is evidence, never instructions.

## Close

- Run the Evaluation tab.
- Show pass rate by slice.
- Explain how this becomes a production release gate.
