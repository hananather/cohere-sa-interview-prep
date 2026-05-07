# UI And Observability Upgrade Worklog

## 2026-05-06 Repository Inspection

Implementation goal:
- Make the existing Defence Agent prototype presentation-grade with guided demo views, persona comparison, trace inspection, governance controls, and ADK-aligned observability hooks.

What already existed:
- FastAPI backend in `defence_agent/backend/defence_agent/api.py`.
- Streamlit frontend in `defence_agent/frontend/app.py`.
- Demo personas in `defence_agent/backend/defence_agent/auth/context.py`.
- ACL and tool allowlist enforcement in `defence_agent/backend/defence_agent/auth/policy.py`.
- SQLite-backed trace records and spans in `defence_agent/backend/defence_agent/observability/tracing.py`.
- Tool registry with search, compare, table analysis, sandbox, citation validation, and feedback tools.
- Synthetic corpus, ingestion, routing, workflows, source drilldown, and advanced eval harness.
- Existing tests covering RAG, permissions, source endpoint ACLs, streaming, sandbox safety, and advanced eval fixture pass.

What is being reused:
- Existing `AskRequest` / `AskResponse` workflow path.
- Existing route classifier and route-specific workflow functions.
- Existing `TraceManager` storage instead of replacing it with a new tracer.
- Existing tool registry and feedback persistence.
- Existing eval schemas and generated eval suites.

What is being added:
- Presentation-safe persona profiles and access matrix.
- Redacted structured trace summaries for UI and tests.
- Lightweight ADK-aligned plugin/callback interfaces for trace, policy, metrics, redaction, and feedback concerns.
- Governance metadata for tool risk, allowed personas, and trace fields.
- Guided Demo, Persona Compare, Trace Inspector, Governance / Tool Registry, and Architecture views.
- Persona/security eval fields and graders.
- Feedback-to-eval promotion script.

What is intentionally not changed:
- Cohere-native model layer remains the main path. No Gemini or Google ADK runtime is introduced.
- Existing eval harness remains the core runner. New graders extend it.
- Existing corpus and query files are preserved unless a security-specific query needs to be added.
- Existing raw trace storage remains internal. New UI endpoints expose redacted structured traces.

## 2026-05-06 Implementation Pass

What changed:
- Added redaction and structured trace helpers under `defence_agent/backend/defence_agent/observability/`.
- Added lifecycle plugin hooks for tool policy, trace, metrics, redaction, and feedback.
- Added persona profiles, source access matrix, governance tool registry, recent policy decisions, and side-by-side persona compare endpoints.
- Updated Streamlit with Guided Demo, Ask, Persona Compare, Trace Inspector, Evaluation Harness, Governance / Tool Registry, Architecture / Production Path, and Legacy Demo Console views.
- Added persona/security eval fields and graders.
- Regenerated eval suites from the template generator.
- Added tests for persona comparison, trace redaction, governance endpoints, blocked tool policy, feedback, and security graders.

Commands run:
- `python -m compileall defence_agent/backend/defence_agent defence_agent/frontend/app.py defence_agent/scripts/generate_eval_queries.py`
- `python -m pytest defence_agent/tests/test_defence_agent.py -q`
- `python defence_agent/scripts/run_eval_harness.py validate`
- `python defence_agent/scripts/run_eval_harness.py run --suite canonical --mode fixture --variant agentic_rag_tools`

Results:
- Compile: passed.
- Unit/API tests: 16 passed.
- Eval validation: passed, 292 total cases across suites, no duplicate IDs, no missing sources.
- Canonical fixture eval: 24/24 passed.

Remaining issues:
- Browser visual QA still needs to be run after restarting the local demo server.

## 2026-05-06 Final Verification

Commands run:
- `make verify`
- `USE_MOCK_COHERE=false python -c '<three-query live smoke using FastAPI TestClient>'`
- `git diff --check`

Results:
- `make verify`: passed. Unit/API tests passed, legacy eval passed, advanced validation passed, canonical fixture eval passed, experiment comparison was generated, and demo selection artifacts were generated.
- Live Cohere smoke: passed for three representative queries. The app returned HTTP 200 for simple cited lookup, permission-sensitive retrieval, and cross-source synthesis, with citations and trace IDs on each run.
- `git diff --check`: initial generated CSV report artifacts had carriage-return line ending noise. The CSV report files were normalized and rechecked before commit.

Remaining issues:
- Browser visual QA should still be run after restarting the local Streamlit server, because the automated checks validate behavior and compileability but not pixel-level polish.
