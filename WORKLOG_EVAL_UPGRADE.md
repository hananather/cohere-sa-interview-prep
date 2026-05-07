# Evaluation Upgrade Worklog

## Repository Inspection

- App framework: FastAPI backend plus Streamlit frontend.
- Existing evals: legacy 24-case golden runner in `defence_agent/backend/defence_agent/evals/runner.py`.
- Existing data: synthetic corpus, demo query file, SQLite/Qdrant retrieval fallback, trace logging, sandbox table analysis.
- Reused: router, agent service, tool registry, hybrid retriever, trace manager, sandbox runner, existing tests.
- Added: broader eval datasets, layered graders, advanced runner, experiment comparison, demo selector, feedback-to-eval script, UI eval harness.

## Phase A: Eval Schema And Datasets

- Goal: create a broad task-complexity eval corpus and validation path.
- Changed:
  - Added `defence_agent/scripts/generate_eval_queries.py`.
  - Added `defence_agent/data/evals/eval_schema.json`.
  - Generated canonical, generated, heldout, regression, adversarial, and demo-candidate suites.
- Verification:
  - `python defence_agent/scripts/generate_eval_queries.py` passed.
  - `PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py validate` passed.
- Result:
  - Generated suite: 170 cases.
  - Canonical suite: 24 cases.
  - Demo candidates: 36 cases.

## Phase B: Layered Fixture Harness

- Goal: grade route, retrieval, filters, rerank, tools, sandbox, answer behavior, citations, safety, and operations separately.
- Changed:
  - Added `defence_agent/backend/defence_agent/evals/advanced_runner.py`.
  - Added grader modules under `defence_agent/backend/defence_agent/evals/graders/`.
  - Added schema and loader modules.
- Verification:
  - `python -m py_compile ...` passed for new harness files.
  - First canonical run exposed six focused gaps.

## Phase C: Canonical Failure Repair

- Goal: fix real routing/workflow gaps rather than weakening the evaluator.
- Changed:
  - Added metadata route triggers for draft/superseded/current-approved adversarial wording.
  - Added cross-source trigger for SOP plus evidence-checklist support questions.
  - Made summary retrieval topic-aware.
  - Made claim verification retrieve the correct source family.
  - Added template-first table analysis for `count approved documents by owner`.
  - Skipped rerank scoring requirements for structured table routes.
- Verification:
  - `PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py run --suite canonical --mode fixture` passed.
- Result:
  - 24/24 canonical cases passed.
  - Route accuracy: 100%.
  - Citation validation: 100%.
  - Structured-analysis exactness: 100%.

## Phase D: Experiment And Demo Selection

- Goal: compare retrieval variants and select reliable live demo queries.
- Commands:
  - `PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py compare --suite canonical --limit 24`
  - `PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=true python defence_agent/scripts/run_eval_harness.py select-demo --mode fixture --runs 3`
- Result:
  - Wrote `reports/eval/latest/experiment_comparison.md`.
  - Wrote `reports/eval/latest/demo_selection_report.md`.
  - Wrote `defence_agent/data/evals/recommended_demo_sequence.yaml`.
- Note:
  - Fixture-mode experiment scores compare local deterministic behavior. They are not live Cohere quality claims.

## Phase E: Regression Tests And Docs

- Goal: keep the eval upgrade runnable through `pytest` and `make verify`.
- Changed:
  - Added advanced eval tests.
  - Added Makefile targets for validation, canonical eval, experiment comparison, and demo selection.
  - Updated docs and runbooks.
- Remaining:
  - Address live rerank score capture when using a real Cohere key.

## Final Verification

- `USE_MOCK_COHERE=true python -m pytest`
  - Passed: 10 tests.
- `make verify`
  - Passed.
  - Includes unit tests, legacy eval, advanced validation, canonical fixture eval, experiment comparison, and demo selection.
- `python defence_agent/scripts/smoke_demo.py`
  - Passed against local backend.
- Live Cohere smoke:
  - Command: `PYTHONPATH=defence_agent/backend USE_MOCK_COHERE=false python defence_agent/scripts/run_eval_harness.py run --suite canonical --mode live --limit 1`
  - Critical result: 1/1 case passed.
  - Remaining live-only issue: rerank score capture was missing in that trace, so full live metrics should not be claimed until that is fixed.

## Running Demo

- Restarted the local demo in mock mode.
- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:8501`
