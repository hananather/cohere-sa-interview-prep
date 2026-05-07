# Cohere SA Interview Prep

Private workspace for the final-round Cohere Solutions Architect interview project.

## High-Signal Files

- `project-notes.md`: current product thesis and Canadian Defence staff decision-support framing.
- `defence_agent/`: runnable Defence Agent Streamlit + FastAPI prototype for the Cohere final-round demo.
- `WORKLOG_UI_OBSERVABILITY.md`: latest implementation notes for guided demo, persona comparison, trace redaction, and governance UI.
- `WORKLOG_EVAL_UPGRADE.md`: evaluation harness implementation notes and verification loop.
- `context-map-april-25.md`: April 25 context snapshot. Treat it as useful but possibly stale.
- `source-materials/presentation-interview-instructions.md`: text extracted from the interview assignment PDF.
- `source-materials/Nov 2025 - SA Presentation Interview Instructions (1).pdf`: original assignment PDF.
- `microsoft-guidance/rag.md`: Microsoft RAG design and evaluation framework, summarized for this project.

## Working Rule

- Keep this repo focused on the final presentation, demo, architecture, and Q&A prep.
- Summarize external context instead of importing large prior prep folders, inbox threads, or raw transcripts.
- The `cohere-compass-sdk/` directory is kept locally as a tool dependency and is intentionally ignored by this repository.
- Treat project context as evolving.
- Ask follow-up questions when assumptions affect the business problem, user workflow, or solution scope.

## Run The Demo

```bash
USE_MOCK_COHERE=true defence_agent/scripts/run_demo.sh
```

Then open:

- Streamlit UI: `http://127.0.0.1:8501`
- FastAPI health: `http://127.0.0.1:8000/healthz`

The Streamlit app now has presentation views for Guided Demo, Ask, Persona Compare, Trace Inspector, Evaluation Harness, Governance / Tool Registry, and Architecture / Production Path.

## Verify

```bash
python -m pytest defence_agent/tests/test_defence_agent.py -q
python defence_agent/scripts/run_eval_harness.py validate
python defence_agent/scripts/run_eval_harness.py run --suite canonical --mode fixture --variant agentic_rag_tools
```
