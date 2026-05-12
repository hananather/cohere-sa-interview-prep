# Cohere SA Interview Prep

Private workspace for the final-round Cohere Solutions Architect interview project.

## The core strategy

For this interview, I would frame the solution as DefTech Defence Agent: a secure, traceable, evidence-grounded assistant that helps authorized central planning staff interrogate approved manuals, procedures, and doctrine; compare guidance across versions; verify claims against source material; summarize long documents; and produce cited planning-support outputs while preserving human accountability.

That framing maps directly to the assignment: the interview is a role-play where you are acting as a Cohere Solution Architect, expected to present a Cohere-based solution, outline the problem, walk through the technical architecture, handle technical and business questions, and show a live demo. The Defence Agent scenario specifically says the customer has PDFs and DOCX files in a central database and wants to increase the efficiency and output of central planning staff while making accuracy and traceability key.

The strategic move is to avoid presenting this as “chat with PDFs.” Present it as a controlled evidence workflow: the system searches approved sources, applies access controls before retrieval, reranks evidence, generates answers with citations, validates support for claims, abstains when evidence is insufficient, and logs the full trace for audit and review. Our own project notes already converge on this: the demo should prove authorized retrieval, hybrid retrieval plus rerank, cited synthesis, ambiguity handling, abstention, trace/audit, and an eval report rather than just answer text.

## High-Signal Files

- `source-materials/presentation-interview-instructions.md`: source-of-truth assignment brief parsed from the Cohere PDF.
- `source-materials/Nov 2025 - SA Presentation Interview Instructions (1).pdf`: original Cohere assignment PDF.
- `project-notes.md`: canonical inferred strategy and Canadian Defence staff decision-support framing.
- `defence_agent/`: runnable ADK-first Defence Agent prototype for the Cohere final-round demo.
- `defence_agent/docs/cohere_capability_matrix.md`: current Cohere source inventory and Q&A positioning.
- `defence_agent/docs/demo_script.md`: demo flow and rehearsal commands.

## Working Rule

- Treat the Cohere assignment brief as the highest-priority context document.
- Tie strategy and implementation back to the brief's requirements: live demo, customer stakeholder clarity, technical walkthrough, scalability, security, PDFs/Docx, staff efficiency/output, accuracy, and traceability.
- Frame DOCX carefully: native `.docx` parsing is not implemented, but one DOCX-origin source is represented through verified PDF-pair normalization and then uses the same page-evidence retrieval workflow.
- Keep this repo focused on the final presentation, demo, architecture, and Q&A prep.
- Summarize external context instead of importing large prior prep folders, inbox threads, or raw transcripts.
- The `cohere-compass-sdk/` directory is kept locally as a tool dependency and is intentionally ignored by this repository.
- Treat project context as evolving.
- Ask follow-up questions when assumptions affect the business problem, user workflow, or solution scope.

## Run The Demo

```bash
make run
```

Then open the primary demo surface:

- Streamlit demo UI: `http://127.0.0.1:8501`

For technical inspection only, run `make run-adk`, then open
`http://127.0.0.1:8502` and choose `defence_agent` in the ADK Dev UI. ADK
startup can run index checks and can trigger paid Cohere Embed calls when the
local index is stale.

The retired legacy prototype is archived outside this main repo at
`/Users/hananather/Desktop/Cohere-legacy-archive`.

## Verify

Fast local checks do not call Cohere and skip expensive local PDF-render checks:

```bash
python -m pytest defence_agent/tests/
```

Run the full offline suite when corpus rendering or source-preview behavior changed:

```bash
python -m pytest defence_agent/tests/ --run-slow-offline
```

Live readiness checks do call Cohere and should be run deliberately:

```bash
python defence_agent/scripts/build_chroma_index.py
python -m pytest defence_agent/tests/ --run-live -m live
python defence_agent/scripts/run_demo_query_registry.py --no-fail
```

The active prototype requires `COHERE_API_KEY` in the repo-root `.env` and uses
real Cohere calls for embeddings, reranking, and final cited answers. Local
Chroma indexes and transcripts stay under `defence_agent/data/` and are ignored
by git.
