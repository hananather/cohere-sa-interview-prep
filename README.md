# Cohere SA Interview Prep

Private workspace for the final-round Cohere Solutions Architect interview project.

## Primary Interview Frame

This repo exists to help win the final-round Cohere Public Sector Solutions Architect interview.

The scenario is fictional. Defence Agent is the vehicle for proving the competencies Cohere is evaluating.

The final presentation has two primary artifacts:

1. Jupyter notebook presentation.
2. Live Defence Agent demo.

Everything else should make those two artifacts stronger or be cut.

The core competencies to demonstrate are technical expertise, applied customer solutioning, live demo execution, implementation depth, scalability, security, adaptability, accuracy, traceability, and stakeholder communication.

Only two sources define competencies for this project:

1. The presentation PDF.
2. The Cohere Public Sector Solutions Architect job description.

Everything else is hypothesis or support material until it maps back to one of
those two sources.

## The core strategy

For this interview, I would frame the solution as DefTech Defence Agent: a secure, traceable, evidence-grounded assistant that helps authorized central planning staff interrogate approved manuals, procedures, and doctrine; compare guidance across versions; verify claims against source material; summarize long documents; and produce cited planning-support outputs while preserving human accountability.

That framing maps directly to the assignment: the interview is a role-play where you are acting as a Cohere Solution Architect, expected to present a Cohere-based solution, outline the problem, walk through the technical architecture, handle technical and business questions, and show a live demo. The Defence Agent scenario specifically says the customer has PDFs and DOCX files in a central database and wants to increase the efficiency and output of central planning staff while making accuracy and traceability key.

The strategic move is to avoid presenting this as “chat with PDFs.” Present it as a controlled evidence workflow: a Research Agent searches approved sources, applies access controls before retrieval, reranks evidence, and produces a cited answer; a Reviewer Agent checks whether the cited evidence supports the claims and returns a credibility score. The demo always shows the answer, citations, score, and trace. The score becomes the quality signal for retry, human review, or pilot go/no-go decisions.

## Evaluation-First Solution Design

The design principle is to build the evaluation set before adding retrieval
complexity. The eval set defines the workload and becomes the reference point
for choosing chunking, retrieval, prompts, tools, models, access-control
behavior, and refusal behavior.

![Eval-driven solution design](notebooks/assets/eval-driven-solution-design.png)

## High-Signal Files

- `source-materials/presentation-interview-instructions.md`: source-of-truth assignment brief parsed from the Cohere PDF.
- `source-materials/Nov 2025 - SA Presentation Interview Instructions (1).pdf`: original Cohere assignment PDF.
- `source-materials/cohere-public-sector-sa-job-description.md`: local job-description competency source.
- `project-notes.md`: canonical inferred strategy and Canadian Defence staff decision-support framing.
- `defence_agent/`: runnable ADK-first Defence Agent prototype for the Cohere final-round demo.
- `defence_agent/docs/role_competency_frame.md`: local rubric for mapping each artifact to the technical Public Sector Solutions Architect competencies it proves.
- `defence_agent/docs/presentation_architecture_walkthrough.md`: presentation-ready architecture walkthrough with user-flow, data-flow, guardrail, agentic-loop, and citation diagrams.
- `defence_agent/docs/technical_deep_dive_sequence.md`: proof-first technical walkthrough sequence for app, Trace, Cursor code anchors, eval, and production mapping.
- `defence_agent/docs/cohere_capability_matrix.md`: current Cohere source inventory and Q&A positioning.
- `defence_agent/docs/demo_script.md`: demo flow and rehearsal commands.

## Technical Presentation Walkthrough

Use `defence_agent/docs/presentation_architecture_walkthrough.md` as the
read-through for the technical portion of the 25-minute interview. It keeps the
story simple:

1. A user query enters Streamlit.
2. The Research Agent sends the user query and session context to the model
   runtime, Cohere Command A (`command-a-03-2025`).
3. The model runtime produces an ordered tool plan: one or more
   `search_documents(query, filters)` calls.
4. Each `search_documents` call embeds its exact tool query with Cohere Embed
   v4, searches Chroma with persona-aware metadata filters, and reranks
   candidate pages with Cohere Rerank v4.
5. Access failures or no-authorized-source cases stay zero-doc; authorized
   evidence gaps continue to grounded generation for model-grounded abstention.
6. Direct Cohere Chat / Command A (`command-a-03-2025`) receives only
   authorized pages and returns native document-RAG citation spans.
7. The Reviewer Agent scores whether the cited evidence supports the answer's
   claims.
8. The UI renders the answer, citation-linked evidence, reviewer score,
   excluded-source metadata, and `answer_audit` trace.

The default demo model remains `command-a-03-2025`, matching Cohere's documented
agentic RAG tool-use examples. If `COHERE_CHAT_MODEL` is set to a
reasoning-capable model such as `command-a-reasoning-08-2025`, native Cohere
`thinking` content blocks are preserved in the generation audit and can be shown
in Trace. Treat those thinking blocks as model-generated reasoning content, not
as a substitute for the observable tool/action audit.

The strongest proof sequence is:

- Planning brief comparison: multi-search public-source answer with citations.
- Access boundary: the same restricted query refuses for an unclassified user
  and answers for a cleared user.
- Audit follow-up: source and access-level details come from prior
  `answer_audit`, not a fresh unrelated search.
- Insufficient-evidence refusal: Command A reviews authorized retrieved pages
  and abstains when the pages do not support the requested claim.

## Working Rule

- Treat the Cohere assignment brief as the source for the interview criteria.
- Treat the job description as the source for role-specific competencies.
- Tie strategy and implementation back to the brief's requirements: live demo, customer stakeholder clarity, technical walkthrough, scalability, security, PDFs/Docx, staff efficiency/output, accuracy, and traceability.
- Tie strategy and implementation back to job-description signals: hands-on demos and proof of concepts, agentic AI/North, scalable and secure solutions, complex workflows, pilots, fine-tuning/custom agents/orchestration, Kubernetes/Docker/cloud deployment, Python/Jupyter, product feedback, and executive/technical communication.
- Use the other PDF scenarios as signals for what Cohere values: relevance, personalization, privacy, confidentiality, integration, transparency, compliance, adaptability, and measurable business impact.
- Tie presentation decisions back to the two primary artifacts: Jupyter notebook and live demo.
- Be actively ambitious. Push toward higher-signal competency proof and cut lower-signal context.
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
