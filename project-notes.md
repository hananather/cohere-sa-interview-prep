# Project Notes

## The core strategy

For this interview, I would frame the solution as DefTech Defence Agent: a secure, traceable, evidence-grounded assistant that helps authorized central planning staff interrogate approved manuals, procedures, and doctrine; compare guidance across versions; verify claims against source material; summarize long documents; and produce cited planning-support outputs while preserving human accountability.

That framing maps directly to the assignment: the interview is a role-play where you are acting as a Cohere Solution Architect, expected to present a Cohere-based solution, outline the problem, walk through the technical architecture, handle technical and business questions, and show a live demo. The Defence Agent scenario specifically says the customer has PDFs and DOCX files in a central database and wants to increase the efficiency and output of central planning staff while making accuracy and traceability key.

The strategic move is to avoid presenting this as “chat with PDFs.” Present it as a controlled evidence workflow: the system searches approved sources, applies access controls before retrieval, reranks evidence, generates answers with citations, validates support for claims, abstains when evidence is insufficient, and logs the full trace for audit and review. Our own project notes already converge on this: the demo should prove authorized retrieval, hybrid retrieval plus rerank, cited synthesis, ambiguity handling, abstention, trace/audit, and an eval report rather than just answer text.

## Source-Of-Truth Anchor

- The governing assignment brief is `source-materials/presentation-interview-instructions.md`.
- The original PDF is `source-materials/Nov 2025 - SA Presentation Interview Instructions (1).pdf`.
- Treat this file as interpretation of the brief, not independent truth.
- Current strategic filter: prioritize trust, security, privacy, accuracy,
  traceability, and technical correctness over feature breadth.

## Product Thesis

DefTech wants an AI assistant to help staff interrogate manuals, procedures, and
doctrine after moving PDFs and DOCX files into a central database. The current
prototype indexes normalized page evidence. PDF-origin and DOCX-origin sources
can use the same Embed v4, Rerank v4, ACL, citation, and audit workflow after
verified normalization to rendered PDF pages.

Native `.docx` parsing is not implemented in this prototype. The interview-safe
claim is narrower and stronger: the retrieval layer is format-agnostic after
verified normalization. At production scale, native enterprise DOCX ingestion
should be handled through Compass or a manifest-compatible parser and
normalization adapter.

The strongest demo is source-grounded staff-work support:

- Find relevant source pages quickly.
- Compare evidence across documents and languages.
- Cite the sources used for each answer.
- Refuse when authorized evidence is insufficient.
- Keep restricted evidence out of the model context unless the persona is
  authorized.

## User And Buyer Assumptions

- The Chief of Staff is the executive sponsor from the prompt.
- "Central planning staff" are interpreted as headquarters-level staff,
  analysts, staff officers, and planners who prepare source-grounded material
  for coordination and leadership review.
- Canadian defence documents are a corpus choice for the prototype, not a fact
  explicitly specified by the brief.
- Outputs should be staff-review-ready, not final, self-authorizing, or a
  replacement for accountable human judgement.

## High-Value Staff Tasks

| Task | User pain | Demo capability |
| --- | --- | --- |
| Source discovery | Finding the right manual, policy, section, and page | Retrieve and cite source pages |
| Claim verification | Checking whether a statement is supported and current | Show citations and refusal behavior |
| Requirements extraction | Turning long guidance into actions, roles, constraints, and risks | Grounded summaries with source metadata |
| Document comparison | Reconciling related guidance across documents | Multi-query retrieval and cited synthesis |
| Access-controlled lookup | Answering only from evidence a user may see | Persona-aware retrieval filters |

## Guardrails And Non-Goals

- Do not frame the assistant as autonomous decision-making.
- Do not claim it approves plans, tasks units, replaces staff judgement, or
  determines policy compliance by itself.
- Do not claim native Docx parsing, scanned-document answering, production
  accreditation, or complete factual verification unless those capabilities are
  implemented and tested.
- Do claim DOCX-origin support only when framed as verified normalization into
  rendered page evidence, not as direct `.docx` ingestion.
- Use "workflow assistance" rather than "workflow automation" unless discussing
  low-risk administrative steps.
- Prefer "decision support outputs," "planning work products," and
  "source-grounded staff products."
- Avoid "planning package" as the main term because it has a formal program
  controls meaning in some contexts.

## Presentation Frame

- Start from the staff workflow, not model features.
- Demo the system before the technical walkthrough.
- Connect each technical choice back to accuracy, traceability, security, or
  staff efficiency.
- Treat citations as a trust bridge, not a guarantee of truth.
- Keep production claims tied to concrete product paths: Cohere private
  deployment, Model Vault, Compass, or customer-approved infrastructure.
