# Project Notes

## The core strategy

The primary goal is to win the final-round Cohere Public Sector Solutions Architect interview.

This is a fictional interview scenario. The Defence Agent demo is the vehicle for proving technical Solution Architect competencies, not an end in itself.

The presentation has only two primary artifacts:

1. A Jupyter notebook presentation.
2. A live Defence Agent demo.

Everything else should support those two artifacts or be cut.

For this interview, I would frame the solution as DefTech Defence Agent: a secure, traceable, evidence-grounded assistant that helps authorized central planning staff interrogate approved manuals, procedures, and doctrine; compare guidance across versions; verify claims against source material; summarize long documents; and produce cited planning-support outputs while preserving human accountability.

That framing maps directly to the assignment: the interview is a role-play where you are acting as a Cohere Solution Architect, expected to present a Cohere-based solution, outline the problem, walk through the technical architecture, handle technical and business questions, and show a live demo. The Defence Agent scenario specifically says the customer has PDFs and DOCX files in a central database and wants to increase the efficiency and output of central planning staff while making accuracy and traceability key.

The strategic move is to avoid presenting this as “chat with PDFs.” Present it as a controlled evidence workflow: the Research Agent searches approved sources, applies access controls before retrieval, reranks evidence, and produces a cited answer; the Reviewer Agent checks whether the citations support the answer's claims and returns a credibility score. The demo should prove authorized retrieval, hybrid retrieval plus rerank, cited synthesis, ambiguity handling, abstention, reviewer scoring, trace/audit, and an eval report rather than just answer text.

The Reviewer Agent story is captured in
`defence_agent/docs/trust_layer_eval_story.md`. The product contract is that
unsupported citations are not shown as trusted final-answer citations. They are
quarantined as reviewer findings in Trace and Eval, used to revise or refuse the
answer, and preserved as audit evidence.

## Source-Of-Truth Anchor

- The governing assignment brief is `source-materials/presentation-interview-instructions.md`.
- The original PDF is `source-materials/Nov 2025 - SA Presentation Interview Instructions (1).pdf`.
- The local job-description source is `source-materials/cohere-public-sector-sa-job-description.md`.
- The local role rubric is `defence_agent/docs/role_competency_frame.md`.
- Treat this file as interpretation of the brief, not independent truth.
- Treat only the PDF and the job description as authoritative competency sources.
- Treat all other competency lists as hypotheses until they map back to the PDF or job description.
- Current strategic filter: keep only the context, demo steps, slides, diagrams, code anchors, and evals that help the panel see technical SA competence.

## Active Ambition Rule

Always push the notebook and demo toward stronger competency proof.

When reviewing any idea, ask:

1. Which PDF or job-description competency does this strengthen?
2. Is this the most impressive way to show that competency?
3. Should it be live demo, Jupyter slide, code anchor, eval artifact, or Q&A?
4. What lower-signal context should be cut to make room?

Prefer ambitious ideas that demonstrate extreme competence over safe but generic completeness.

## Interview Competency Thesis

The presentation should be designed backward from what the interview is testing.

| Interview signal | What the presentation must prove |
| --- | --- |
| Technical expertise | Show real Cohere calls, retrieval, rerank, citations, trace, Reviewer Agent score, evals, and code anchors. |
| Applied customer solutioning | Tie every technical choice to central planning staff efficiency, accuracy, traceability, or security. |
| Scalable architecture | Show offline ingestion, bounded retrieval/rerank, deployment path, and operational metrics. |
| Security | Show authorization before generation and no restricted text in unauthorized model context. |
| Accuracy | Show refusal, Reviewer Agent score, eval metrics, and citation-support checks. |
| Traceability | Show citations, source pages, metadata, audit trace, and answer provenance. |
| Adaptability | Show document metadata, status/version thinking, ingestion path, and eval-driven improvement. |
| Stakeholder communication | Use the notebook to speak to executive, technical, security, and product concerns. |
| Job-description fit | Show hands-on demos, agentic AI/North mapping, Python/Jupyter, production deployment path, model customization judgment, pilots, and product feedback. |

The ambition is not to add more random features. The ambition is to make the notebook and demo prove the highest-value competencies cleanly.

## Cross-Scenario Signals From The PDF

The other PDF scenarios are relevant because they reveal what Cohere values.

| Scenario | Signal to borrow |
| --- | --- |
| Ecommerce | Relevance, personalization, secure use of customer data, measurable business outcome. |
| Healthcare | Confidentiality, integration, high-stakes accuracy. |
| Financial services | Transparency, compliance, adaptation to changing regulations. |
| Education | Personalization, privacy, adaptation to different standards. |
| Defence Agent | Accuracy, traceability, access control, document interrogation, staff efficiency. |

Use these signals to strengthen Defence Agent. Do not add separate scenario
digressions to the final presentation.

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

- Start from the interview goal: get the job by proving technical Solutions Architect competence.
- Start from the staff workflow, not model features.
- Demo the system before the technical walkthrough.
- Connect each technical choice back to accuracy, traceability, security, or
  staff efficiency.
- Connect each artifact back to a competency: build, agent design, architecture, security, evaluation, deployment, adaptability, or pilot delivery.
- Treat citations as a trust bridge, not a guarantee of truth.
- Keep production claims tied to concrete product paths: Cohere private
  deployment, Model Vault, Compass, or customer-approved infrastructure.
- Cut anything that does not make the Jupyter notebook or live demo stronger.
