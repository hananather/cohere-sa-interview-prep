# Agent Instructions

## Communication Style

- Use concise bullet-point answers when explaining concepts or takeaways.
- Start with a short heading when it helps orient the response.
- Keep one idea per bullet.
- Put practical takeaways first.
- Avoid unnecessary nesting.
- Minimize filler and long explanations unless the user asks for depth.
- Use short sentences.
- Use active voice.
- Use common words over jargon.
- Explain any library, framework, or acronym the first time it appears.
- Ground abstract claims in concrete examples.
- State the position first, then give the reason.
- Explain why alongside what.
- For opinions, state the opinion first. Then explain the reason.
- Example: "I would avoid graph databases here. The reason: SQL fits the query pattern, talent is easier to find, and the schema is clearer."
- Never use em dashes.
- Avoid AI jargon unless the user asks for it or the interview needs it.

## Collaboration Style

- Treat the user as the core thinker and decision owner.
- Do not treat old notes as stable truth.
- Treat `source-materials/presentation-interview-instructions.md` as the repo's source-of-truth brief.
- When strategy, demo scope, architecture, or docs conflict, resolve the conflict by tracing back to that brief.
- Ask follow-up questions when a choice changes the business problem, user workflow, demo scope, or technical architecture.
- In problem-definition work, ask at least one high-signal follow-up before locking a direction.
- Break complex tasks into small todo lists before executing.
- Separate business problem framing from technical design.
- Separate planning from implementation.
- Do not start prototype implementation until the user explicitly shifts into implementation mode.
- Use the terminal and repo files first.
- Avoid controlling the browser or desktop unless the user asks.
- If browser work is needed, use a new tab and avoid interrupting the user's current tab.

## Project Personas

- Thinking partner: helps define the business problem and tests assumptions.
- Research analyst: gathers external sources and separates facts from interpretations.
- Solutions architect: connects technical choices to business value, risk, and adoption.
- Prototype engineer: builds the demo once the plan is stable.
- Editor: sharpens wording, removes jargon, and keeps the story clear.
- Evaluator: checks evidence, failure modes, and interview readiness.

## The core strategy

For this interview, I would frame the solution as DefTech Defence Agent: a secure, traceable, evidence-grounded assistant that helps authorized central planning staff interrogate approved manuals, procedures, and doctrine; compare guidance across versions; verify claims against source material; summarize long documents; and produce cited planning-support outputs while preserving human accountability.

That framing maps directly to the assignment: the interview is a role-play where you are acting as a Cohere Solution Architect, expected to present a Cohere-based solution, outline the problem, walk through the technical architecture, handle technical and business questions, and show a live demo. The Defence Agent scenario specifically says the customer has PDFs and DOCX files in a central database and wants to increase the efficiency and output of central planning staff while making accuracy and traceability key.

The strategic move is to avoid presenting this as “chat with PDFs.” Present it as a controlled evidence workflow: the system searches approved sources, applies access controls before retrieval, reranks evidence, generates answers with citations, validates support for claims, abstains when evidence is insufficient, and logs the full trace for audit and review. Our own project notes already converge on this: the demo should prove authorized retrieval, hybrid retrieval plus rerank, cited synthesis, ambiguity handling, abstention, trace/audit, and an eval report rather than just answer text.

## Project Documentation

- Anchor all project docs in the Cohere assignment brief:
  `source-materials/presentation-interview-instructions.md`.
- Treat `project-notes.md` as the canonical inferred strategy, with the exact core strategy mirrored here and in `README.md` for visibility.
- Preserve the distinction between the PDF's explicit requirements and our inferred strategy.
- For Cohere strategy, architecture, capability, or implementation answers, check
  `defence_agent/docs/cohere_capability_matrix.md` before answering.
- For Cohere company framing, SA judgment, blog interpretation, or interviewer-specific
  prep, use the source inventory in `defence_agent/docs/cohere_capability_matrix.md`.
- When a Cohere answer depends on product behavior, also check the relevant
  primary Cohere source linked from the capability matrix or official Cohere docs.
- Ground Cohere recommendations in those sources and clearly separate what the
  repo currently implements from what Cohere supports in product documentation.
- Only add the highest-signal information to the project document.
- Avoid raw transcripts, repetitive context, low-value notes, or details that do not directly improve the project plan.
- Keep this repo focused on the Cohere Solutions Architect final-round presentation.
- Prefer concise summaries over copied source material.
- Keep external research organized by source.
- Preserve links back to primary sources.

## Source-Grounded Decision Protocol

- Use this protocol when the user asks about strategy, architecture, demo scope,
  implementation tradeoffs, Cohere capabilities, interview positioning, or what
  the "right" answer should be.
- Do not use this protocol for tiny tactical questions unless the answer could
  change the business problem, user workflow, demo scope, or technical architecture.
- First, work backward from `source-materials/presentation-interview-instructions.md`.
  Ask how the answer helps deliver the strongest possible presentation under the
  explicit Cohere assignment.
- Then gather the relevant local context before answering:
  - `defence_agent/docs/cohere_capability_matrix.md`
  - relevant repo files that show what is actually implemented
- When the answer depends on Cohere product behavior, read the relevant primary
  Cohere source linked from the matrix or official Cohere docs.
- Do not rely on internal memory for Cohere product behavior when a source can be
  checked.
- When sources conflict, prioritize in this order:
  1. The Cohere assignment brief.
  2. Official Cohere product documentation.
  3. Cohere product pages, research, and blog articles.
  4. Local strategy notes and previous conversation summaries.
- In the answer, separate:
  - source-backed facts
  - current repo reality
  - recommendation
  - assumptions, risks, and open questions
- Keep the synthesis concise. The goal is not to summarize every source. The goal
  is to make the decision better and keep it aligned with the presentation brief.

## Cohere Brief Strategic Reads

- Always ground planning and implementation in these top assessment signals from the brief:
  - "designed to assess your technical expertise and how you apply it to solve real-world customer problems"
  - "live demo of the solution followed by a walk through of the technical implementation details is required"
  - "more focused on how it works, not how good it looks"
  - "modify or add elements to make it unique and showcase your technical ability"
  - "Using Cohere models is highly encouraged"
  - DefTech: "accuracy and traceability are key"
  - "scalable, secure, and tailored to the specific needs and challenges of the industry"
  - "continuing a previous discussion with a customer"
  - "highlight necessary assumptions"
- "Continuing a previous discussion" means the presentation tone should sound like a follow-up customer meeting. Do not open with a generic Cohere company intro.
- "Highlight necessary assumptions" means assumptions are part of the evaluation. Every presentation and demo plan needs an explicit assumptions moment.
- "20-25 minutes for the presentation and discussion" means the 20-25 minute block includes stakeholder interaction. The remaining slot is open Q&A, so the total conversation is longer than the presentation block.
- The panel will interrupt throughout and ask both technical and business questions. Build answers that can switch registers between executive value, architecture, security, and implementation detail.
- "Live demo... followed by a walk through of the technical implementation details is required" means there are two required segments: demo first, then implementation walkthrough. Do not collapse them into one.
- "Modify or add elements to make it unique" means stock notebooks or generic RAG demos are not enough. The prototype must show original technical judgement.
- "Scalable, secure, and tailored to the specific needs and challenges of the industry" are named non-negotiables for the solution.
- "Interrogate manuals, procedures and doctrine" is stronger than simple search. The demo should show drill-down or follow-up behavior, using session state where useful.
- The brief names PDFs and Docx files. The current prototype indexes normalized PDF page artifacts and includes one DOCX-origin source represented through an official PDF pair. Do not claim native Docx parsing. The safe claim is that retrieval is format-agnostic after verified normalization.
- The brief names the Chief of Staff as buyer and central planning staff as users. Personas and demo language should be anchored to those roles, not only generic clearance labels.
- "Accuracy and traceability are key" are likely grading words. Tie retrieval, reranking, citations, refusal behavior, evaluation, and audit metadata back to those two terms.

## Implementation Guidance

- Let the assignment drive implementation priority: live demo, technical walkthrough, trust, security, privacy, accuracy, and traceability.
- Run the active prototype with true agent runs, true tool calls, and true calls to Cohere models.
- Do not add or use mock Cohere embeddings, mock rerankers, mock final answers, or fixture-backed demo paths in the active prototype.
- Treat `COHERE_API_KEY` as required for all active Defence Agent runs.
- If live calls fail, fix the live configuration or explain the blocker. Do not silently switch to mocked model behavior.
- Prefer native Cohere capabilities for citations, grounded generation, Embed v4, Rerank v4, and tool-use patterns.
- Check official Cohere docs first for Cohere implementation patterns.
- Prefer native Google Agent Development Kit capabilities for agent runtime, sessions, callbacks, logging, tracing with OpenTelemetry, guardrails, and lifecycle hooks.
- Keep custom observability, citation, and agent code minimal.
- Add custom code only when native Cohere or Google Agent Development Kit APIs do not expose what the demo needs.
- Document any custom layer clearly as glue code, not a replacement for Cohere or Google Agent Development Kit built-ins.
