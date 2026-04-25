# Project Notes

## Canadian Defence Staff Decision Support Thesis

- DefTech wants an AI assistant to help staff interrogate manuals, procedures, and doctrine after moving PDFs and DOCX files into a central database.
- The Chief of Staff wants higher efficiency and output for central planning staff, with accuracy and traceability as explicit constraints.
- Treat "central planning staff" as a prompt assumption, not a verified user-research finding.
- For the presentation, interpret the primary users as Defence Team planning staff: headquarters-level staff, staff officers, analysts, and planners who prepare source-grounded materials for coordination and leadership review.
- The assistant should support staff work by helping users find, surface, cite, verify, extract, compare, flag, draft, and assemble materials from approved institutional documents.
- Outputs should be staff-review-ready, not final, self-authorizing, or a replacement for accountable human judgement.

## Defensible Working Assumption

- Central planning staff are likely not frontline operators; they are likely a headquarters or central-office team that coordinates across functions, prepares leadership decisions, and turns guidance into usable staff products.
- Their bottleneck is time spent finding current guidance, checking source currency, reconciling conflicting documents, drafting repeatable materials, and proving where each claim came from.
- The solution should therefore be framed as source-grounded staff-work support, not as a generic chatbot and not as autonomous planning.
- Review package assembly is an optional workflow, not the core product.

## Terminology: Decision Support Outputs

- **Planning work product**: a staff-review-ready artifact such as a brief, checklist, source pack, requirements matrix, options note, action tracker, decision log, coordination log, or one-off staff document.
- **Review package**: an optional, human-reviewed bundle assembled for coordination or leadership review. It may include one or more planning work products plus cited sources, assumptions, risks, open questions, owners, deadlines, and a change or decision log.
- **Planning package**: avoid as a primary term because it has a formal Earned Value Management / program-controls meaning for future scope that is identified and budgeted but not yet defined into work packages. Source: [NASA PP&C Glossary](https://www.nasa.gov/ocfo/ppc-corner/ppc-glossary/).
- Prefer "decision support outputs," "planning work products," "source-grounded staff products," and "staff-review-ready materials."

## Assistant Modes

| Mode | What the assistant helps with | Staff-review-ready output |
| --- | --- | --- |
| Source discovery | Find relevant manuals, procedures, doctrine, templates, and prior work from approved repositories | Source pack with document titles, links, metadata, and relevant passages |
| Source currency and claim verification | Check whether claims are cited, current, unsupported, contradicted, or based on stale guidance | Claim review with citations, uncertainty flags, and required follow-up |
| Requirements extraction | Extract required actions, tasks, constraints, approvals, roles, deadlines, dependencies, and risks | Requirements matrix or planning checklist |
| Document comparison | Compare current, superseded, draft, or conflicting guidance | Conflict log, change summary, and reviewer questions |
| Single-document drafting | Draft a brief, options note, checklist, agenda, tracker, or one-off staff document from approved templates and cited evidence | Source-grounded draft for human review |
| Existing-work-product review | Review an existing brief, tracker, checklist, or assembled materials against source guidance | Marked issues, missing citations, outdated claims, and open questions |
| Optional review-package assembly | Assemble selected work products and supporting evidence for coordination or leadership review | Optional review package for human validation |

## Guardrails And Non-Goals

- The assistant does not make decisions, approve plans, task units, replace staff judgement, or independently determine policy compliance.
- The assistant must not bypass classification, access control, records management, legal, policy, security, or human review processes.
- The assistant can surface candidates for authoritative sources, but source authority and currency depend on metadata quality, repository governance, and human validation.
- The assistant should cite sources, show provenance, flag uncertainty, and separate evidence-backed statements from user-provided assumptions.
- Avoid framing such as "autonomous decision-making," "single source of truth," "replace planners," "complete review package generation," or unqualified "decision-ready."
- Use "workflow assistance" rather than "workflow automation" unless discussing low-risk administrative steps such as routing, formatting, reminders, or checklist support.

## Canadian Defence Source Base

- DND/CAF's Data Strategy supports language around trusted data, data governance, stewardship, accessibility, security, quality, and information advantage. Source: [DND/CAF Data Strategy](https://www.canada.ca/en/department-national-defence/corporate/reports-publications/data-strategy/data-strategy.html).
- DND/CAF's AI Strategy supports an augmentation frame: AI can assist humans, but outputs must be assessed against expert judgement, organizational constraints, and AI limitations. Sources: [What is AI?](https://www.canada.ca/en/department-national-defence/corporate/reports-publications/dnd-caf-artificial-intelligence-strategy/what-is-ai.html), [Guiding Principles](https://www.canada.ca/en/department-national-defence/corporate/reports-publications/dnd-caf-artificial-intelligence-strategy/guiding-principles.html), [Ethics, Safety, and Trust](https://www.canada.ca/en/department-national-defence/corporate/reports-publications/dnd-caf-artificial-intelligence-strategy/line-of-effort-3.html).
- The CAF Digital Campaign Plan and Pan-Domain Command and Control concept support decision advantage, information advantage, interoperability, and human-machine teaming, but should be used as context for staff-work support rather than as proof of an operational command system. Sources: [CAF Digital Campaign Plan](https://www.canada.ca/en/department-national-defence/corporate/reports-publications/canadian-armed-forces-digital-campaign-plan.html), [PDC2 Concept Paper](https://www.canada.ca/en/department-national-defence/corporate/reports-publications/pan-domain-command-control.html).
- The 2026-27 Departmental Plan supports current language around responsible AI adoption, sovereign data stewardship, trusted data, metadata, improved search, and information that is prepared for operators and AI ingestion. Source: [DND/CAF 2026-27 Departmental Plan](https://www.canada.ca/en/department-national-defence/corporate/reports-publications/departmental-plans/departmental-plan-2026-27.html).
- DAOD 6500-1 is a useful anchor for access-controlled, governed, searchable, interoperable, and auditable data access for planning and decision support, but it should not be stretched into blanket approval for broad AI access. Source: [DAOD 6500-1, Data Access](https://www.canada.ca/en/department-national-defence/corporate/policies-standards/defence-administrative-orders-directives/6000-series/6500/6500-1-data-access.html).
- Vendor public-sector material should be secondary evidence for feasibility only: grounded retrieval, document analysis, secure or private deployment, auditability, usage monitoring, and human review support. Sources: [Cohere Public Sector](https://cohere.com/solutions/public-sector), [Cohere Private Deployments](https://cohere.com/private-deployments), [Google Public Sector AI Trends](https://cloud.google.com/blog/topics/public-sector/5-ai-trends-shaping-the-future-of-the-public-sector-in-2025), [Microsoft Government AI](https://www.microsoft.com/en-us/us-government/ai-for-government), [OpenAI Government](https://openai.com/solutions/industries/government/), [Anthropic Government](https://www.anthropic.com/solutions/government).

## Illustrative Task Emphasis

| Task bucket | Common pain | AI opportunity |
| --- | --- | --- |
| Source discovery and verification | Finding the right document, current version, relevant section, and supporting passage | Find, surface, cite, and rank relevant source material |
| Requirements extraction | Turning long guidance into actions, approvals, roles, deadlines, constraints, and risks | Extract structured requirements for staff validation |
| Comparison and assurance | Reconciling current, draft, outdated, or conflicting guidance | Compare, flag conflicts, and produce reviewer questions |
| Drafting support | Preparing briefs, options notes, checklists, agendas, trackers, and summaries | Draft source-grounded work products from approved templates |
| Coordination and follow-up | Tracking owners, missing inputs, open questions, and changes | Maintain action trackers, open-question logs, and change summaries |

## Stakeholders

- Defence Team planning staff are the assumed primary users for the presentation.
- The Chief of Staff is the executive sponsor in the prompt.
- Document owners, reviewers, IT, security, legal, policy, and records stakeholders determine whether the solution can be trusted and adopted.

## Planning Approach

- Start with the real staff workflow before defining AI features.
- Map recurring planning tasks, decisions, bottlenecks, handoffs, and information gaps.
- Derive each AI capability from a concrete user action: find, surface, cite, verify, extract, compare, flag, draft, or assemble for review.
- Position the AI system as governed workflow assistance for staff judgement, coordination, and traceability.
- Avoid starting from model features, generic automation, or a promise to generate complete packages.

## Core Framing

- The AI system should help Defence Team planning staff turn approved institutional documents into cited, reviewable, staff-review-ready planning work products.
- Basic Q&A is one interaction mode inside a broader staff-work workflow, not the whole product.
- Existing-work-product review and fact verification are first-class use cases because accuracy, source currency, and traceability are central to the scenario.
- Optional review-package assembly is useful only after source discovery, verification, extraction, comparison, and human review have established enough confidence.
