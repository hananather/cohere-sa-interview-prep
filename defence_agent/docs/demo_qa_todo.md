# Defence Agent Demo QA Tracker

This is the working tracker for Defence Agent demo review. Treat browser comments as raw signals, then convert them into source-backed implementation asks.

## Status

- Working artifact: repo markdown tracker.
- Phase: implementation pass completed on 2026-05-06.
- Code changes in this phase: synthetic corpus, canonical demo queries, route vocabulary, metadata filters, source drilldown, sandboxed table analysis, trace JSON files, eval harness, and docs were updated.
- Primary goal: keep this tracker as the rationale/backlog record for future UI and demo refinements.

## Source Inventory

Use these labels when adding or reviewing issues.

| Label | Source | Relevant guidance |
| --- | --- | --- |
| `G1` | Google Doc `Cohere Project`, `Tab 1` | Define the business problem first, document assumptions, validate whether tasks are fixed workflow steps or open-ended, and reason about latency, cost, and model calls. |
| `G2` | Google Doc `Cohere Project`, `Tab 1` | Public-sector trust needs access controls, inline source citations, audit trails, privacy boundaries, and data sovereignty. |
| `G3` | Google Doc `Cohere Project`, `Tab 1` | Build the evaluation set before adding complexity. Include answerable, unanswerable, ambiguous, permission-sensitive, hard, adversarial, multi-hop, and time-sensitive cases. |
| `G4` | Google Doc `Cohere Project`, `Tab 1` | The product should reduce time to verifiable answers, improve answer quality, and reduce manual document work by task bucket. |
| `G5` | Google Doc `Cohere Project`, `Todo` tab | Synthetic documents and representative queries should be designed together so the queries are answerable and include hard cases. |
| `G6` | Older Google Doc `Cohere` | The interview requires a live demo, assumptions, technical architecture, and a solution that is scalable, secure, and tailored to the public-sector scenario. |
| `L1` | `project-notes.md` | Frame the product as source-grounded staff-work support, not autonomous planning or a generic chatbot. |
| `L2` | `project-notes.md` | First-class modes are source discovery, claim verification, requirements extraction, document comparison, drafting support, work-product review, and optional review-package assembly. |
| `L3` | `project-notes.md` | Outputs should be staff-review-ready and human-reviewable. The assistant must not make decisions, approve plans, task units, or bypass access controls. |
| `D1` | `defence_agent/docs/architecture.md` | Current architecture is Streamlit + FastAPI, direct Cohere `ClientV2`, router, workflows, tools, ACL-filtered retrieval, rerank, cited generation, traces, and evals. |
| `D2` | `defence_agent/docs/evaluation.md` | Current eval metrics include route accuracy, tool accuracy, retrieval recall, context precision, citation validation, permission correctness, abstention, safety, latency, and errors. |
| `D3` | `defence_agent/docs/demo_script.md` | Current live demo has five flows: direct RAG, version comparison, table analysis, permission-aware retrieval, and prompt-injection defense. |
| `P1` | Current prototype | Corpus now has canonical Markdown, generated PDF/DOCX artifacts, and `doctrine_review_tracker.csv` under `defence_agent/data/synthetic_corpus`. |
| `P2` | Current prototype | Router/workflows now expose the canonical routes: `evidence_lookup`, `grounded_summary`, `metadata_aware_retrieval`, `cross_source_synthesis`, `version_comparison`, `structured_table_analysis`, `claim_verification`, `permission_sensitive_retrieval`, `bilingual_retrieval`, `refuse_or_clarify`. |
| `P3` | Current prototype | Golden dataset has 24 passing fixture cases across task families and risk conditions. |
| `P4` | Current prototype | User View now hides the stream toggle, moves route override under advanced controls, uses `Inspect evidence`, and uses compact feedback controls. Remaining UI polish should be validated visually before the interview. |

## Product Principles

- The prototype is a pilot-validation artifact, not a final production product.
- User View should feel like a polished customer-facing assistant.
- Demo Console should prove how it works: route, plan, trace, tools, candidates, rerank scores, evals, corpus, security, latency, cost, and degradations.
- The LLM is not the access-control layer. Retrieval must be permission-aware before model context.
- Simple lookup can cite one source. Comparison, synthesis, stale/conflict, and multi-hop tasks should cite multiple sources when multiple sources support material claims.
- Retrieved candidates are not the same thing as cited evidence. User View should show cited evidence; Demo Console can show candidates.
- The demo should make staff-work workflows concrete: find, answer, summarize, synthesize, compare, verify, calculate, clarify/refuse, and escalate.

## Cross-Reference Matrix

### 1. Demo Query Taxonomy And Distribution

- Priority: `P0`.
- Captured issue: the current demo queries do not feel right, and the app should work backward from the eight task families and query distribution in the notes.
- What you meant: the demo should show that the architecture follows the real workload shape, not five scripted examples.
- Source alignment: `G1`, `G3`, `G4`, `G5`, `L1`, `L2`, `D3`.
- Current prototype gap: the sidebar has five demo query buttons, while the notes discuss broader task buckets and hard cases. The eval file has 24 cases, but not a clean task-family/risk-condition contract.
- Decision: define a two-axis demo/eval taxonomy:
  - `task_family`: direct factual lookup, procedural checklist/basic answer, summarization or brief drafting, multi-document synthesis, old-vs-new comparison, table/calculation analysis, multi-hop cross-reference, registry/status lookup if retained.
  - `risk_condition`: ambiguous, unanswerable, permission-sensitive, prompt-injection/adversarial, stale/conflicting source, high-risk human review, restricted metadata leakage.
- Acceptance checks:
  - Each sidebar demo scenario maps to one task family.
  - Each golden eval maps to `task_family` and `risk_condition`.
  - Demo Console shows expected route versus selected route.
  - Simple lookup stays fast and does not over-orchestrate.
  - Complex queries visibly use comparison, table analysis, multi-source retrieval, or human-review behavior.

### 2. User View Versus Demo Console Ownership

- Priority: `P0`.
- Captured issue: technical details and product UI have been mixed together, making User View feel cluttered and confusing.
- What you meant: non-technical stakeholders should see a great product. Technical stakeholders should inspect the system from a separate control plane.
- Source alignment: `G2`, `G6`, `L1`, `D1`, `D2`.
- Current prototype gap: User View has improved, but sidebar and labels still expose implementation details such as route override, model mode, and streaming toggle.
- Decision:
  - User View owns: question, answer, inline citations, cited evidence cards, evidence drilldown, compact feedback, and polished progress.
  - Demo Console owns: route override, runtime mode, trace ID, source count, latency, token/cost, degradations, retrieved candidates, rerank scores, safety events, corpus, evals, and security controls.
- Acceptance checks:
  - User View has no route, trace, token, cost, rerank, degradation, or model-debug clutter.
  - Demo Console contains every technical detail needed for the implementation walkthrough.
  - Sidebar controls are presentation controls only, or they are moved under Demo Console.

### 3. Citation Logic And Evidence Drilldown

- Priority: `P0`.
- Captured issue: answers often show only `C1`, while multiple sources appear elsewhere. This feels logically inconsistent and undermines the evidence story.
- What you meant: citations should behave like product-grade evidence references. If only one citation is inline, only that cited source should appear in User View. If a task requires multiple documents, the answer should cite multiple documents inline.
- Source alignment: `G2`, `G3`, `G4`, `L1`, `L3`, `D1`, `D2`.
- Current prototype gap: User View now filters to inline-cited sources, which is correct, but the corpus/workflows/generation often produce only one inline citation. That exposes a deeper coverage issue for synthesis and comparison tasks.
- Decision:
  - User View source cards represent final cited evidence only.
  - Demo Console retrieved-candidate lists represent broader retrieval and rerank context.
  - Use multiple inline citations when the answer makes claims from multiple documents.
  - Keep one citation for true single-source lookups.
  - Rename `Open source` to `Inspect evidence` or `View cited passage`.
  - Source drilldown should show the cited chunk, nearby chunks, table markdown where present, metadata, and authorized document access.
- Acceptance checks:
  - Direct lookup can show one inline citation and one source card.
  - Version comparison cites both 2024 and 2025 sources.
  - Multi-document synthesis cites at least two sources.
  - Prompt-injection test cites the test document only as untrusted evidence.
  - Source cards match inline citation chips exactly in User View.
  - Unauthorized users cannot inspect restricted sources or infer restricted source names.

### 4. Persona Permissions And Side-By-Side Access Demo

- Priority: `P0`.
- Captured issue: permissions are too abstract. The demo should visually show that different personas have different source universes.
- What you meant: identity-aware retrieval is a core proof point, and it should be concrete enough to demo without a long explanation.
- Source alignment: `G2`, `G3`, `L3`, `D1`, `D2`.
- Current prototype gap: the app has persona selection and permission-sensitive behavior, but no dedicated access matrix or side-by-side comparison view.
- Decision:
  - Add a presenter-facing access view later that shows persona, role, clearance, groups, allowed classifications, accessible source counts, blocked source counts, tool allowlist, and ACL filter.
  - Add a side-by-side persona comparison view later with the same query run for `planning_analyst` and `planning_lead`.
  - Replace the weak restricted query wording with a natural query that does not name restricted documents.
- Candidate side-by-side query:
  - `What is the current exception-handling guidance for planning requests that need approval outside the standard review path?`
- Expected behavior:
  - Analyst answer uses only authorized public/internal/protected evidence and does not name restricted documents.
  - Lead answer can cite restricted evidence when authorized.
- Acceptance checks:
  - Same query produces visibly different source sets by persona.
  - Unauthorized persona does not see restricted titles, IDs, snippets, blocked-source counts, tool args, trace fields, or progress details in User View.
  - Demo Console can explain the ACL decision without leaking restricted content in the wrong view.

### 5. Corpus Depth And Evaluation Coverage

- Priority: `P1`, with P0 dependency on taxonomy.
- Captured issue: the current corpus feels too small, and the evaluation suite does not yet make the use-case coverage obvious.
- What you meant: add documents only when they support specific demo/eval cases. Do not add filler just to increase counts.
- Source alignment: `G3`, `G5`, `L2`, `D2`, `P1`, `P3`.
- Current prototype gap: corpus has 7 documents. The eval suite has 24 cases, but slices do not map cleanly to the desired two-axis demo/eval taxonomy.
- Decision:
  - Keep the current 7 documents as the base.
  - Add targeted synthetic documents only for missing proof points: multi-hop cross-reference, stale/current conflict, source authority, registry/status, briefing template, scanned/OCR-like page, and optional bilingual variant.
  - Update evals to include `task_family`, `risk_condition`, expected source documents/passages, expected answer behavior, expected citation pattern, and persona.
- Acceptance checks:
  - Corpus tab looks credible because each document has a demo/eval purpose.
  - Eval tab reports by task family and risk condition.
  - Every P0 demo scene has at least one golden case.
  - Permission, safety, citation, and human-review failures are not hidden by aggregate pass rate.

### 6. Streaming And Async Presentation

- Priority: `P1`.
- Captured issue: `Stream answer` appears to do nothing, and answers appear instantly even when streaming is enabled.
- What you meant: streaming should be the default product behavior, not a toggle. The app should present quick tasks and longer research tasks differently.
- Source alignment: `G1`, `G4`, `G6`, `D1`, `P4`.
- Current prototype gap: streaming endpoint exists, but Streamlit does not make the progressive answer experience visible. The toggle feels like a dead control.
- Decision:
  - Remove `Stream answer` from User View/sidebar later.
  - Always use a polished staged ask flow.
  - Use brief stages for fast tasks and richer progress for medium/long tasks.
  - Do not fake long latency for simple queries.
  - Reserve deep-research/report UX for longer synthesis, multi-hop, stale/conflict, or brief-drafting tasks.
- Acceptance checks:
  - No visible streaming toggle remains.
  - Ask flow always shows meaningful progress.
  - Direct lookup remains fast.
  - Version comparison, table analysis, and synthesis show tool/retrieval stages.
  - Long-running report output is clearly labeled as a research/report workflow, not a normal lookup.

### 7. Feedback And Control Labels

- Priority: `P1`.
- Captured issue: feedback buttons wrap badly, and labels like `Mock Cohere`, `Route override`, and `Open source` are confusing.
- What you meant: every visible control should help the presenter tell the story. Debug controls need a reason and a home.
- Source alignment: `G6`, `L1`, `D1`, `P4`.
- Current prototype gap: `Open source`, `Route override`, `Stream answer`, `Mock Cohere`, and long feedback labels still appear in the frontend code.
- Decision:
  - Replace long feedback buttons with compact thumbs up/down controls beside the answer.
  - Move `Mock Cohere`/model mode to Demo Console and label it as runtime mode.
  - Move `Route override` to advanced Demo Console controls and rename it `Force workflow` if retained.
  - Rename `Open source` to `Inspect evidence` or `View cited passage`.
- Acceptance checks:
  - Feedback does not wrap vertically.
  - User View has no large standalone feedback section.
  - Runtime/model mode is not prominent in the product demo.
  - Route override does not undermine automatic routing.
  - Evidence-inspection wording is clear to non-technical stakeholders.

### 8. Architecture And Eval Consistency

- Priority: `P0`.
- Captured issue: the prototype needs to prove the architecture thesis systematically, not just show isolated features.
- What you meant: routes, tools, eval slices, corpus documents, UI views, and talk track should tell the same story.
- Source alignment: `G1`, `G2`, `G3`, `G4`, `L1`, `L2`, `L3`, `D1`, `D2`, `D3`.
- Current prototype gap:
  - Backend routes exist, but they do not map one-to-one to the broader task-family story.
  - Current demo script has five flows, not the full query distribution.
  - Eval slices are useful but not yet a demo/eval contract.
  - Registry/status exists as a tool/API lane but is not clearly framed as a task family or optional production pattern.
- Decision:
  - Do not force all eight task families to become backend routes.
  - Treat task families as demo/eval categories first.
  - Keep current routes unless implementation evidence shows a new route is necessary.
  - Use risk conditions across task families rather than turning every risk into its own product category.
  - Make every talk-track claim either visible in the app, proven in evals, explained as prototype scope, or removed.
- Acceptance checks:
  - Demo script, sidebar scenarios, eval dataset, and architecture docs use the same terms.
  - `restricted_access`, `security_test`, and `human_review` are explained as governed workflows/risk handling, not just normal task families.
  - Registry/status lookup is either implemented as a clear read-only tool demo or deferred from the live story.

## Demo Contract

These are the recommended live scenes after the tracker items are implemented. They are intentionally fewer than the full eval set.

| Scene | Audience value | Task family | Risk condition | Persona | Query | What it proves |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Fast cited answer | Planning analyst sees immediate value | Procedural checklist/basic answer | None | `planning_analyst` | `What review steps should planning staff complete before approving a cross-unit planning request?` | Fast grounded answer with inline citation and evidence inspection. |
| 2. Version comparison | Doctrine steward sees change awareness | Old-vs-new comparison | Stale/source currency | `planning_lead` | `Compare the 2024 and 2025 review gate procedure. What changed and what is the impact?` | Agentic decomposition, old/new retrieval, comparison table, citations to both versions. |
| 3. Table analysis | Analyst sees tool use | Table/calculation analysis | None | `planning_analyst` | `Using the readiness review table, which units fall below the 80% readiness threshold?` | Sandboxed Python over authorized table data with cited source table. |
| 4. Persona access comparison | Security stakeholder sees access control | Direct answer or synthesis | Permission-sensitive | `planning_analyst` vs `planning_lead` | `What is the current exception-handling guidance for planning requests that need approval outside the standard review path?` | Same query, different authorized evidence, no restricted metadata leakage. |
| 5. Prompt-injection defense | Governance stakeholder sees safety | Summarization | Prompt injection/adversarial | `planning_lead` | `Summarize the exception handling guidance from the test document.` | Retrieved text is untrusted evidence, not instructions. |
| Optional 6. Multi-hop/stale source | Technical panel sees why agentic RAG matters | Multi-hop cross-reference | Stale/conflicting source | `planning_lead` | `What procedure applies if a planning brief cites a superseded manual?` | Follow-up retrieval, source chain, current-versus-superseded handling. |

## Final Backlog

### P0: Must Resolve Before The Demo Story Is Trusted

- Finalize the two-axis taxonomy: `task_family` plus `risk_condition`.
- Rewrite demo queries and labels around the demo contract.
- Update eval metadata so every case maps to the taxonomy and expected evidence.
- Define User View versus Demo Console ownership and remove leakage between them.
- Define persona authorization and restricted metadata non-disclosure rules.
- Fix citation semantics: cited evidence versus retrieved candidates.
- Decide whether registry/status lookup is in the live story or deferred.
- Resolve high-risk/human-review eval contradictions before using eval pass rate in the presentation.

### P1: Should Fix For A Polished Interview Demo

- Add targeted corpus documents for missing proof points.
- Improve source drilldown and rename source buttons to evidence-oriented language.
- Make staged/streaming progress default and remove the streaming toggle.
- Replace feedback section with compact thumbs up/down controls.
- Add side-by-side persona comparison view.
- Add access matrix/security lens view.
- Improve Evaluation tab by task family, risk condition, and architecture stage.
- Move runtime/model mode and route override into Demo Console.

### P2: Useful If Time Allows

- Add richer deep-research/report workflow for long-running synthesis.
- Add row/column filtering visualization for synthetic tables.
- Add bilingual English/French coverage if it strengthens the interview story.
- Add workload distribution chart showing why simple and agentic paths coexist.
- Expand corpus beyond targeted proof points only after P0/P1 are stable.

## Validation Rules For Future Updates

- Every new issue must map to at least one source label or be marked `New idea`.
- Every P0 item must include a concrete browser QA check.
- Every eval-related item must map to `task_family` or `risk_condition`.
- Every implementation item must state whether it belongs in User View, Demo Console, backend, corpus, evals, or docs.
- Any disagreement between the Cohere notes and the current prototype must be called out explicitly.
- Do not implement from this tracker unless the user explicitly asks to move from capture/review into implementation.

## Known Current-State Disagreements

- Notes say the eval set should represent the workload before complexity is added, but the current eval file lacks explicit `task_family` and `risk_condition` fields.
- Notes emphasize source discovery, verification, extraction, comparison, drafting, and work-product review, but the live demo currently emphasizes Q&A, comparison, table analysis, permissions, and prompt injection.
- Notes emphasize access controls and auditability, but User View does not yet have a dedicated persona access matrix or side-by-side access comparison.
- Notes emphasize inline citations and inspectable sources, but current citation behavior still needs multi-document citation coverage for tasks that require synthesis or comparison.
- Notes treat the prototype as a workflow-validation artifact, but some UI controls still look like developer/debug switches rather than presenter-friendly demo controls.

## Regression Checks To Preserve

- User View shows only cited evidence, not all retrieved candidates.
- Demo Console shows broader retrieved candidates and technical scores.
- Dark mode is forced and does not depend on system theme.
- Header and top spacing are not clipped at the in-app browser size.
- User View does not show route/trace/latency metric cards, token/cost lines, persistent success banners, or degradation banners.
- The direct RAG flow still returns a trace ID in the backend response.
- Restricted content never enters unauthorized model context.
