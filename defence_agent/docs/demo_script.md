# Defence Agent Demo Script

## Opening Line

Defence Agent helps central planning staff interrogate doctrine, manuals, and
procedures with cited, access-controlled answers. The technical proof is a
  page-level normalized evidence stack: Cohere Embed v4 for multilingual page-image
search, Cohere Rerank v4 for evidence ordering, and persona-aware filters so
the model only sees authorized pages.

## Strategic Frame

- Customer problem: faster staff work without losing accuracy, source
  traceability, or access control.
- Trust: grounded answers, refusal when evidence is insufficient, and an audit
  trail showing which sources supported the answer.
- Security: authorization happens before generation. The model is not the
  security boundary.
- Privacy: this prototype uses local corpus storage and live Cohere APIs for
  speed; production would place prompts, embeddings, documents, and outputs
  inside the customer-approved data boundary.
- Data choice: normalize the demo corpus to rendered page evidence and use
  Cohere Embed v4's mixed text/image retrieval path instead of custom document
  parsing.

## Before The Demo

- Run `python -m pip install -e ".[dev]"`.
- Keep `.env` out of git.
- Confirm the real Cohere API key is set only in `.env` as
  `COHERE_API_KEY`.
- Prefer a trial key for rehearsal when limits are sufficient, with
  `COHERE_REQUESTS_PER_MINUTE=20` or lower.
- Rebuild the index with `python defence_agent/scripts/build_chroma_index.py`.
- Run the demo registry with `python defence_agent/scripts/run_demo_query_registry.py --no-fail` and mark any failing case as `do_not_demo`.
- Start the primary demo UI with `make run`, then open
  `http://127.0.0.1:8501`.
- Use `make run-adk` and `http://127.0.0.1:8502` only for technical
  inspection. ADK startup can run index checks and can trigger paid Cohere
  Embed calls when the local index is stale.
- Treat every rehearsal as a true agent run with true tool calls and true
  Cohere model calls.

## Flow 1: Agentic Planning Brief

- Persona: `clearance_unclassified`.
- Query: `I am preparing a planning brief. First search Canada's defence policy for AI-enabled modernization. Then search the DND/CAF AI Strategy for AI-enabled modernization. Compare both and cite the strongest source pages.`
- Show: one or more `search_documents` calls, retrieved policy/AI strategy pages, rerank scores, and citations.
- Show: `answer_audit.generation.citation_resolution.coverage` so the panel can see that claim-like answer sentences were cited.
- Point: agentic RAG handles a planning task that needs document comparison, not just a single vector lookup.
- CLI rehearsal:

```bash
python defence_agent/scripts/run_agent_session.py \
  "I am preparing a planning brief. First search Canada's defence policy for AI-enabled modernization. Then search the DND/CAF AI Strategy for AI-enabled modernization. Compare both and cite the strongest source pages." \
  --persona clearance_unclassified \
  --show-audit
```

## Flow 2: Language-Agnostic Retrieval

- Persona: `clearance_unclassified`.
- Query: `What core tasks does NATO assign to the Alliance in its Strategic Concept, and why do they matter for a Canadian planning brief?`
- Show: source language is selected by retrieval, not hard-coded by the user query.
- Point: the corpus can include English and French evidence; the user does not need to know the source language in advance.

## Flow 3: Permission-Aware Retrieval

- Persona A: `clearance_unclassified`.
- Persona B: `clearance_secret`.
- Query: `For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?`
- Show: unclassified persona says there is not enough authorized context; secret persona retrieves `SYN-FUSION-S-RELEASE-001` and answers from the restricted workflow.
- Point: identity maps to evidence access before generation.
- Show `answer_audit.retrieval.excluded_sources` for the unclassified persona.
- Show `answer_audit.citations` and `answer_audit.retrieval.sources_sent_to_answer` for the secret persona.
- Show `answer_audit.generation.citation_resolution`. If citation coverage
  fails, the app refuses rather than showing unsupported claim-like sentences.

## Flow 3b: Interrogate Doctrine Follow-Up

- Persona: `clearance_secret`.
- First query: `For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?`
- Follow-up: `Which source page supports that threshold, and what access level was required?`
- Show: the same session answers the source/access question from prior audit metadata and prints citation/source metadata through `--show-audit`.
- Point: the demo is not separate one-off searches; it supports doctrine interrogation over multiple turns.

```bash
python defence_agent/scripts/run_agent_session.py \
  "For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?" \
  --follow-up "Which source page supports that threshold, and what access level was required?" \
  --persona clearance_secret \
  --show-audit
```

## Flow 3c: Multi-Query Retrieval

- Persona: `clearance_unclassified`.
- Query: `For a planning update, find the current source pages for NATO's core tasks and Canada's AI strategy priorities, then summarize both.`
- Show: multiple `search_documents` calls in one turn when the model splits the task.
- Point: call this multi-query retrieval unless the trace proves actual concurrent execution.

## Flow 4: Top Secret Contrast

- Persona A: `clearance_secret`.
- Persona B: `clearance_top_secret`.
- Query: `What routing codeword opens the restricted relay path for the Fusion Model?`
- Show: secret persona says there is not enough authorized context; top-secret persona retrieves `SYN-FUSION-TS-ANNEX-002` and answers from the higher-tier annex.
- Point: the same retrieval stack supports tiered access without prompt-only guardrails.

## Flow 5: Insufficient Evidence Refusal

- Persona: `clearance_unclassified`.
- Query: `What does the corpus say about the approved Arctic submarine basing schedule for 2031?`
- Show: retrieval finds related public defence-policy pages but the answerability gate sends zero documents to final generation because the scheduled 2031 claim is unsupported.
- Point: citations alone are not enough. The workflow must abstain when authorized evidence does not support the user's claim.

## Close

- Show the public provenance section of `defence_agent/data/corpus/SOURCES.md`
  if source provenance comes up.
- Show `defence_agent/data/corpus/synthetic/README.md` if fictional restricted
  data comes up.
- Mention the eval YAML files if asked about readiness, but do not screen-share
  raw ACL answer facts during the live access-control contrast.
- Show one `answer_audit` payload from the CLI.
- State that the demo uses live Cohere calls for embeddings, reranking, and
  final cited answers.
- State that native `.docx` parsing is not implemented.
- State that the prototype includes one DOCX-origin source through the
  publisher's official PDF pair, proving that DOCX-origin and PDF-origin sources
  use the same Embed v4, Rerank v4, ACL, citation, and audit workflow after
  verified normalization.
- For production native DOCX ingestion, point to Compass or a
  manifest-compatible parser and normalization adapter.
