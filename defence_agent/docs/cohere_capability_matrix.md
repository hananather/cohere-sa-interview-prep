# Cohere Capability Matrix

Living artifact for the Defence Agent SA presentation. Use this file to decide what to add, what to mention in Q&A, and what to ignore.

Last updated: 2026-05-14

## Source Inventory

These are the primary Cohere sources checked so far. Re-check them before the interview because Cohere ships quickly.
The 2026-05-14 spot check reconfirmed the active prototype's use of Chat `documents=`,
native citation objects, Embed v4, Rerank v4, and Command A `command-a-03-2025`.
Command A remains the default demo model because Cohere's agentic RAG tool-use
tutorials use it for tool planning and tool calls. Command A Reasoning is
supported as an optional comparison path for returned `thinking` content blocks,
but thinking blocks are not the same thing as application audit trace.

Use the source inventory before answering Cohere product, architecture, or Q&A questions. Prefer primary Cohere docs, product pages, research reports, and status pages over memory or generic web summaries.

Use this file as the single Cohere source inventory. Product behavior should be
verified against official Cohere docs before implementation claims; blog posts
are used for customer framing and Q&A language.

### Tier 1

| Source | URL | Why it matters |
|---|---|---|
| Cohere docs AI index | https://docs.cohere.com/llms.txt | Agent-readable docs index. Use this to find current Cohere docs quickly. |
| RAG citations | https://docs.cohere.com/docs/rag-citations | Native answer spans, source IDs, accurate vs fast citation modes. |
| RAG with Cohere | https://docs.cohere.com/docs/rag-with-cohere | End-to-end retrieve, embed, rerank, answer, cite pattern. |
| Chat API reference | https://docs.cohere.com/reference/chat | `documents`, citation behavior, safety mode caveats, limits, and response shape. |
| Agentic RAG for PDFs with mixed data | https://docs.cohere.com/page/agentic-rag-mixed-data | Closest official cookbook to the Defence Agent corpus: PDFs with tables/text, RAG tools, and follow-up handling. |
| Agentic RAG | https://docs.cohere.com/v2/docs/agentic-rag | Cohere's pattern for tool-using RAG across multiple sources and turns. |
| Agentic multi-stage RAG | https://docs.cohere.com/page/agentic-multi-stage-rag | More advanced pattern for staged retrieval and tool-based reasoning. |
| Tool use overview | https://docs.cohere.com/v2/docs/tool-use-overview | Native Cohere tool-use protocol and cited tool results. |
| Multi-step tool use | https://docs.cohere.com/v2/docs/multi-step-tool-use | Sequential and multi-step agent behavior. |
| Embed v4 launch | https://docs.cohere.com/changelog/embed-multimodal-v4 | Multimodal embeddings, 128k context, Matryoshka dimensions. |
| Embed API reference | https://docs.cohere.com/v2/reference/embed | Input limits, `inputs`, image data URI support, `output_dimension`. |
| Command A | https://docs.cohere.com/v1/docs/command-a | Current generation model for RAG, agents, tool use, multilingual tasks. |
| Command A technical report | https://cohere.com/research/papers/command-a-technical-report.pdf | Deep source for Command A enterprise, RAG, grounding, tool-use, and benchmark claims. |
| Command A Reasoning | https://docs.cohere.com/v2/docs/reasoning | Reasoning model with returned `thinking` content blocks. Useful for comparison and harder reasoning-heavy tasks, not the default demo backend. |
| Aya Expanse docs | https://docs.cohere.com/v2/docs/aya-expanse | Multilingual generation option for benchmarking. |
| Aya research | https://cohere.com/research/aya | Cohere's broader multilingual research story. |
| Aya Expanse blog | https://cohere.com/blog/aya-expanse-connecting-our-world | Launch narrative for multilingual AI. |
| Deployment overview | https://docs.cohere.com/docs/deployment-options-overview | Platform, cloud AI services, private cloud, and on-prem options. |
| Private deployment | https://docs.cohere.com/docs/private-deployment-overview | VPC and on-prem answer for defence data boundaries. |
| Private deployment usage | https://docs.cohere.com/v2/docs/private-deployment-usage | Shows the SDK change needed to point the same code at a private deployment. |
| Model Vault | https://docs.cohere.com/docs/model-vault | Cohere-managed single-tenant inference option with isolated serving. |
| Cohere Trust Center | https://trustcenter.cohere.com/ | Compliance, security, and audit artifact source for trust/security/privacy Q&A. |
| Cohere security page | https://cohere.com/security | Cohere's enterprise security and data protection positioning. |
| Cohere privacy policy | https://cohere.com/privacy | Source for data handling and privacy statements. |

### Tier 2

| Source | URL | Why it matters |
|---|---|---|
| RAG evaluation deep dive | https://docs.cohere.com/page/rag-evaluation-deep-dive | Retrieval precision/recall/MAP and generation faithfulness/correctness/coverage framing. |
| Retrieval eval with LLM-as-judge | https://docs.cohere.com/page/retrieval-eval-pydantic-ai | Practical retrieval evaluation pattern for document-grounded systems. |
| Advanced document parsing | https://docs.cohere.com/page/document-parsing-for-enterprises | Parsing and chunking strategy for enterprise PDFs with mixed structure. |
| PDF extractor with native multi-step tool use | https://docs.cohere.com/page/pdf-extractor | PDF extraction and validation pattern using Cohere tool use. |
| Rerank v4 launch | https://docs.cohere.com/changelog/rerank-v4.0 | Rerank v4 variants, multilingual support, JSON support, 32k context. |
| Rerank overview | https://docs.cohere.com/docs/rerank | Current model list and `rerank-v4.0-pro` vs `rerank-v4.0-fast`. |
| Reranking tutorial | https://docs.cohere.com/docs/reranking-with-cohere | YAML pattern for semi-structured rerank input. |
| Structured outputs | https://docs.cohere.com/v2/docs/structured-outputs | JSON schema mode and `strict_tools`; important RAG limitation. |
| Safety modes | https://docs.cohere.com/v2/docs/safety-modes | `CONTEXTUAL` vs `STRICT`; Command A RAG/tool mode constraint. |
| Embed Jobs guide | https://docs.cohere.com/v1/docs/embed-jobs-api | Batch embedding story for production scale. |
| Embed Job reference | https://docs.cohere.com/v1/reference/create-embed-job | API reference for async embedding jobs. |
| Datasets API | https://docs.cohere.com/v2/docs/datasets | Dataset lifecycle and metadata support for Embed Jobs and batch workflows. |
| Batches API | https://docs.cohere.com/reference/create-batch | Batch request execution for offline or large-scale jobs. |
| Rate limits | https://docs.cohere.com/v2/docs/rate-limits | Account throughput, important for demo-day risk and scaling Q&A. |
| Going Live | https://docs.cohere.com/v1/docs/going-live | Real-key setup, sensitive-use review, and status subscription guidance. |
| Deprecations | https://docs.cohere.com/docs/deprecations | Model and endpoint lifecycle checks before presenting or shipping. |
| SDK cloud platform compatibility | https://docs.cohere.com/v2/docs/cohere-works-everywhere | Portability across Cohere platform, cloud AI services, and private deployment. |
| OpenAI SDK compatibility | https://docs.cohere.com/docs/compatibility-api | Migration path for teams already using OpenAI-compatible clients. |
| Usage policy | https://docs.cohere.com/docs/usage-policy | Responsible-use boundary for customer and security questions. |
| Command R/R+ model card | https://docs.cohere.com/v2/docs/responsible-use | Model card example for safety limitations and responsible-use framing. |
| Models overview | https://docs.cohere.com/v2/docs/models | Current model status and context lengths. |

### Tier 3

| Source | URL | Why it matters |
|---|---|---|
| Compass | https://cohere.com/compass | Managed enterprise search, parsing, indexing, RBAC, DOCX support. |
| North | https://cohere.com/north | Cohere's enterprise agent platform positioning. |
| Cohere Toolkit | https://github.com/cohere-ai/cohere-toolkit | Reference RAG app architecture and deployment patterns. |
| RAG with Chat, Embed, Rerank via Pinecone | https://docs.cohere.com/page/rag-with-chat-embed | Reference integration pattern for vector DB plus Cohere RAG stack. |
| Embed Jobs with Pinecone | https://docs.cohere.com/page/embed-jobs-serverless-pinecone | Reference pattern for large-scale async embedding plus hosted vector search. |
| Cohere status page | https://status.cohere.com/ | Demo-day operational check and incident-subscription source. |

### Company Framing Sources

| Source | URL | Why it matters |
|---|---|---|
| Master citations to build trustworthy AI | https://cohere.com/blog/master-citations-to-build-trustworthy-ai | Citation recall, citation precision, and trust UX. |
| From GraphRAG to agentic search | https://cohere.com/blog/ai-retrieval-graphrag-and-agentic-search | When simple RAG needs stronger retrieval structure. |
| Building AI agents: hurdles and fixes | https://cohere.com/blog/building-ai-agents | Practical delivery risks for enterprise agents. |
| Master agentic AI deployment | https://cohere.com/blog/agentic-ai-deployment | Rollout and deployment framing for future production discussion. |
| Private deployments of AI | https://cohere.com/blog/why-more-businesses-choose-private-deployments-of-ai | Security, control, and deployment tradeoffs. |

## Triage Legend

- **MUST-ADD**: Add before the interview because it strengthens a required demo moment.
- **MENTION-ONLY**: Do not build now. Prepare a precise Q&A answer.
- **IGNORE**: Out of scope for this prototype or likely to add more risk than value.

Repo state:

- **NONE**: Not present.
- **PARTIAL**: Some code or docs exist, but the demo does not fully prove it.
- **FULL**: Present and defensible in the current prototype.

Effort:

- **S**: Under 2 hours.
- **M**: Half day.
- **L**: More than a day.

## Capability Matrix

Sorted by triage first, then by demo value.

| # | Capability | Repo state | Effort | Demo impact | Triage | Decision | Source |
|---|---|---|---|---|---|---|---|
| 1 | Trace-rich citation display | FULL | S | SHOW-LIVE | MENTION-ONLY | `run_agent_session.py --show-audit` and the registry runner expose cited answer spans, source IDs, doc IDs, page, language, access level, source title, vector score, rerank score, and citation validation. | https://docs.cohere.com/docs/rag-citations |
| 2 | Interrogate-style follow-up demo | FULL | S | SHOW-LIVE | MENTION-ONLY | The demo script and registry include a source/access follow-up that answers from prior audit metadata instead of doing an unrelated search. | https://docs.cohere.com/v2/docs/agentic-rag |
| 3 | Live bilingual retrieval transcript | FULL | S | SHOW-LIVE | MENTION-ONLY | Saved live readiness transcripts include the natural multilingual NATO case with Embed v4 and Rerank v4. Re-run before the interview if model behavior or the corpus changes. | https://docs.cohere.com/changelog/embed-multimodal-v4 |
| 3a | Model-grounded insufficiency refusal | FULL | S | SHOW-LIVE | MENTION-ONLY | Retrieval returns related authorized pages for unsupported dated or scheduled claims, sends those pages to Command A through `documents=`, and records the model-grounded abstention in `answer_audit`. | https://docs.cohere.com/page/rag-evaluation-deep-dive |
| 4 | Lightweight eval runner | FULL | S | SHOW-LIVE | MENTION-ONLY | `run_demo_query_registry.py` runs the demo cases through the ADK harness, checks expected docs, expected refusals, source metadata, facet coverage, search counts, and citation source resolution. | https://docs.cohere.com/page/rag-evaluation-deep-dive |
| 5 | Native Cohere citations | FULL | S | SHOW-LIVE | MENTION-ONLY | Current `grounding.py` uses direct non-streaming Cohere Chat with `documents=` and reads native post-response citation spans. Live testing showed Command A document RAG rejected explicit `citation_options`, so the demo relies on Cohere's default accurate document-RAG citation behavior. | https://docs.cohere.com/docs/rag-citations |
| 6 | Custom citation document IDs | FULL | S | SHOW-LIVE | MENTION-ONLY | Current final answer documents include stable IDs, which lets citations point back to repo source metadata instead of anonymous auto IDs. Keep this as a technical walkthrough point. | https://docs.cohere.com/docs/rag-citations |
| 7 | Cohere canonical RAG path | FULL | S | SHOW-LIVE | MENTION-ONLY | The repo follows Cohere's core pattern: retrieve evidence, rerank evidence, pass authorized evidence through `documents=`, and read native citations from Chat. | https://docs.cohere.com/docs/rag-with-cohere |
| 8 | Embed v4 multimodal page embeddings | FULL | S | SHOW-LIVE | MENTION-ONLY | Current pipeline embeds a lightweight doc/page/title label plus the rendered page image. Extracted page text is retained for Chroma results, Rerank, grounding, citations, and audit. This is the right core technical story for PDF doctrine, tables, and layout-heavy material. | https://docs.cohere.com/changelog/embed-multimodal-v4 |
| 9 | Embed v4 `output_dimension` trade-off | PARTIAL | S | MENTION-VERBAL | MENTION-ONLY | The repo already configures `COHERE_EMBED_OUTPUT_DIMENSION`, defaulting to 1536. Explain 1536 as max quality for demo and smaller dimensions as cost/storage options. Do not tune before the interview unless latency becomes a problem. | https://docs.cohere.com/v2/reference/embed |
| 10 | Embed request batching | PARTIAL | M | MENTION-VERBAL | MENTION-ONLY | The current indexer sends one rendered page per call. The Embed API supports multiple `inputs`, but image-per-call behavior needs re-verification before changing. Keep the simple path for correctness. | https://docs.cohere.com/v2/reference/embed |
| 11 | Rerank v4 Pro | FULL | S | SHOW-LIVE | MENTION-ONLY | Current gateway uses Cohere Rerank and current config points at v4. Pro is the right default for quality. Fast is the production latency option. | https://docs.cohere.com/changelog/rerank-v4.0 |
| 12 | YAML rerank records | FULL | S | SHOW-LIVE | MENTION-ONLY | Current `_rerank_document` serializes title, doc ID, page, language, status, version, access level, and content as YAML. This matches Cohere's semi-structured rerank pattern. | https://docs.cohere.com/docs/reranking-with-cohere |
| 13 | Command A for final grounded answers | FULL | S | SHOW-LIVE | MENTION-ONLY | Command A is appropriate because Cohere positions it for RAG, tool use, agents, and multilingual work. Keep it as the default final answer model. | https://docs.cohere.com/v1/docs/command-a |
| 14 | Chat API compatibility details | FULL | S | MENTION-VERBAL | MENTION-ONLY | The final-answer path depends on Chat `documents`, citation objects, and safety-mode compatibility. Use the Chat API reference when answering implementation details. | https://docs.cohere.com/reference/chat |
| 15 | Agentic RAG for mixed PDFs | PARTIAL | S | MENTION-VERBAL | MENTION-ONLY | The repo uses page-level multimodal PDF retrieval. It does not yet implement table-aware parsing, multi-vector retrieval, or query augmentation from the mixed-PDF cookbook. Mention as a future accuracy path, not current behavior. | https://docs.cohere.com/page/agentic-rag-mixed-data |
| 16 | Enterprise document parsing and DOCX strategy | PARTIAL | S | MENTION-VERBAL | MENTION-ONLY | The corpus includes one official DOCX-origin source normalized through the publisher's PDF pair and tracked in manifest metadata. Do not claim native `.docx` parsing. Production options are Compass or a manifest-compatible parser and normalization adapter for mixed enterprise documents. | https://docs.cohere.com/page/document-parsing-for-enterprises |
| 17 | Command A Reasoning thinking blocks | PARTIAL | S | MENTION-VERBAL | MENTION-ONLY | Do not swap the live demo backend by default. The backend can now preserve native Cohere `thinking` blocks when a reasoning-capable Chat model returns them, and the UI can display those blocks when present. Use this as a notebook/demo comparison, while keeping Command A as the stable tool-loop model. | https://docs.cohere.com/v2/docs/reasoning |
| 18 | Aya Expanse | NONE | M | MENTION-VERBAL | MENTION-ONLY | Do not swap now. The bilingual retrieval proof mainly depends on Embed v4 and Rerank v4. Aya is a credible follow-up benchmark if the panel asks about multilingual generation. | https://docs.cohere.com/v2/docs/aya-expanse |
| 19 | Structured outputs JSON mode | NONE | S | MENTION-VERBAL | MENTION-ONLY | Useful for downstream doctrine extraction, but Cohere docs say JSON mode is not supported in RAG mode. For trace output, serialize deterministic metadata from the retrieval and citation objects instead of asking the RAG answer call for JSON. | https://docs.cohere.com/v2/docs/structured-outputs |
| 20 | `strict_tools` | NONE | M | MENTION-VERBAL | MENTION-ONLY | Cohere-native tool calls can enforce tool schemas with `strict_tools`, but this repo uses Google ADK for orchestration. Justify ADK as the runtime layer and Cohere as the model/retrieval/citation layer. | https://docs.cohere.com/v2/docs/structured-outputs |
| 21 | Cohere-native tool use | NONE | L | MENTION-VERBAL | MENTION-ONLY | Do not migrate from ADK before the interview. Explain that Cohere's protocol is a valid alternative, while ADK gives session/runtime structure and Cohere handles model calls, Rerank, Embed, and citations. | https://docs.cohere.com/v2/docs/tool-use-overview |
| 22 | Safety modes | PARTIAL | S | MENTION-VERBAL | MENTION-ONLY | Command A RAG/tool-use paths use `CONTEXTUAL`. Do not overstate this as security. Security in this demo is retrieval authorization before documents reach the model. | https://docs.cohere.com/v2/docs/safety-modes |
| 23 | Embed Jobs API | NONE | M | MENTION-VERBAL | MENTION-ONLY | Good scalability answer for 100K+ text documents and periodic batch updates. Do not build now because the prototype indexes rendered PDF pages and the guide has version nuance to re-check before implementation. | https://docs.cohere.com/v1/docs/embed-jobs-api |
| 24 | Datasets API for batch embedding | NONE | M | MENTION-VERBAL | MENTION-ONLY | Datasets are the staging layer for Embed Jobs and preserve metadata fields. Mention for large periodic corpus rebuilds, not the live prototype. | https://docs.cohere.com/v2/docs/datasets |
| 25 | Batches API | NONE | M | MENTION-VERBAL | MENTION-ONLY | Useful for offline batch jobs, but it is not needed for the live RAG loop. Mention only if asked about back-office processing at scale. | https://docs.cohere.com/reference/create-batch |
| 26 | Rate limits and real-key readiness | NONE | S | MENTION-VERBAL | MENTION-ONLY | The prototype demo needs a rate-limit answer because it uses real Cohere calls. Rehearse with the real key or a prebuilt index, and know image Embed and Rerank limits. | https://docs.cohere.com/v2/docs/rate-limits |
| 27 | Private deployment | NONE | S | MENTION-VERBAL | MENTION-ONLY | Prepare a crisp answer: prototype uses API for speed; production can move to private cloud VPC or on-prem when data boundary requires it. | https://docs.cohere.com/docs/private-deployment-overview |
| 28 | Private deployment SDK portability | NONE | S | MENTION-VERBAL | MENTION-ONLY | The private-deployment usage docs show the same SDK can target a private base URL. This supports a practical migration answer without re-architecting the demo. | https://docs.cohere.com/v2/docs/private-deployment-usage |
| 29 | Model Vault | NONE | S | MENTION-VERBAL | MENTION-ONLY | Position as Cohere-managed single-tenant inference for teams that want isolation without operating the full serving stack. | https://docs.cohere.com/docs/model-vault |
| 30 | Trust, security, and privacy posture | NONE | S | MENTION-VERBAL | MENTION-ONLY | Use Trust Center, security, privacy, and deployment docs for claims about compliance posture and data boundary options. Do not invent security guarantees beyond the sources. | https://trustcenter.cohere.com/ |
| 31 | Deprecation hygiene | NONE | S | MENTION-VERBAL | MENTION-ONLY | Re-check model and endpoint lifecycle before the interview. This prevents stale model names or deprecated endpoint claims in the walkthrough. | https://docs.cohere.com/docs/deprecations |
| 32 | SDK cloud platform compatibility | NONE | S | MENTION-VERBAL | MENTION-ONLY | Good answer for customer portability across Cohere platform, Bedrock, SageMaker, Azure, OCI, and private deployment. | https://docs.cohere.com/v2/docs/cohere-works-everywhere |
| 33 | Compass | NONE | S | MENTION-VERBAL | MENTION-ONLY | Compass is the natural production search product: managed parsing, indexing, DOCX/PDF/PPT/XLSX support, role-based access controls, and document-level security. The prototype exposes the mechanics for the interview. | https://cohere.com/compass |
| 34 | North | NONE | S | MENTION-VERBAL | MENTION-ONLY | North is the enterprise agent platform. Position this prototype as a tailored technical proof that maps to North-style secure agents, not a replacement for North. | https://cohere.com/north |
| 35 | Cohere Toolkit | NONE | S | MENTION-VERBAL | MENTION-ONLY | Toolkit is useful as a reference RAG app, but adopting it would bloat the demo and reduce the value of explaining the custom design. | https://github.com/cohere-ai/cohere-toolkit |
| 36 | OpenAI SDK compatibility | NONE | S | MENTION-VERBAL | MENTION-ONLY | Useful if the panel asks about migrating an OpenAI-shaped app to Cohere. Not relevant to this repo because we use the Cohere SDK directly for final answers and embeddings. | https://docs.cohere.com/docs/compatibility-api |
| 37 | Full Toolkit migration | NONE | L | NONE | IGNORE | Too much churn. It would turn the interview into a framework integration instead of a clear customer-specific prototype. | https://github.com/cohere-ai/cohere-toolkit |
| 38 | Aya Vision generation | NONE | L | NONE | IGNORE | The demo does not need a vision-language generator. Embed v4 already handles the multimodal retrieval requirement for rendered PDF pages. | https://cohere.com/research/aya |
| 39 | Fine-tuning | NONE | L | NONE | IGNORE | The problem is retrieval, citations, access control, and deployment. Fine-tuning would distract and add evaluation burden. | https://docs.cohere.com/v2/docs/models |

## Live Readiness Reminder

The saved live readiness transcripts cover the primary bilingual and ACL
queries. Before the interview, re-run the transcript set with `--show-audit`
after any retrieval, prompt, corpus, model, or answerability change.

## Q&A Positioning

Use these short answers when the panel probes missing capabilities.

- **Why ADK instead of Cohere-native tool use?** ADK owns the agent runtime, sessions, and lifecycle. Cohere owns the model calls, Embed v4 retrieval vectors, Rerank v4 ordering, and native citations. That keeps the prototype small while still using Cohere where it matters most.
- **Why not North?** North is a production platform for secure enterprise agents. This prototype exposes the mechanics because the interview asks for a live demo and technical walkthrough.
- **Why not Compass?** Compass is the likely managed-search production path, especially for native DOCX parsing and large managed indexes. This prototype implements the underlying pattern directly so the panel can inspect access control, embeddings, reranking, and citations.
- **What about DOCX?** The brief names PDFs and DOCX. Native `.docx` parsing is not implemented. The prototype proves the ingestion-normalization boundary with one official DOCX-origin source represented through the publisher's PDF pair; after verified normalization, PDF-origin and DOCX-origin evidence use the same Embed v4, Rerank v4, ACL, citation, and audit workflow. Production can use Compass parsing or add a parser and normalization adapter with the same metadata and ACL contract.
- **What about private data?** The local prototype uses Cohere APIs for speed. A defence deployment can use VPC, on-premises, or Model Vault depending on data residency, isolation, and operational constraints.
- **Why not structured JSON answers?** Cohere JSON mode is not supported in the same RAG mode used for cited answers. The safer trace design is deterministic: generate the answer with citations, then serialize trace metadata from retrieved sources and citation objects.
- **How do you prove accuracy?** Use retrieval checks against expected doc IDs, refusal checks for denied evidence, and claim-level grounding checks where needed. Cohere's RAG evaluation material frames this as retrieval quality plus generated-answer faithfulness.
- **What happens at production scale?** Real-time queries use the Embed/Rerank/Chat path. Large periodic corpus jobs can use Datasets and Embed Jobs for text-heavy corpora, while production document search may move to Compass.
- **How do you handle live-demo risk?** Build the index before presenting, check the Cohere status page and rate limits, keep a saved transcript, and be clear that the active prototype uses live Cohere calls.
- **How do you avoid stale Cohere claims?** Re-check the models page, deprecations page, and source inventory before interview week. Do not rely on old blog-memory for model names or endpoint behavior.

## Recheck List

Before the interview:

- Re-check model names in `defence_agent/config.py` against Cohere's models page.
- Confirm Rerank model default is `rerank-v4.0-pro`, not a deprecated alias.
- Confirm direct Cohere Chat still returns native citation spans without explicit `citation_options`.
- Confirm the bilingual transcript was run with live Cohere Embed v4 and Rerank v4.
- Confirm the DOCX-origin normalization answer is present in the Q&A notes.
- Confirm the DOCX-origin source remains searchable and exposes `source_format`, `normalized_format`, and `normalization_method` in audit metadata.
- Check Cohere status before a live run: https://status.cohere.com/
- Check rate limits and whether the demo key is trial or production.
- Check Cohere deprecations and model names before recording or presenting.
- Check Trust Center, security, privacy, and deployment docs before making security or data-boundary claims.
