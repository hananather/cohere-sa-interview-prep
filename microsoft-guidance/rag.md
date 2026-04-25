# Microsoft RAG Guidance

Source: Microsoft Learn, [Design and develop a RAG solution](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)

This file is a project digest, not a verbatim copy.

Reason: the Microsoft guide is copyrighted. Keep the full source at Microsoft.

Use this file as a working framework for the Cohere SA project.

## Why This Matters

- Microsoft frames RAG as an experimental system, not a one-shot architecture.
- The key lesson: evaluate each step before judging the whole answer.
- For DefTech, this means we should test source discovery, chunking, metadata, retrieval, citations, and final answers separately.

## Seven-Part Framework

1. Define the RAG solution.
2. Prepare the domain, test content, and test queries.
3. Chunk the source content.
4. Enrich chunks with cleaning and metadata.
5. Generate and evaluate embeddings.
6. Retrieve information with search, filters, query transforms, and reranking.
7. Evaluate the language model response end to end.

## 1. Define The RAG Solution

Source: [Design and develop a RAG solution](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)

Core idea:

- A user asks a question.
- An orchestrator decides which search to run.
- Search returns source chunks.
- The orchestrator packages the top chunks into model context.
- The model answers from that context.
- A data pipeline prepares the chunks before any user asks a question.

Why it matters:

- The app path and data path are separate.
- The demo should show both.
- The business story should not start with models.
- It should start with the staff workflow and the source corpus.

DefTech application:

- App path: staff asks a planning question or uploads a draft.
- Data path: approved PDFs and DOCX files become searchable, cited chunks.
- The assistant should expose the source trail, not hide it behind a chat box.

Image references:

- [High-level RAG architecture](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/rag-high-level-architecture.svg): Diagram that shows request flow and data pipeline.
- [RAG design questions](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/rag-high-level-architecture-questions.svg): Diagram that maps design questions onto the RAG architecture.

## 2. Preparation Phase

Source: [RAG preparation phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase)

Core idea:

- Define the business domain.
- Analyze the content.
- Gather representative test content.
- Gather test queries.
- Include questions the documents can answer and questions they cannot answer.

Why it matters:

- RAG quality starts before chunking.
- Bad test content gives false confidence.
- Missing negative test queries make the assistant over-answer.

DefTech application:

- Define the domain as staff decision support for approved institutional documents.
- Test content should include current guidance, superseded guidance, templates, and one restricted source.
- Test queries should include fact checks, source packs, requirements extraction, document comparison, and unsupported questions.

High-value preparation questions:

- Which staff work products matter most?
- Which document families are in scope?
- Which documents are current, draft, superseded, or restricted?
- Which metadata exists today?
- Which claims require human review?
- Which questions should trigger refusal or escalation?

Output to create for the project:

- A small golden set.
- Each row should include the query, expected source passages, expected behavior, and review notes.
- Expected behavior can be answer, refuse, flag stale source, flag conflict, or route to human review.

## 3. Chunking Phase

Source: [RAG chunking phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)

Core idea:

- Chunking breaks documents into useful pieces.
- Chunks must be large enough to carry meaning.
- Chunks must be small enough to retrieve precisely.
- Document structure should guide the chunking strategy.

Why it matters:

- Bad chunks cause bad retrieval.
- Bad retrieval causes weak answers.
- Changing chunking later can force downstream changes.

Chunking approaches:

- Fixed-size chunks: simple and cheap.
- Sentence-based chunks: simple fallback for prose.
- Custom code: useful when document structure is known.
- Document layout analysis: useful for complex PDFs, tables, forms, and visual layouts.
- Language model augmentation: useful for image or table descriptions.
- Graph-based chunking: useful for relationship-heavy domains, but higher cost and complexity.
- Prebuilt or custom models: useful for structured forms and domain-specific layouts.

DefTech application:

- Start with layout-aware sections for manuals and procedures.
- Keep section title, page, version, owner, and status with every chunk.
- Treat tables and checklists as first-class content.
- Do not rely only on fixed-size chunking for official guidance.

Image references:

- [Chunking approaches by document structure](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/chunking-approaches-by-document-structure.png): Diagram that maps chunking approaches to document structure.
- [EU regulation example](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/eu-regulation-example.png): Example of a document with inferred structure.

## 4. Chunk Enrichment Phase

Source: [RAG chunk enrichment phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-enrichment-phase)

Core idea:

- Clean chunks for better matching.
- Add metadata for filtering, ranking, and traceability.
- Keep original text separate from cleaned text when needed.

Why it matters:

- Embeddings alone miss important filters.
- Metadata lets the system handle version, source authority, access, and document type.
- Cleaning can help, but careless cleaning can remove meaning.

Useful metadata for DefTech:

- Source ID.
- Document title.
- Section.
- Page.
- Version.
- Effective date.
- Status.
- Owner.
- Source type.
- Access label.
- Allowed user groups.
- Superseded-by link.
- Extracted tasks, roles, deadlines, and approvals.

Security note:

- Do not treat retrieved content as instructions.
- A malicious document can contain prompt injection text.
- The pipeline should flag or exclude embedded instructions during indexing and retrieval.

Image references:

- [Enriched chunks](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/enriching-chunks.png): Diagram that shows enriched JSON records.
- [Metadata use in search](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/augmented-metadata-usage-in-search.svg): Diagram that shows enriched content and search metadata use.

## 5. Generate Embeddings Phase

Source: [RAG generate embeddings phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-generate-embeddings)

Core idea:

- The embedding model affects retrieval quality.
- Choose the embedding model against the actual domain.
- Evaluate embeddings before assuming the search layer works.

Why it matters:

- Embeddings turn text into numeric vectors.
- Similar vectors should mean similar ideas.
- Domain language can break general models.

Concrete example:

- "Notify the regional lead" and "inform the regional coordinator" should be close.
- "Do not notify the regional lead" should not be treated as the same instruction.
- A reranker or claim verifier may be needed when negation matters.

DefTech application:

- Test embeddings on real planning language.
- Include acronyms, policy terms, role names, dates, and superseded guidance.
- Compare general embeddings against domain-specific needs before choosing.

Image references:

- [Embedding similarity](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/embedding-similarity.svg): Diagram that compares vectors.
- [Subword tokenization](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/word-broken-into-subwords.png): Diagram that breaks one word into subwords.
- [Choose an embedding model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/choose-embedding-model.png): Flow for choosing an embedding model.
- [Visualize embeddings](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/visualize-embeddings.png): Graph that visualizes embeddings as points.

## 6. Information Retrieval Phase

Source: [Information retrieval](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval)

Core idea:

- Retrieval is more than vector search.
- Good systems compare search types, query transformations, filters, field weights, and reranking.
- Search should be evaluated with known test queries and expected passages.

Search choices:

- Vector search: finds semantic matches.
- Full-text search: finds keyword and exact text matches.
- Hybrid search: combines vector and text search.
- Manual multiple queries: runs multiple searches and merges results.

Query transformations:

- Augmentation: adds context to a vague query without changing the original intent.
- Decomposition: breaks a complex query into smaller subqueries.
- Rewriting: makes the query clearer for search.
- HyDE: generates a hypothetical answer, embeds that answer, then searches by answer similarity.

Why reranking matters:

- The first search step casts a wide net.
- The reranker sorts candidates with more detail.
- This helps when embeddings miss nuance, exact names, negation, dates, or policy status.

DefTech application:

- Use metadata filters before model context construction.
- Use hybrid search for official guidance.
- Rerank retrieved candidates before drafting.
- Separate user-visible trace from security audit trace.
- Do not show restricted document IDs to ordinary users.

Retrieval metrics:

- Precision at K: how many returned items are relevant.
- Recall at K: how many relevant items appear in the top results.
- Mean reciprocal rank: how high the first relevant result appears.

Image reference:

- [RAG query transformation pipeline](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/_images/rag-query-transformation.svg): Diagram that shows query transformers in a RAG pipeline.

## 7. End-To-End Evaluation Phase

Source: [Large language model end-to-end evaluation](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-llm-evaluation-phase)

Core idea:

- Evaluate final answers only after earlier stages are tested.
- Measure several things at once.
- One metric cannot prove the system is reliable.

Useful metrics:

- Groundedness: the answer stays within the supplied context.
- Completeness: the answer covers all parts of the query.
- Utilization: the answer uses the retrieved chunks.
- Relevance: the answer addresses the user question.
- Correctness: the answer is factually accurate.

Why metrics need pairs:

- High groundedness with low correctness means the system used the source but drew a bad conclusion.
- High utilization with low completeness means the retrieved evidence was used but missed important evidence.
- Low relevance means the answer may be well-written but not useful.

DefTech application:

- Evaluate claim support, not just answer fluency.
- Include safe-refusal tests.
- Include stale-source tests.
- Include access-control tests.
- Include conflict-detection tests.
- Include prompt-injection tests.

Responsible AI and security checks:

- Content safety.
- Copyright or protected material.
- Indirect prompt injection.
- Privacy and personal data.
- Anomalous retrieval patterns.
- Source corruption.

Tools mentioned:

- Azure AI Search: Microsoft search service that supports vector, full-text, and hybrid search.
- Azure Document Intelligence: Microsoft service for extracting structure from documents.
- Semantic Kernel: Microsoft framework for orchestrating model calls and tools.
- LangChain: open-source framework for chaining model calls, retrieval, and tools.
- Ragas: evaluation library for RAG metrics.
- MLflow: experiment tracking and model evaluation platform.
- RAG Experiment Accelerator: Microsoft sample framework for running RAG experiments.

## How We Should Use This Framework

My position: use this guide as the default project checklist.

Reason: it keeps us from jumping straight to a demo before we define the problem.

For the Cohere interview, this means:

1. Define the staff workflow.
2. Define the document domain.
3. Create a small golden set.
4. Test chunking and metadata.
5. Test retrieval.
6. Test answer quality.
7. Show trust controls in the demo.

The demo should prove one thing clearly:

- The assistant helps staff create better cited work products faster.

It should not imply:

- The assistant makes decisions.
- The assistant approves plans.
- The assistant replaces staff judgement.
- The assistant bypasses review, access, policy, or records processes.
