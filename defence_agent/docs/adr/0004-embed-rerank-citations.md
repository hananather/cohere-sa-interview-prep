# 0004 Embed V4, Rerank V4, And Cohere Citations

## Decision

Embed rendered PDF pages with Cohere Embed v4, rerank authorized candidate pages
with Cohere Rerank v4, and generate final answers with Cohere Chat document
citations.

## Reason

The assignment prioritizes accuracy and traceability. Page-level multimodal
embeddings help find relevant evidence in PDF manuals. Rerank improves evidence
ordering. Native Cohere document citations link answer spans back to source
pages.

## Consequence

`answer_audit` records query, persona, filters, source IDs, page numbers, vector
scores, rerank scores, Cohere document IDs, citation spans, and citation source
resolution. Citation coverage is automated; citation precision still needs live
transcript review.
