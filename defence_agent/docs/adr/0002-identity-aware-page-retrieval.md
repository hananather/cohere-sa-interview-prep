# 0002 Identity-Aware Page Retrieval

## Decision

Apply persona access filters before generation and before evidence reaches the
final answer model.

## Reason

The model is not the security boundary. Unauthorized source text must never be
sent to Cohere for users who cannot access it.

## Consequence

The same query can produce different authorized evidence for
`clearance_unclassified`, `clearance_secret`, and `clearance_top_secret`.
Excluded source metadata can appear in audit output, but excluded source text is
not returned.
