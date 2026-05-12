# 0003 Mixed Normalized Page Corpus

## Decision

Use a manifest-driven corpus with official public English/French defence PDFs,
one DOCX-origin document represented through an official PDF pair, and a small
number of fictional restricted PDFs.

## Reason

Real public sources make the retrieval demo credible. The DOCX-origin entry
proves the normalization boundary without adding native `.docx` parsing in this
pass. Fictional restricted PDFs make access-control behavior safe to
demonstrate. A manifest keeps document IDs, language, source provenance, and
access tiers stable.

## Consequence

The canonical corpus lives under `defence_agent/data/corpus/`. The retrieval
layer is format-agnostic after verified normalization to page evidence. Native
DOCX parsing is not implemented; production native DOCX ingestion should use
Compass or a manifest-compatible parser and normalization adapter.
