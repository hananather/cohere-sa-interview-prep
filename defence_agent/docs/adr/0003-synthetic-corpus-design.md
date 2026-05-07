# 0003 Synthetic Corpus Design

## Context

The interview needs safe but realistic public-sector documents that demonstrate status, language, version, access, and table metadata.

## Decision

Use Markdown with YAML frontmatter as the canonical corpus, generate PDF/DOCX artifacts where practical, and include a structured `doctrine_review_tracker.csv` table.

## Alternatives Considered

- Use real defense documents.
- Use only plain text snippets.

## Consequences

The corpus is safe, repeatable, and metadata-rich enough for evals and demo traces.

## Status

Accepted.
