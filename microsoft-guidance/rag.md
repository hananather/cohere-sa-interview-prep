# Microsoft RAG Source Index

This is a source map for the Microsoft Learn RAG series.

It is not a summary.

Use this file to find the right source page, phase file, and user-provided verbatim text.

## Source PDFs

- Main source: `microsoft-guidance/azure-architecture-ai-ml.pdf`
- Intro export: `microsoft-guidance/Design and Develop a RAG Solution - Azure Architecture Center _ Microsoft Learn.pdf`

## Handling Rules

- Keep Microsoft prose out of Markdown unless the user provides the exact text.
- Add user-provided exact text only under `## Verbatim Text Provided By User`.
- Do not summarize inside verbatim sections.
- Preserve pasted source wording, order, lists, and headings as closely as Markdown allows.
- Reference Microsoft images by PDF page and role.
- Do not extract Microsoft image assets into this repo.
- If the user provides screenshots or image exports, place them in `microsoft-guidance/rag/assets/`.

## Seven-Part RAG Series

| Part | File | PDF pages | Official link |
|---|---|---:|---|
| 1 | [Design and develop](rag/01-design-and-develop.md) | 260 to 264 | [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide) |
| 2 | [Preparation](rag/02-preparation.md) | 265 to 274 | [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase) |
| 3 | [Chunking](rag/03-chunking.md) | 275 to 286 | [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase) |
| 4 | [Chunk enrichment](rag/04-chunk-enrichment.md) | 287 to 291 | [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-enrichment-phase) |
| 5 | [Generate embeddings](rag/05-generate-embeddings.md) | 292 to 299 | [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-generate-embeddings) |
| 6 | [Information retrieval](rag/06-information-retrieval.md) | 300 to 315 | [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval) |
| 7 | [End-to-end evaluation](rag/07-end-to-end-evaluation.md) | 316 to 327 | [Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-llm-evaluation-phase) |

## Coverage Check

- The RAG series in the main PDF runs from page 260 through page 327.
- The seven files cover every page in that range.
- No page in the RAG range is intentionally skipped.
- Image references are page-level references only.
- Verbatim sections are empty until the user provides exact text.

## How To Add Verbatim Text

1. Paste the exact Microsoft text into the chat.
2. Identify the phase, page, or heading if you know it.
3. Place the text under `## Verbatim Text Provided By User` in the matching phase file.
4. Do not rewrite the pasted text.
5. Add only short local notes under `## Notes For Future Use`.
