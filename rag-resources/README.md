# RAG Resources

This folder contains third-party RAG learning material.

The main resource is `systematically-improving-rag`, added as a Git submodule.

## Why This Matters

- This book focuses on evaluation-first RAG design.
- It is useful before building the Cohere prototype.
- It helps avoid common mistakes in retrieval, chunking, feedback loops, routing, and production evaluation.
- Treat it as a reference source, not as project-owned content.

## Read The Book Locally

From the project root:

```bash
cd rag-resources/systematically-improving-rag
uv sync
uv run mkdocs serve -a 127.0.0.1:8001
```

Open:

```text
http://127.0.0.1:8001/systematically-improving-rag/
```

## Build The Static Site

```bash
cd rag-resources/systematically-improving-rag
uv run mkdocs build
```

The generated site stays inside the submodule and is ignored by that repo.

## Build The PDF Ebook

```bash
cd rag-resources/systematically-improving-rag
bash build_book.sh
```

Expected output:

```text
rag-resources/systematically-improving-rag/ebook/systematically_improving_rag_book.pdf
```

The generated ebook stays inside the submodule and is ignored by that repo.

If the upstream script fails on generated Markdown, run this fallback after `ebook/book_all.md` exists:

```bash
python3 - <<'PY'
from pathlib import Path

src = Path("ebook/book_all.md")
out = Path("/tmp/systematically_improving_rag_book_fixed.md")
lines = src.read_text(encoding="utf-8").splitlines()
fixed = []

for idx, line in enumerate(lines):
    if idx > 3 and line.strip() == "---":
        line = "***"
    line = line.replace("Expected\\Predicted", "Expected/Predicted")
    line = line.replace("Expected\\Selected", "Expected/Selected")
    fixed.append(line)

out.write_text("\n".join(fixed) + "\n", encoding="utf-8")
print(out)
PY

pandoc /tmp/systematically_improving_rag_book_fixed.md \
  -s \
  -o ebook/systematically_improving_rag_book.pdf \
  --toc \
  --toc-depth=2 \
  -V geometry:margin=1in \
  -V documentclass=book \
  -V classoption=oneside \
  -V urlcolor=blue \
  -V linkcolor=blue \
  -V colorlinks=true \
  --resource-path="$PWD:$PWD/docs:$PWD/docs/assets/images:$PWD/docs/talks:$PWD/docs/workshops" \
  --pdf-engine=tectonic
```

## Update The Submodule

```bash
git submodule update --remote rag-resources/systematically-improving-rag
git add rag-resources/systematically-improving-rag
git commit -m "Update RAG book submodule"
```

## Ownership Note

This is third-party material from:

```text
https://github.com/jxnl/systematically-improving-rag
```

Do not rewrite it as if it belongs to this repo.
