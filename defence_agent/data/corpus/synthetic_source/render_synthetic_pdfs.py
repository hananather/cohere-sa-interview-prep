#!/usr/bin/env python3
"""Render synthetic markdown sources to simple multi-page PDFs."""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import wrap

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


SOURCE_DIR = Path(__file__).resolve().parent
OUT_DIR = SOURCE_DIR.parent / "synthetic"


def strip_front_matter(text: str) -> str:
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)


def markdown_to_flowables(markdown: str):
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontName = "Helvetica"
    body.fontSize = 10
    body.leading = 14
    heading1 = styles["Heading1"]
    heading2 = styles["Heading2"]

    flowables = []
    for block in re.split(r"\n\s*\n", markdown.strip()):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            flowables.append(Paragraph(block[2:].strip(), heading1))
            flowables.append(Spacer(1, 8))
            continue
        if block.startswith("## "):
            flowables.append(Paragraph(block[3:].strip(), heading2))
            flowables.append(Spacer(1, 4))
            continue
        html = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", html)
        html = "<br/>".join(" ".join(wrap(line, width=110)) for line in html.splitlines())
        flowables.append(Paragraph(html, body))
        flowables.append(Spacer(1, 7))
    return flowables


def render_one(source_path: Path) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = strip_front_matter(source_path.read_text(encoding="utf-8"))
    out_path = OUT_DIR / f"{source_path.stem}.pdf"
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title=source_path.stem,
    )
    doc.build(markdown_to_flowables(markdown))
    return out_path


def main() -> None:
    for source_path in sorted(SOURCE_DIR.glob("*.md")):
        out_path = render_one(source_path)
        print(out_path)


if __name__ == "__main__":
    main()
