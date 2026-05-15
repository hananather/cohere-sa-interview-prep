#!/usr/bin/env python3
"""Build a self-contained static backup page for the Defence Agent demo."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "presentation_backup" / "index.html"
MANIFEST_PATH = ROOT / "defence_agent" / "data" / "corpus" / "manifest.yaml"
TRANSCRIPT_ROOT = ROOT / "defence_agent" / "data" / "transcripts"


@dataclass(frozen=True)
class DemoRun:
    title: str
    purpose: str
    transcript: Path
    presenter_note: str
    accent: str = "neutral"


DEMO_RUNS = [
    DemoRun(
        title="Planning Brief Comparison",
        purpose="One planning question becomes multiple ordered searches across defence policy and AI strategy sources.",
        transcript=TRANSCRIPT_ROOT
        / "live_readiness_flagship_retry_20260511_013333"
        / "flagship_planning_brief_modernization.json",
        presenter_note="Use this as the main answer demo. It proves multi-document planning, citations, and traceability.",
        accent="primary",
    ),
    DemoRun(
        title="Access Boundary: Unclassified User",
        purpose="The same question refuses when the relevant source is outside the user's access scope.",
        transcript=TRANSCRIPT_ROOT
        / "live_readiness_final_20260511_013937"
        / "acl_unclassified_sensor_fusion_release_rule.json",
        presenter_note="Show this before the secret-user version. The point is authorization before generation.",
        accent="warn",
    ),
    DemoRun(
        title="Access Boundary: Secret User",
        purpose="A cleared user receives a cited answer because the restricted source is authorized for that persona.",
        transcript=TRANSCRIPT_ROOT
        / "live_readiness_final_20260511_013937"
        / "acl_secret_sensor_fusion_release_rule.json",
        presenter_note="Pair this with the unclassified refusal to show that prompts are not the security boundary.",
        accent="secure",
    ),
    DemoRun(
        title="Scanned Manual Retrieval",
        purpose="Digitized physical-manual evidence enters the same retrieval, rerank, citation, and trace workflow.",
        transcript=TRANSCRIPT_ROOT
        / "eval_full_registry_20260514_020915"
        / "scanned_manual_technical_intelligence.json",
        presenter_note="Use this if you want to prove the customer workflow is broader than clean PDFs.",
        accent="primary",
    ),
    DemoRun(
        title="Evidence Gap Refusal",
        purpose="Related authorized pages are retrieved, but the answer refuses when the exact claim is unsupported.",
        transcript=TRANSCRIPT_ROOT
        / "insufficient_evidence_strategy_realignment_20260511_131708"
        / "insufficient_evidence_planning_topic.json",
        presenter_note="This is the accuracy story. Rerank finds related sources; grounded generation still abstains.",
        accent="warn",
    ),
    DemoRun(
        title="Source And Access Follow-Up",
        purpose="A follow-up question can cite the source page and access level behind the prior answer.",
        transcript=TRANSCRIPT_ROOT
        / "live_readiness_final_20260511_013937"
        / "trace_follow_up_source_access.json",
        presenter_note="Use this to show interrogation over a session, not disconnected one-off searches.",
        accent="secure",
    ),
    DemoRun(
        title="French NATO Doctrine Answer",
        purpose="The same evidence workflow preserves citations for a French answer over NATO doctrine sources.",
        transcript=TRANSCRIPT_ROOT
        / "eval_full_registry_20260514_020915"
        / "french_nato_doctrine_answer.json",
        presenter_note="Keep this as optional backup if the panel asks about bilingual retrieval or multilingual users.",
        accent="neutral",
    ),
]


def main() -> None:
    manifest = _load_yaml(MANIFEST_PATH)
    transcripts = [(run, _load_json(run.transcript)) for run in DEMO_RUNS]
    html = build_page(manifest, transcripts)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


def build_page(manifest: dict[str, Any], transcripts: list[tuple[DemoRun, dict[str, Any]]]) -> str:
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    documents = list(manifest.get("documents", []) or [])
    body = "\n".join(
        [
            _hero(built_at, documents, transcripts),
            _database_section(documents),
            _walkthrough_section(transcripts),
            _implementation_map_section(),
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Defence Agent Presentation Backup</title>
  <style>
{CSS}
  </style>
</head>
<body>
  <div class="shell">
    {body}
  </div>
</body>
</html>
"""
    return re.sub(r"[ \t]+(?=\n)", "", html)


def _hero(
    built_at: str,
    documents: list[dict[str, Any]],
    transcripts: list[tuple[DemoRun, dict[str, Any]]],
) -> str:
    total_sources = len(documents)
    source_types = Counter(_source_kind(doc) for doc in documents)
    access_levels = Counter(str(doc.get("access_level", "unknown")) for doc in documents)
    citation_count = sum(int(data.get("citation_count") or 0) for _, data in transcripts)
    return f"""
<header class="hero">
  <div>
    <p class="eyebrow">Static backup page</p>
    <h1>Defence Agent Presentation Backup</h1>
    <p class="lead">A deterministic walkthrough of the same evidence workflow: database, planning answer, trace, citations, access boundary, scanned manual retrieval, refusal, and follow-up.</p>
  </div>
  <div class="hero-panel">
    <div class="stat"><span>{total_sources}</span><label>catalog sources</label></div>
    <div class="stat"><span>{len(transcripts)}</span><label>demo runs</label></div>
    <div class="stat"><span>{citation_count}</span><label>citation spans</label></div>
    <div class="stat"><span>{escape(_counter_label(source_types))}</span><label>source formats</label></div>
    <div class="stat"><span>{escape(_counter_label(access_levels))}</span><label>access levels</label></div>
    <div class="stat"><span>{escape(built_at)}</span><label>page build time</label></div>
  </div>
</header>
<nav class="quick-nav" aria-label="Page sections">
  <a href="#database">Database</a>
  <a href="#planning">Planning</a>
  <a href="#access-unclassified">Access</a>
  <a href="#scanned">Scanned Manual</a>
  <a href="#refusal">Refusal</a>
  <a href="#follow-up">Follow-Up</a>
  <a href="#implementation">Implementation Map</a>
</nav>
"""


def _database_section(documents: list[dict[str, Any]]) -> str:
    rows = "\n".join(_document_row(doc) for doc in documents)
    by_access = Counter(str(doc.get("access_level", "unknown")) for doc in documents)
    by_language = Counter(str(doc.get("language", "unknown")) for doc in documents)
    by_source = Counter(_source_kind(doc) for doc in documents)
    return f"""
<section class="section" id="database">
  <div class="section-head">
    <p class="eyebrow">Step 1</p>
    <h2>Database And Source Catalog</h2>
    <p>This is the document universe the assistant searches. The important point is that source metadata travels with every retrieved page: document ID, access level, language, status, source type, and provenance.</p>
  </div>
  <div class="metric-row">
    {_metric_card("Access", _counter_label(by_access))}
    {_metric_card("Language", _counter_label(by_language))}
    {_metric_card("Formats", _counter_label(by_source))}
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Document ID</th>
          <th>Title</th>
          <th>Access</th>
          <th>Language</th>
          <th>Source Format</th>
          <th>Owner</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</section>
"""


def _walkthrough_section(transcripts: list[tuple[DemoRun, dict[str, Any]]]) -> str:
    cards = []
    for index, (run, data) in enumerate(transcripts, start=1):
        anchor = _anchor_for_title(run.title)
        if index == 1:
            anchor = "planning"
        cards.append(_run_section(index, anchor, run, data))
    return "\n".join(cards)


def _run_section(index: int, anchor: str, run: DemoRun, data: dict[str, Any]) -> str:
    audit = _audit(data)
    retrieval = audit.get("retrieval", {}) if isinstance(audit, dict) else {}
    generation = audit.get("generation", {}) if isinstance(audit, dict) else {}
    answer = str(data.get("answer", "") or "")
    citations = list(audit.get("citations", []) or [])
    sources_sent = _unique_sources(retrieval.get("sources_sent_to_answer", []) or [])
    authorized_sources = _unique_sources(retrieval.get("authorized_sources", []) or [])
    excluded_sources = list(retrieval.get("excluded_sources", []) or [])
    searches = _searches(data, audit, retrieval)
    source_rows = sources_sent or authorized_sources
    citation_validation = data.get("citation_validation") or generation.get("citation_resolution") or {}
    coverage = citation_validation.get("coverage", {}) if isinstance(citation_validation, dict) else {}
    doc_ids = data.get("doc_ids") or _doc_ids_from_sources(source_rows)
    badges = [
        _badge(str(data.get("persona_id") or audit.get("persona_id") or "persona unknown")),
        _badge(f"{data.get('search_count') or retrieval.get('search_count') or len(searches)} search call(s)"),
        _badge(f"{len(citations)} citation span(s)"),
        _badge(f"{len(source_rows)} evidence page(s)"),
    ]
    if data.get("passed") is True:
        badges.append(_badge("verified pass", "good"))
    elif data.get("passed") is False:
        badges.append(_badge("needs review", "warn"))
    answerability = _answerability_summary(retrieval)
    return f"""
<section class="section run run-{escape(run.accent)}" id="{escape(anchor)}">
  <div class="section-head">
    <p class="eyebrow">Step {index + 1}</p>
    <h2>{escape(run.title)}</h2>
    <p>{escape(run.purpose)}</p>
  </div>
  <div class="run-grid">
    <article class="panel main-panel">
      <div class="panel-top">
        <h3>Question</h3>
        <div class="badge-row">{''.join(badges)}</div>
      </div>
      <p class="query">{escape(str(data.get("query", "") or ""))}</p>
      {_follow_up_block(data)}
      <h3>Answer</h3>
      <div class="answer">{_text_block(answer)}</div>
    </article>
    <aside class="panel presenter-panel">
      <h3>Presenter Cue</h3>
      <p>{escape(run.presenter_note)}</p>
      <dl class="compact-dl">
        <div><dt>Mode</dt><dd>{escape(str(data.get("mode", "") or "not recorded"))}</dd></div>
        <div><dt>Model</dt><dd>{escape(str(generation.get("model", "") or "not recorded"))}</dd></div>
        <div><dt>Session</dt><dd>{escape(str(audit.get("session_id", "") or "not recorded"))}</dd></div>
        <div><dt>Cited docs</dt><dd>{escape(', '.join(map(str, doc_ids)) or "none")}</dd></div>
      </dl>
      {_coverage_card(coverage, citation_validation)}
      {answerability}
    </aside>
  </div>
  {_trace_block(searches, retrieval, generation)}
  {_source_block("Evidence Sent To Answer", source_rows)}
  {_excluded_block(excluded_sources)}
  {_citation_block(citations)}
</section>
"""


def _trace_block(searches: list[dict[str, Any]], retrieval: dict[str, Any], generation: dict[str, Any]) -> str:
    if not searches:
        searches = _fallback_searches(retrieval)
    items = "\n".join(_trace_item(i, item) for i, item in enumerate(searches, start=1))
    allowed_access = ", ".join(map(str, retrieval.get("allowed_access") or [])) or "not recorded"
    doc_count = generation.get("document_count")
    citation_mode = generation.get("citation_mode") or "not recorded"
    return f"""
  <details class="details trace-details" open>
    <summary>Trace timeline</summary>
    <div class="trace">
      <div class="trace-meta">
        {_metric_card("Allowed access", allowed_access)}
        {_metric_card("Documents to model", str(doc_count if doc_count is not None else "not recorded"))}
        {_metric_card("Citation mode", str(citation_mode))}
      </div>
      <ol class="trace-list">
        {items}
        <li>
          <strong>Generate cited answer</strong>
          <span>Command A receives the selected authorized pages through the grounded answer path and returns answer text plus citation spans.</span>
        </li>
      </ol>
    </div>
  </details>
"""


def _trace_item(index: int, item: dict[str, Any]) -> str:
    query = item.get("query") or item.get("args", {}).get("query") or "search query not recorded"
    filters = item.get("filters_applied") or item.get("filters") or {}
    policy = item.get("policy_decision") or item.get("policy") or ""
    answerable = item.get("answerable")
    if answerable is None:
        answerable = item.get("answerability_reason") or ""
    metrics = [
        ("authorized", item.get("authorized_source_count")),
        ("sent", item.get("sources_sent_to_answer_count")),
        ("excluded", item.get("excluded_source_count")),
        ("top vector", item.get("top_authorized_vector_score")),
        ("top rerank", item.get("top_authorized_rerank_score")),
    ]
    metric_html = "".join(
        f"<span>{escape(label)}: <strong>{escape(_short_value(value))}</strong></span>"
        for label, value in metrics
        if value not in (None, "", [])
    )
    return f"""
        <li>
          <strong>Search {index}: {escape(str(query))}</strong>
          <span>{escape(_filters_label(filters))}</span>
          <span>{escape(str(policy or answerable or ""))}</span>
          <div class="mini-metrics">{metric_html}</div>
        </li>
"""


def _source_block(title: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    rows = "\n".join(_evidence_row(source) for source in sources)
    return f"""
  <details class="details" open>
    <summary>{escape(title)} ({len(sources)})</summary>
    <div class="table-wrap compact">
      <table>
        <thead>
          <tr>
            <th>Label</th>
            <th>Document</th>
            <th>Page</th>
            <th>Access</th>
            <th>Scores</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </details>
"""


def _excluded_block(excluded_sources: list[dict[str, Any]]) -> str:
    if not excluded_sources:
        return ""
    rows = "\n".join(
        f"""
        <tr>
          <td>{escape(str(source.get("access_level", "restricted") or "restricted"))}</td>
          <td>{escape(str(source.get("reason", "access denied") or "access denied"))}</td>
          <td>{escape(str(source.get("doc_id", "withheld") or "withheld"))}</td>
          <td>{escape(str(source.get("title", "restricted source metadata") or "restricted source metadata"))}</td>
        </tr>
"""
        for source in excluded_sources
    )
    return f"""
  <details class="details" open>
    <summary>Excluded Source Summary ({len(excluded_sources)})</summary>
    <p class="muted">The backup shows denied-source metadata only. It does not expose restricted source text.</p>
    <div class="table-wrap compact">
      <table>
        <thead><tr><th>Access</th><th>Reason</th><th>Document</th><th>Title</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </details>
"""


def _citation_block(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return """
  <details class="details">
    <summary>Citations (0)</summary>
    <p class="muted">This answer is an abstention or metadata lookup without generated citation spans.</p>
  </details>
"""
    items = "\n".join(_citation_item(i, citation) for i, citation in enumerate(citations, start=1))
    return f"""
  <details class="details">
    <summary>All Citation Spans ({len(citations)})</summary>
    <div class="citation-grid">{items}</div>
  </details>
"""


def _citation_item(index: int, citation: dict[str, Any]) -> str:
    sources = citation.get("sources") or []
    labels = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        label = source.get("label") or source.get("citation_id") or source.get("source_id") or source.get("chunk_id")
        doc_id = source.get("doc_id")
        page = source.get("page")
        if label:
            labels.append(f"{label}: {doc_id} p.{page}")
    return f"""
      <article class="citation-card">
        <h4>Citation span {index}</h4>
        <p>{escape(str(citation.get("text", "") or ""))}</p>
        <small>{escape('; '.join(labels) or "source metadata not recorded")}</small>
      </article>
"""


def _implementation_map_section() -> str:
    return """
<section class="section" id="implementation">
  <div class="section-head">
    <p class="eyebrow">Technical walkthrough</p>
    <h2>Implementation Map</h2>
    <p>Use this if the live app is unavailable and you still need to explain how the visible behavior maps to code.</p>
  </div>
  <div class="map-grid">
    <article class="panel"><h3>1. Agent Loop</h3><p><code>defence_agent/agent.py</code> and <code>defence_agent/tools.py</code> let the runtime call <code>search_documents</code> once or multiple times for a planning task.</p></article>
    <article class="panel"><h3>2. Retrieval</h3><p><code>defence_agent/retrieval/chroma_index.py</code> embeds the query, applies metadata filters, searches Chroma, and sends candidates to Cohere Rerank.</p></article>
    <article class="panel"><h3>3. Authorization</h3><p>Persona access is applied before evidence is sent to generation. Restricted source text is not used for unauthorized answers.</p></article>
    <article class="panel"><h3>4. Grounded Answer</h3><p><code>defence_agent/grounding.py</code> sends selected documents to Cohere Chat and reads native citation spans back into the audit object.</p></article>
    <article class="panel"><h3>5. Trace</h3><p><code>defence_agent/session.py</code> and the UI view model turn tool calls, retrieval audits, sources, exclusions, and citations into the trace shown above.</p></article>
    <article class="panel"><h3>6. Evaluation</h3><p><code>defence_agent/data/evals/demo_query_registry.yaml</code> and the eval reports check expected sources, refusals, citation resolution, and search behavior.</p></article>
  </div>
</section>
"""


def _document_row(doc: dict[str, Any]) -> str:
    title = str(doc.get("title") or doc.get("title_en") or "")
    return f"""
        <tr>
          <td><code>{escape(str(doc.get("doc_id", "")))}</code></td>
          <td>{escape(title)}</td>
          <td>{_pill(str(doc.get("access_level", "unknown")))}</td>
          <td>{escape(str(doc.get("language", "")))}</td>
          <td>{escape(_source_kind(doc))}</td>
          <td>{escape(str(doc.get("owner") or doc.get("source_organization") or ""))}</td>
        </tr>
"""


def _evidence_row(source: dict[str, Any]) -> str:
    label = source.get("label") or source.get("citation_id") or ""
    doc_id = source.get("doc_id") or ""
    page = source.get("page") or ""
    title = source.get("title") or ""
    scores = []
    if source.get("vector_score") not in (None, ""):
        scores.append(f"vector {_short_value(source.get('vector_score'))}")
    if source.get("rerank_score") not in (None, ""):
        scores.append(f"rerank {_short_value(source.get('rerank_score'))}")
    source_url = source.get("canonical_url") or source.get("source_url") or source.get("source_pdf_url") or ""
    link = (
        f'<a href="{escape(str(source_url))}" target="_blank" rel="noreferrer">source</a>'
        if source_url
        else "not recorded"
    )
    return f"""
          <tr>
            <td>{escape(str(label))}</td>
            <td><code>{escape(str(doc_id))}</code><br><span>{escape(str(title))}</span></td>
            <td>{escape(str(page))}</td>
            <td>{_pill(str(source.get("access_level", "unknown")))}</td>
            <td>{escape(', '.join(scores) or "not recorded")}</td>
            <td>{link}</td>
          </tr>
"""


def _coverage_card(coverage: dict[str, Any], validation: dict[str, Any]) -> str:
    if not coverage and not validation:
        return ""
    claim_count = coverage.get("claim_count", "not recorded")
    covered = coverage.get("covered_claim_count", "not recorded")
    uncited = coverage.get("uncited_claim_count", "not recorded")
    passed = validation.get("passed")
    status = "passed" if passed is True else "recorded"
    return f"""
      <div class="callout">
        <strong>Citation coverage: {escape(status)}</strong>
        <span>{escape(str(covered))} covered claim(s) / {escape(str(claim_count))} total. {escape(str(uncited))} uncited claim(s).</span>
      </div>
"""


def _answerability_summary(retrieval: dict[str, Any]) -> str:
    answerability = retrieval.get("answerability") or []
    if not answerability:
        return ""
    latest = answerability[-1] if isinstance(answerability[-1], dict) else {}
    reason = latest.get("reason") or latest.get("answerability_reason") or "recorded"
    evidence_quality = latest.get("evidence_quality") or {}
    status = evidence_quality.get("status") or reason
    return f"""
      <div class="callout">
        <strong>Answerability: {escape(str(status))}</strong>
        <span>Reason: {escape(str(reason))}</span>
      </div>
"""


def _follow_up_block(data: dict[str, Any]) -> str:
    follow_up = data.get("follow_up") or {}
    if not isinstance(follow_up, dict) or not follow_up.get("query"):
        return ""
    return f"""
      <h3>Follow-Up</h3>
      <p class="query">{escape(str(follow_up.get("query", "")))}</p>
"""


def _searches(data: dict[str, Any], audit: dict[str, Any], retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    searches = retrieval.get("searches")
    if isinstance(searches, list) and searches:
        return [item for item in searches if isinstance(item, dict)]
    records = audit.get("tool_call_records")
    if isinstance(records, list) and records:
        output = []
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            args = record.get("args") if isinstance(record.get("args"), dict) else {}
            output.append(
                {
                    "query": args.get("query") or record.get("query") or "",
                    "policy_decision": _list_get(retrieval.get("policy_decisions"), index - 1),
                    "filters_applied": _list_get(retrieval.get("filters_applied"), index - 1),
                }
            )
        if output:
            return output
    queries = retrieval.get("search_queries") or []
    if not queries and data.get("query"):
        queries = [data.get("query")]
    output = []
    for index, query in enumerate(queries, start=1):
        output.append(
            {
                "query": query,
                "policy_decision": _list_get(retrieval.get("policy_decisions"), index - 1),
                "filters_applied": _list_get(retrieval.get("filters_applied"), index - 1),
            }
        )
    return output


def _fallback_searches(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    queries = retrieval.get("search_queries") or []
    return [{"query": query} for query in queries]


def _audit(data: dict[str, Any]) -> dict[str, Any]:
    audit = data.get("final_answer_audit") or data.get("first_answer_audit") or {}
    return audit if isinstance(audit, dict) else {}


def _unique_sources(sources: list[Any]) -> list[dict[str, Any]]:
    seen = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        key = (
            source.get("source_id")
            or source.get("chunk_id")
            or source.get("citation_id")
            or f"{source.get('doc_id')}:{source.get('page')}"
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _doc_ids_from_sources(sources: list[dict[str, Any]]) -> list[str]:
    values = []
    for source in sources:
        doc_id = str(source.get("doc_id", "") or "")
        if doc_id and doc_id not in values:
            values.append(doc_id)
    return values


def _metric_card(label: str, value: str) -> str:
    return f'<div class="metric"><span>{escape(value)}</span><label>{escape(label)}</label></div>'


def _badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{escape(kind)}">{escape(text)}</span>'


def _pill(text: str) -> str:
    kind = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unknown"
    return f'<span class="pill pill-{escape(kind)}">{escape(text)}</span>'


def _text_block(text: str) -> str:
    return escape(text).replace("\n", "<br>")


def _source_kind(doc: dict[str, Any]) -> str:
    return str(doc.get("source_format") or doc.get("source_type") or doc.get("normalized_format") or "unknown")


def _counter_label(counter: Counter[str]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items())) or "none"


def _filters_label(filters: Any) -> str:
    if not isinstance(filters, dict) or not filters:
        return "filters not recorded"
    parts = []
    for key, value in filters.items():
        if isinstance(value, list):
            value_label = ", ".join(map(str, value))
        else:
            value_label = str(value)
        parts.append(f"{key}={value_label}")
    return "filters: " + "; ".join(parts)


def _short_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _list_get(value: Any, index: int) -> Any:
    if isinstance(value, list) and 0 <= index < len(value):
        return value[index]
    return None


def _anchor_for_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if "access" in slug and "unclassified" in slug:
        return "access-unclassified"
    if "access" in slug and "secret" in slug:
        return "access-secret"
    if "scanned" in slug:
        return "scanned"
    if "refusal" in slug:
        return "refusal"
    if "follow" in slug:
        return "follow-up"
    return slug


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing transcript: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


CSS = r"""
:root {
  --bg: #f6f7f4;
  --panel: #ffffff;
  --text: #17211d;
  --muted: #64706a;
  --line: #dfe5df;
  --green: #266756;
  --green-soft: #e5f1ec;
  --blue: #215b7d;
  --blue-soft: #e4eef5;
  --amber: #8a5a13;
  --amber-soft: #f7ecd7;
  --red-soft: #f4e4e0;
  --shadow: 0 16px 40px rgba(23, 33, 29, 0.08);
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  color: var(--text);
  background: var(--bg);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

a {
  color: var(--blue);
}

code {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 0.92em;
}

.shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 72px;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
  gap: 24px;
  align-items: stretch;
  padding: 28px;
  border: 1px solid var(--line);
  background: var(--panel);
  box-shadow: var(--shadow);
}

.hero h1,
.section h2,
.panel h3,
.citation-card h4 {
  margin: 0;
  line-height: 1.15;
}

.hero h1 {
  font-size: clamp(34px, 5vw, 60px);
  max-width: 820px;
}

.lead {
  max-width: 760px;
  margin: 18px 0 0;
  color: var(--muted);
  font-size: 18px;
}

.eyebrow {
  margin: 0 0 10px;
  color: var(--green);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

.hero-panel,
.metric-row,
.trace-meta,
.map-grid {
  display: grid;
  gap: 12px;
}

.hero-panel {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.stat,
.metric,
.panel,
.details {
  border: 1px solid var(--line);
  background: var(--panel);
}

.stat,
.metric {
  padding: 14px;
}

.stat span,
.metric span {
  display: block;
  font-weight: 800;
  overflow-wrap: anywhere;
}

.stat label,
.metric label {
  display: block;
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
}

.quick-nav {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 18px 0 0;
  padding: 10px;
  border: 1px solid var(--line);
  background: rgba(246, 247, 244, 0.94);
  backdrop-filter: blur(10px);
}

.quick-nav a {
  padding: 8px 11px;
  border: 1px solid var(--line);
  color: var(--text);
  background: var(--panel);
  text-decoration: none;
  font-size: 13px;
  font-weight: 700;
}

.section {
  margin-top: 28px;
  scroll-margin-top: 84px;
}

.section-head {
  margin-bottom: 14px;
}

.section-head h2 {
  font-size: 30px;
}

.section-head p:not(.eyebrow) {
  max-width: 840px;
  margin: 10px 0 0;
  color: var(--muted);
}

.metric-row {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-bottom: 12px;
}

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  background: var(--panel);
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 11px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}

th {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}

td span {
  color: var(--muted);
}

.run-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(290px, 0.65fr);
  gap: 14px;
}

.panel {
  padding: 18px;
}

.panel-top {
  display: flex;
  gap: 12px;
  justify-content: space-between;
  align-items: flex-start;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: flex-end;
}

.badge,
.pill {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 4px 8px;
  border: 1px solid var(--line);
  background: #f9faf8;
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.badge-good,
.pill-unclassified {
  border-color: #b9d6c8;
  background: var(--green-soft);
  color: var(--green);
}

.badge-warn,
.pill-secret,
.pill-top-secret {
  border-color: #e2c58f;
  background: var(--amber-soft);
  color: var(--amber);
}

.query {
  margin: 12px 0 18px;
  padding: 14px;
  border-left: 4px solid var(--blue);
  background: var(--blue-soft);
  font-weight: 700;
}

.answer {
  margin-top: 10px;
  padding: 16px;
  border: 1px solid var(--line);
  background: #fbfcfa;
  white-space: normal;
}

.presenter-panel {
  background: #fbfcfa;
}

.compact-dl {
  margin: 14px 0 0;
}

.compact-dl div {
  display: grid;
  grid-template-columns: 90px 1fr;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid var(--line);
}

.compact-dl dt {
  color: var(--muted);
  font-weight: 700;
}

.compact-dl dd {
  margin: 0;
  overflow-wrap: anywhere;
}

.callout {
  display: grid;
  gap: 4px;
  margin-top: 12px;
  padding: 12px;
  border: 1px solid #cdded6;
  background: var(--green-soft);
}

.callout span {
  color: var(--muted);
}

.details {
  margin-top: 12px;
}

.details summary {
  cursor: pointer;
  padding: 14px 16px;
  font-weight: 800;
}

.details > *:not(summary) {
  margin-left: 16px;
  margin-right: 16px;
}

.trace {
  padding-bottom: 16px;
}

.trace-meta {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-bottom: 12px;
}

.trace-list {
  display: grid;
  gap: 10px;
  margin: 0 16px 0 34px;
  padding: 0 0 2px;
}

.trace-list li {
  padding: 12px;
  border: 1px solid var(--line);
  background: #fbfcfa;
}

.trace-list span {
  display: block;
  color: var(--muted);
}

.mini-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.mini-metrics span {
  padding: 4px 7px;
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--text);
  font-size: 12px;
}

.compact th,
.compact td {
  font-size: 13px;
}

.muted {
  color: var(--muted);
}

.citation-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  padding-bottom: 16px;
}

.citation-card {
  padding: 12px;
  border: 1px solid var(--line);
  background: #fbfcfa;
}

.citation-card p {
  margin: 8px 0;
}

.citation-card small {
  color: var(--muted);
}

.map-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.run-primary .section-head .eyebrow {
  color: var(--blue);
}

.run-warn .query {
  border-left-color: var(--amber);
  background: var(--amber-soft);
}

.run-secure .query {
  border-left-color: var(--green);
  background: var(--green-soft);
}

@media (max-width: 900px) {
  .hero,
  .run-grid,
  .metric-row,
  .trace-meta,
  .map-grid {
    grid-template-columns: 1fr;
  }

  .citation-grid {
    grid-template-columns: 1fr;
  }
}

@media print {
  body {
    background: #fff;
  }

  .shell {
    width: 100%;
    padding: 0;
  }

  .quick-nav {
    display: none;
  }

  .hero,
  .panel,
  .details,
  .table-wrap,
  .stat,
  .metric {
    box-shadow: none;
    break-inside: avoid;
  }

  .section {
    break-inside: avoid;
  }
}
"""


if __name__ == "__main__":
    main()
