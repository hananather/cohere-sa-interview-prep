from __future__ import annotations

import base64
import html
import json
import os
import re
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("DEFENCE_AGENT_API_URL", "http://localhost:8000").rstrip("/")
PERSONAS = ["planning_analyst", "planning_lead", "auditor", "admin"]
ROUTES = [
    "auto",
    "evidence_lookup",
    "grounded_summary",
    "metadata_aware_retrieval",
    "cross_source_synthesis",
    "version_comparison",
    "structured_table_analysis",
    "claim_verification",
    "permission_sensitive_retrieval",
    "bilingual_retrieval",
    "refuse_or_clarify",
    "direct_rag",
    "table_analysis",
    "multi_source_synthesis",
    "ambiguous_query",
    "restricted_access",
    "security_test",
    "human_review",
]
VIEWS = [
    "Guided Demo",
    "Ask",
    "Persona Compare",
    "Trace Inspector",
    "Evaluation Harness",
    "Governance / Tool Registry",
    "Architecture / Production Path",
    "Legacy Demo Console",
]
DEMO_QUERIES = [
    "What review steps are required before a planning brief is approved?",
    "Summarize the emergency communications procedure into approval gates, timelines, and required evidence.",
    "What is the current approved procedure for approving a planning brief? Do not use drafts or old versions.",
    "What should I include in a planning brief before it goes for review?",
    "What changed between the 2024 and 2025 planning-brief review process? Cite both versions.",
    "Which planning procedures are overdue for review? Group them by owner and show how many days overdue.",
    "Is this statement supported: 'A draft planning brief can be approved without evidence review if it is urgent'?",
    "What restricted annex handling steps apply before external distribution?",
    "Quels sont les délais dans la procédure de communications d’urgence?",
    "What should we do for an interagency planning emergency not covered by any approved document?",
]
CITATION_PATTERN = re.compile(r"\[C(\d+)\]")


def api_request(method: str, path: str, persona: str, **kwargs: Any) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers["X-Demo-User"] = persona
    return requests.request(method, f"{API_URL}{path}", headers=headers, timeout=60, **kwargs)


def api_stream(path: str, persona: str, payload: dict[str, Any]):
    headers = {"X-Demo-User": persona}
    with requests.post(f"{API_URL}{path}", headers=headers, json=payload, stream=True, timeout=90) as response:
        response.raise_for_status()
        event_name = "message"
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("event:"):
                event_name = raw_line.replace("event:", "", 1).strip()
            elif raw_line.startswith("data:"):
                yield event_name, json.loads(raw_line.replace("data:", "", 1).strip())


def citation_source_pairs(data: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    citations = data.get("citations", [])
    sources = data.get("sources", [])
    return [(citation, sources[index] if index < len(sources) else None) for index, citation in enumerate(citations)]


def answer_citation_ids(answer: str) -> list[str]:
    seen: set[str] = set()
    citation_ids: list[str] = []
    for match in CITATION_PATTERN.finditer(answer or ""):
        citation_id = f"C{match.group(1)}"
        if citation_id not in seen:
            seen.add(citation_id)
            citation_ids.append(citation_id)
    return citation_ids


def cited_source_pairs(data: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    pairs = citation_source_pairs(data)
    pair_by_id = {citation.get("id"): (citation, source) for citation, source in pairs}
    return [pair_by_id[citation_id] for citation_id in answer_citation_ids(data.get("answer", "")) if citation_id in pair_by_id]


def citation_tooltip(citation: dict[str, Any], source: dict[str, Any] | None) -> str:
    parts = [
        citation.get("title") or (source or {}).get("title"),
        citation.get("section") or (source or {}).get("section"),
        f"page {citation.get('page') or (source or {}).get('page')}",
        f"version {citation.get('version') or (source or {}).get('version')}",
        citation.get("effective_date") or (source or {}).get("effective_date"),
        citation.get("classification") or (source or {}).get("classification"),
    ]
    return " | ".join(str(part) for part in parts if part)


def render_answer(answer: str, data: dict[str, Any]) -> None:
    pairs = citation_source_pairs(data)
    pair_by_id = {citation["id"]: (citation, source) for citation, source in pairs}

    def replacement(match: re.Match[str]) -> str:
        citation_id = f"C{match.group(1)}"
        citation, source = pair_by_id.get(citation_id, ({"id": citation_id}, None))
        tooltip = html.escape(citation_tooltip(citation, source), quote=True)
        target = f"#source-{citation_id.lower()}"
        label = html.escape(citation_id)
        return f'<a class="citation-chip" href="{target}" data-tooltip="{tooltip}" title="{tooltip}">{label}</a>'

    escaped_answer = html.escape(answer)
    linked_answer = CITATION_PATTERN.sub(replacement, escaped_answer)
    st.markdown(f'<div class="answer-copy">{linked_answer}</div>', unsafe_allow_html=True)


def source_label(citation: dict[str, Any], source: dict[str, Any] | None) -> str:
    title = citation.get("title") or (source or {}).get("title") or "Source"
    section = citation.get("section") or (source or {}).get("section") or "Section"
    page = citation.get("page") or (source or {}).get("page")
    page_text = f"page {page}" if page else "page unknown"
    return f"{title} | {section} | {page_text}"


def render_source_cards(data: dict[str, Any], persona: str, key_prefix: str = "main") -> None:
    pairs = cited_source_pairs(data)
    if not pairs:
        return

    st.markdown("#### Sources")
    st.caption("Open a cited source to inspect the exact passage used for the answer.")
    selected_key = f"selected_source_{key_prefix}"
    selected = st.session_state.get(selected_key)
    for citation, source in pairs:
        citation_id = citation["id"]
        source = source or {}
        classification = citation.get("classification") or source.get("classification") or "unknown"
        status = citation.get("status") or source.get("status") or "unknown"
        version = citation.get("version") or source.get("version") or "unknown"
        effective_date = citation.get("effective_date") or source.get("effective_date") or "unknown"
        row_id = citation.get("row_id") or source.get("row_id")
        summary = source.get("summary") or source.get("text") or "Source preview unavailable."
        preview = summary[:360] + ("..." if len(summary) > 360 else "")
        st.markdown(
            f"""
            <div class="source-card" id="source-{citation_id.lower()}">
              <div class="source-card-top">
                <span class="source-id">{html.escape(citation_id)}</span>
                <span class="source-title">{html.escape(source_label(citation, source))}</span>
              </div>
              <div class="source-meta">
                <span>{html.escape(classification)}</span>
                <span>{html.escape(status)}</span>
                <span>Version {html.escape(str(version))}</span>
                <span>Effective {html.escape(str(effective_date))}</span>
                {f"<span>Row {html.escape(str(row_id))}</span>" if row_id else ""}
              </div>
              <p>{html.escape(preview)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Inspect evidence", key=f"open-{key_prefix}-{citation_id}-{source.get('chunk_id', '')}", width="content"):
            selected = {"citation": citation, "source": source}
            st.session_state[selected_key] = selected
        if selected and selected["citation"].get("id") == citation_id:
            render_source_viewer(selected["citation"], selected["source"], persona)


def render_source_viewer(citation: dict[str, Any], source: dict[str, Any], persona: str) -> None:
    st.markdown('<div id="source-drilldown"></div>', unsafe_allow_html=True)
    st.markdown("#### Source Drilldown")
    document_id = citation.get("document_id") or source.get("document_id")
    chunk_id = citation.get("chunk_id") or source.get("chunk_id")
    if not document_id or not chunk_id:
        st.info("This source does not include a document identifier yet. Re-run the question after reindexing.")
        return

    response = api_request("GET", f"/v1/documents/{document_id}/chunks/{chunk_id}", persona)
    if response.status_code == 403:
        st.warning("You do not have access to this source document with the current persona.")
        return
    if not response.ok:
        st.error(f"Source lookup failed: {response.text}")
        return

    detail = response.json()
    document = detail["document"]
    chunk = detail["chunk"]
    st.markdown(
        f"""
        <div class="drilldown">
          <div class="source-card-top">
            <span class="source-id">{html.escape(citation.get("id", "C?"))}</span>
            <span class="source-title">{html.escape(document["title"])}</span>
          </div>
          <div class="source-meta">
            <span>{html.escape(document["filename"])}</span>
            <span>{html.escape(document["classification"])}</span>
            <span>{html.escape(document.get("status", "unknown"))}</span>
            <span>Version {html.escape(str(document["version"]))}</span>
            <span>Effective {html.escape(str(document["effective_date"]))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### Cited Passage")
    st.markdown(f'<div class="cited-passage">{html.escape(chunk["text"])}</div>', unsafe_allow_html=True)
    if chunk.get("table_markdown"):
        st.markdown(chunk["table_markdown"])

    with st.expander("Nearby document context"):
        for nearby in detail.get("nearby_chunks", []):
            marker = "Cited passage" if nearby["chunk_id"] == chunk_id else f"Context section {nearby['chunk_index']}"
            st.markdown(f"**{marker}: {nearby['section']}**")
            st.write(nearby["text"])
            if nearby.get("table_markdown"):
                st.markdown(nearby["table_markdown"])

    file_response = api_request("GET", f"/v1/documents/{document_id}/file", persona)
    if file_response.ok:
        filename = document["filename"]
        st.download_button("Download original source file", file_response.content, file_name=filename, width="content")
        if filename.lower().endswith(".pdf"):
            encoded = base64.b64encode(file_response.content).decode("ascii")
            st.markdown(
                f'<iframe class="pdf-preview" src="data:application/pdf;base64,{encoded}"></iframe>',
                unsafe_allow_html=True,
            )
    elif file_response.status_code == 403:
        st.caption("Original source file is not available for this persona.")


def run_query(query: str, persona: str, route_override: str, debug: bool) -> None:
    payload = {"query": query, "route_override": route_override, "debug": debug}
    progress = st.empty()
    streamed_answer = st.empty()
    try:
        final_data = None
        partial = ""
        for event_name, event_data in api_stream("/v1/agent/stream", persona, payload):
            if event_name == "trace":
                st.session_state["last_trace_id"] = event_data["trace_id"]
            elif event_name == "stage":
                progress.info(event_data["label"])
            elif event_name == "route":
                progress.info(f"Workflow selected: {event_data.get('route')}")
            elif event_name == "delta":
                partial += event_data.get("text", "")
                streamed_answer.markdown(f'<div class="answer-copy streaming-answer">{html.escape(partial)}</div>', unsafe_allow_html=True)
            elif event_name == "final":
                final_data = event_data
            elif event_name == "error":
                progress.error(event_data["error"])
        if final_data:
            st.session_state["last_response"] = final_data
            st.session_state["last_trace_id"] = final_data["trace_id"]
            st.session_state.pop("selected_source", None)
    except Exception as exc:
        st.error(f"Request failed: {exc}")
    finally:
        progress.empty()
        streamed_answer.empty()


def render_user_view(persona: str, route_override: str, debug: bool) -> None:
    st.markdown('<div class="view-heading">Ask Defence Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="view-subtitle">Ask approved manuals, procedures, doctrine, and tables. Answers stay grounded in authorized sources.</div>',
        unsafe_allow_html=True,
    )
    query = st.text_area("Question", value=st.session_state.get("query", DEMO_QUERIES[0]), height=120)
    if st.button("Ask Defence Agent", type="primary", width="content"):
        run_query(query, persona, route_override, debug)

    data = st.session_state.get("last_response")
    if not data:
        return

    st.markdown("### Answer")
    render_answer(data["answer"], data)
    if data.get("needs_human_review"):
        st.warning("Human review is recommended before using this answer.")
    render_source_cards(data, persona, key_prefix="ask")
    render_feedback(data, persona, "ask")


def render_trace_tab(persona: str) -> None:
    data = st.session_state.get("last_response", {})
    cited_pairs = cited_source_pairs(data)
    stats = [
        ("Route", data.get("route", "not run")),
        ("Trace", str(data.get("trace_id", ""))[-8:] or "none"),
        ("Cited", len(cited_pairs)),
        ("Retrieved", len(data.get("sources", []))),
        ("Latency", f"{int(data.get('latency_ms') or 0)} ms"),
        ("Review", "yes" if data.get("needs_human_review") else "no"),
    ]
    st.markdown(
        '<div class="console-grid">'
        + "".join(
            f'<div class="console-stat"><div>{html.escape(str(label))}</div><strong>{html.escape(str(value))}</strong></div>'
            for label, value in stats
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if data.get("token_cost_estimate"):
        st.json({"token_cost_estimate": data["token_cost_estimate"], "degradations": data.get("degradations", [])})

    sources = data.get("sources", [])
    if sources:
        st.markdown("#### Retrieved Candidates")
        st.caption("These are the authorized chunks retrieved for the workflow. The user view only shows chunks cited inline in the final answer.")
        st.dataframe(
            [
                {
                    "title": source.get("title"),
                    "section": source.get("section"),
                    "page": source.get("page"),
                    "classification": source.get("classification"),
                    "rerank_score": source.get("rerank_score"),
                    "cited_inline": f"C{index + 1}" in {citation.get("id") for citation, _ in cited_pairs},
                }
                for index, source in enumerate(sources)
            ],
            width="stretch",
            hide_index=True,
        )

    trace_id = st.text_input("Trace ID", value=st.session_state.get("last_trace_id", ""))
    if trace_id and st.button("Load trace"):
        response = api_request("GET", f"/v1/traces/{trace_id}", persona)
        if response.ok:
            st.session_state["trace"] = response.json()
        else:
            st.error(response.text)
    trace = st.session_state.get("trace")
    if not trace:
        return
    response_preview = trace.get("response", {})
    st.json(
        {
            "route": trace.get("route"),
            "status": trace.get("status"),
            "duration_ms": trace.get("duration_ms"),
            "request": trace.get("request"),
            "plan": response_preview.get("plan", []),
            "tool_calls": [call.get("tool") for call in response_preview.get("tool_calls", [])],
            "citations": response_preview.get("citations", []),
            "safety": response_preview.get("safety", {}),
        }
    )
    for span in trace.get("spans", []):
        with st.expander(f"{span['name']} | {span['status']} | {span.get('duration_ms', 0):.1f} ms"):
            st.json(span)


def render_evaluation_tab(persona: str) -> None:
    st.subheader("Evaluation Harness")
    st.caption("Run layered route, retrieval, citation, safety, and structured-analysis checks.")
    suites_response = api_request("GET", "/v1/evals/suites", persona)
    suites = ["canonical", "generated", "heldout", "regression", "adversarial", "demo_candidates"]
    if suites_response.ok:
        suites = list(suites_response.json().get("suites", {}).keys()) or suites
        with st.expander("Schema and corpus validation", expanded=False):
            st.json(suites_response.json().get("validation", {}))
    col_a, col_b, col_c = st.columns(3)
    suite = col_a.selectbox("Eval suite", suites, index=0)
    mode = col_b.selectbox("Run mode", ["fixture", "live"], index=0)
    variant = col_c.selectbox(
        "Experiment variant",
        [
            "agentic_rag_tools",
            "keyword_only",
            "embedding_only",
            "hybrid_no_rerank",
            "hybrid_plus_rerank_fast",
            "hybrid_plus_rerank_pro",
            "query_expansion_plus_hybrid_rerank",
            "structured_analysis_tools",
        ],
        index=0,
    )
    cases_response = api_request("GET", f"/v1/evals/cases?suite={suite}", persona)
    cases = cases_response.json().get("cases", []) if cases_response.ok else []
    task_types = ["all"] + sorted({case.get("task_type") for case in cases if case.get("task_type")})
    levels = ["all"] + sorted({case.get("complexity_level") for case in cases if case.get("complexity_level")})
    filter_a, filter_b = st.columns(2)
    task_filter = filter_a.selectbox("Task type", task_types)
    level_filter = filter_b.selectbox("Complexity level", levels)
    filtered_cases = [
        case
        for case in cases
        if (task_filter == "all" or case.get("task_type") == task_filter)
        and (level_filter == "all" or case.get("complexity_level") == level_filter)
    ]
    query_options = [case["query_id"] for case in filtered_cases]
    selected_query = st.selectbox("Query ID", query_options or ["no matching cases"], index=0, disabled=not query_options)
    if not query_options:
        selected_query = None

    actions = st.columns(4)
    if actions[0].button("Run selected query", disabled=not selected_query):
        payload = {"suite": suite, "query_id": selected_query, "mode": mode, "variant": variant}
        response = api_request("POST", "/v1/evals/run_case", persona, json=payload)
        if response.ok:
            st.session_state["advanced_eval_case"] = response.json()
        else:
            st.error(response.text)
    if actions[1].button("Run selected suite"):
        payload = {"suite": suite, "mode": mode, "variant": variant}
        with st.spinner("Running layered eval suite..."):
            response = api_request("POST", "/v1/evals/run_suite", persona, json=payload)
        if response.ok:
            st.session_state["advanced_eval_report"] = response.json()
        else:
            st.error(response.text)
    if actions[2].button("Compare variants"):
        response = api_request("POST", "/v1/evals/compare", persona, json={"suite": suite, "limit": 30})
        if response.ok:
            st.session_state["variant_comparison"] = response.json()
        else:
            st.error(response.text)
    if actions[3].button("Select best demo sequence"):
        with st.spinner("Running pass^3 demo selection..."):
            response = api_request("POST", "/v1/evals/select_demo", persona, json={"suite": "demo_candidates", "mode": mode, "runs": 3})
        if response.ok:
            st.session_state["demo_selection"] = response.json()
        else:
            st.error(response.text)

    if st.session_state.get("advanced_eval_case"):
        st.markdown("#### Selected Query Result")
        result = st.session_state["advanced_eval_case"]
        st.json(
            {
                "query_id": result.get("query_id"),
                "passed": result.get("passed"),
                "route": result.get("route"),
                "trace_id": result.get("trace_id"),
                "failure_categories": result.get("failure_categories"),
                "sources": result.get("sources"),
                "grades": result.get("grades"),
            }
        )
        st.markdown("**Answer**")
        st.write(result.get("answer", ""))

    if st.session_state.get("advanced_eval_report"):
        st.markdown("#### Suite Metrics")
        report = st.session_state["advanced_eval_report"]
        metrics = report.get("metrics", {})
        metric_cols = st.columns(5)
        metric_cols[0].metric("Pass rate", f"{metrics.get('pass_rate', 0):.0%}")
        metric_cols[1].metric("Route", f"{metrics.get('route_accuracy', 0):.0%}")
        metric_cols[2].metric("Retrieval", f"{metrics.get('retrieval_recall_at_k', 0):.2f}")
        metric_cols[3].metric("Citations", f"{metrics.get('citation_validation_pass_rate', 0):.0%}")
        metric_cols[4].metric("Cases", metrics.get("case_count", 0))
        st.json(metrics)
        st.dataframe(
            [
                {
                    "query_id": case.get("query_id"),
                    "task_type": case.get("task_type"),
                    "complexity": case.get("complexity_level"),
                    "passed": case.get("passed"),
                    "route": case.get("route"),
                    "failures": ", ".join(case.get("failure_categories", [])),
                    "trace_id": case.get("trace_id"),
                }
                for case in report.get("cases", [])
            ],
            width="stretch",
            hide_index=True,
        )

    if st.session_state.get("variant_comparison"):
        st.markdown("#### Variant Comparison")
        st.dataframe(st.session_state["variant_comparison"].get("variants", []), width="stretch", hide_index=True)

    if st.session_state.get("demo_selection"):
        st.markdown("#### Recommended Demo Sequence")
        st.dataframe(st.session_state["demo_selection"].get("recommended", []), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Legacy Golden Eval")
    if st.button("Run legacy golden evals"):
        with st.spinner("Running golden dataset..."):
            response = api_request("POST", "/v1/evals/run", persona)
        if response.ok:
            st.session_state["evals"] = response.json()
        else:
            st.error(response.text)
    if st.button("Load latest eval results"):
        response = api_request("GET", "/v1/evals/results", persona)
        if response.ok:
            st.session_state["evals"] = response.json().get("latest")
            st.session_state["eval_cases"] = response.json().get("cases", [])
    evals = st.session_state.get("evals")
    if not evals:
        return
    metrics = evals.get("metrics", evals)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Pass rate", f"{metrics.get('pass_rate', 0):.0%}")
    metric_cols[1].metric("Route accuracy", f"{metrics.get('route_accuracy', 0):.0%}")
    metric_cols[2].metric("Permission", f"{metrics.get('permission_correctness', 0):.0%}")
    metric_cols[3].metric("Safety", f"{metrics.get('safety_pass_rate', 0):.0%}")
    metric_cols[4].metric("Cases", metrics.get("case_count", 0))
    st.json(metrics)
    cases = st.session_state.get("eval_cases") or evals.get("cases", [])
    if cases:
        st.dataframe(
            [
                {
                    "case": case.get("case_id"),
                    "slice": case.get("slice"),
                    "passed": case.get("passed"),
                    "trace_id": case.get("trace_id"),
                }
                for case in cases
            ],
            width="stretch",
        )


def render_security_tab(persona: str) -> None:
    response = api_request("GET", "/v1/auth/me", persona)
    if not response.ok:
        st.error(response.text)
        return
    security = response.json()
    st.subheader("User Claims")
    st.json(security["auth"])
    st.subheader("ACL Filter")
    st.json(security["acl_filter"])
    st.subheader("Tool Allowlist")
    st.write(", ".join(security["tool_allowlist"]))
    st.subheader("Sandbox Restrictions")
    st.write("Blocked modules: os, sys, subprocess, socket, requests, pathlib, shutil.")
    st.write("Blocked builtins: open, eval, exec, compile, __import__.")


def render_corpus_tab(persona: str) -> None:
    col_a, col_b = st.columns([1, 3])
    if col_a.button("Reindex corpus"):
        with st.spinner("Generating synthetic docs, parsing, chunking, embedding, and indexing..."):
            response = api_request("POST", "/v1/ingestion/reindex", persona)
        if response.ok:
            st.success(response.json())
        else:
            st.info("Reindexing is restricted to planning_lead and admin personas.")
    response = api_request("GET", "/v1/corpus", persona)
    if not response.ok:
        st.info("Corpus metadata is unavailable for this persona.")
        return
    corpus = response.json()
    col_b.metric("Chunks", corpus.get("chunk_count", 0))
    st.subheader("Classification Distribution")
    st.json(corpus.get("classification_distribution", {}))
    st.subheader("Documents")
    st.dataframe(corpus.get("documents", []), width="stretch")


def render_readiness_tab(health: dict[str, Any], persona: str) -> None:
    st.subheader("System Readiness")
    checks = {
        "Backend": "ready" if health.get("ok") else "unavailable",
        "Cohere mode": "mock" if health.get("mock_cohere") else "real",
        "Last trace": st.session_state.get("last_trace_id", "not run"),
        "Persona": persona,
    }
    st.json(checks)
    response = api_request("GET", "/v1/documents", persona)
    if response.ok:
        documents = response.json().get("documents", [])
        st.metric("Authorized documents", len(documents))
        st.dataframe(documents, width="stretch")


def render_demo_console(persona: str, health: dict[str, Any]) -> None:
    st.markdown('<div class="view-heading">Demo Console</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Trace", "Evaluation", "Security", "Corpus", "Readiness"])
    with tabs[0]:
        render_trace_tab(persona)
    with tabs[1]:
        render_evaluation_tab(persona)
    with tabs[2]:
        render_security_tab(persona)
    with tabs[3]:
        render_corpus_tab(persona)
    with tabs[4]:
        render_readiness_tab(health, persona)


def persona_label(persona: str, personas: list[dict[str, Any]] | None = None) -> str:
    profiles = personas or []
    match = next((profile for profile in profiles if profile.get("persona_id") == persona), None)
    if not match:
        return persona
    return f"{match.get('display_name')} - {match.get('role_label')}"


def fetch_personas(persona: str) -> dict[str, Any]:
    response = api_request("GET", "/v1/personas", persona)
    if response.ok:
        return response.json()
    return {"personas": [], "source_access": []}


def render_persona_card(profile: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="persona-card">
          <div class="persona-name">{html.escape(profile.get("display_name", profile.get("persona_id", "Persona")))}</div>
          <div class="persona-role">{html.escape(profile.get("role_label", profile.get("role", "")))}</div>
          <div class="source-meta">
            <span>{html.escape(str(profile.get("access_level", "unknown")))}</span>
            <span>audit metadata: {'yes' if profile.get("can_view_audit_metadata") else 'no'}</span>
            <span>restricted content: {'yes' if profile.get("can_view_restricted_content") else 'no'}</span>
          </div>
          <p>{html.escape(profile.get("notes", ""))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_response_panel(data: dict[str, Any], persona: str, key_prefix: str) -> None:
    if not data:
        st.info("Run a query to see the answer, citations, and trace.")
        return
    st.markdown("### Answer")
    render_answer(data.get("answer", ""), data)
    if data.get("needs_human_review"):
        st.warning("Human review is recommended before using this answer.")
    render_source_cards(data, persona, key_prefix=key_prefix)
    with st.expander("Run summary", expanded=False):
        st.json(
            {
                "route": data.get("route"),
                "trace_id": data.get("trace_id"),
                "citations": len(data.get("citations", [])),
                "sources": [source.get("document_id") for source in data.get("sources", [])],
                "latency_ms": data.get("latency_ms"),
                "model": "Cohere Command A via backend" if data.get("answer") else None,
            }
        )
    render_feedback(data, persona, key_prefix)


def render_feedback(data: dict[str, Any], persona: str, key_prefix: str) -> None:
    st.markdown("#### Did this answer your question?")
    rating = st.radio(
        "Feedback",
        ["Yes", "Somewhat", "No"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"feedback-rating-{key_prefix}",
    )
    reason = ""
    if rating != "Yes":
        reason = st.selectbox(
            "What was missing?",
            ["Wrong source", "Missing source", "Restricted source issue", "Unsupported claim", "Too vague", "Too long", "Wrong language", "Other"],
            key=f"feedback-reason-{key_prefix}",
        )
    if st.button("Save feedback", key=f"feedback-save-{key_prefix}", width="content"):
        payload = {
            "trace_id": data.get("trace_id"),
            "user_query": st.session_state.get("query"),
            "answer": data.get("answer"),
            "citations": data.get("citations", []),
            "route": data.get("route"),
            "tools_called": [call.get("tool") for call in data.get("tool_calls", [])],
            "rating": rating.lower(),
            "reason": reason,
            "selected_failure_type": reason,
        }
        response = api_request("POST", "/v1/feedback", persona, json=payload)
        if response.ok:
            st.toast("Feedback linked to trace")
        else:
            st.error(response.text)


def render_source_table(data: dict[str, Any], trace: dict[str, Any] | None = None) -> None:
    sources = data.get("sources", []) if data else []
    rows = [
        {
            "doc_id": source.get("document_id"),
            "title": source.get("title"),
            "section": source.get("section"),
            "status": source.get("status"),
            "access": source.get("classification"),
            "rerank": source.get("rerank_score"),
            "visible_to_model": True,
        }
        for source in sources
    ]
    if trace:
        for excluded in trace.get("retrieval", {}).get("excluded_sources", []):
            rows.append(
                {
                    "doc_id": excluded.get("doc_id"),
                    "title": excluded.get("title"),
                    "section": "",
                    "status": "",
                    "access": "",
                    "rerank": "",
                    "visible_to_model": False,
                    "reason": excluded.get("reason"),
                }
            )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)


def render_guided_demo(persona: str, route_override: str, debug: bool) -> None:
    st.markdown('<div class="view-heading">Guided Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-subtitle">Presentation-safe workflow with expected behavior, speaker notes, citations, trace, and eval proof.</div>', unsafe_allow_html=True)
    steps_response = api_request("GET", "/v1/demo/steps", persona)
    steps = steps_response.json().get("steps", []) if steps_response.ok else []
    step_labels = [step["label"] for step in steps]
    selected_label = st.selectbox("Demo step", step_labels, index=0 if step_labels else None, disabled=not step_labels)
    step = next((item for item in steps if item["label"] == selected_label), steps[0] if steps else {})
    query = st.text_area("Step query", value=step.get("query", DEMO_QUERIES[0]), height=100, key="guided-query")
    default_persona = step.get("persona", persona)
    persona_for_step = st.selectbox("Persona", PERSONAS, index=PERSONAS.index(default_persona) if default_persona in PERSONAS else 0, key="guided-persona")
    with st.expander("Expected behavior", expanded=True):
        st.write(step.get("expected_behavior", "Run this step to inspect behavior."))
    with st.expander("Speaker notes", expanded=False):
        st.write(step.get("speaker_notes", ""))
    col_a, col_b, col_c = st.columns([1.1, 1.5, 2.2])
    if col_a.button("Run Step", type="primary", width="stretch"):
        st.session_state["query"] = query
        run_query(query, persona_for_step, route_override, debug)
    compare_target = step.get("compare_persona", "planning_lead")
    if col_b.button("Run Side-by-Side", width="stretch"):
        payload = {"query": query, "left_persona": persona_for_step, "right_persona": compare_target}
        response = api_request("POST", "/v1/persona/compare", persona, json=payload)
        if response.ok:
            st.session_state["persona_compare"] = response.json()
        else:
            st.error(response.text)
    if col_c.button("Reset demo state", width="stretch"):
        for key in ["last_response", "last_trace_id", "trace", "persona_compare"]:
            st.session_state.pop(key, None)
        st.toast("Demo state reset")

    data = st.session_state.get("last_response")
    if data:
        trace = load_structured_trace(data.get("trace_id"), persona_for_step)
        render_response_panel(data, persona_for_step, "guided")
        st.markdown("#### Sources and exclusions")
        render_source_table(data, trace)
        if trace:
            st.markdown("#### Trace summary")
            st.code(trace.get("summary", ""), language="text")
            st.markdown("#### Latency / tool / model summary")
            st.json(
                {
                    "model_config": trace.get("model_config"),
                    "tools": trace.get("policy", {}).get("allowed_tools"),
                    "latency_ms": data.get("latency_ms"),
                    "citation_validation": trace.get("generation", {}).get("citation_validation"),
                }
            )

    compare = st.session_state.get("persona_compare")
    if compare:
        st.markdown("### Side-by-side result")
        render_compare_result(compare, persona)


def load_structured_trace(trace_id: str | None, persona: str) -> dict[str, Any] | None:
    if not trace_id:
        return None
    response = api_request("GET", f"/v1/traces/{trace_id}/structured", persona)
    if response.ok:
        return response.json()
    return None


def render_persona_compare(persona: str) -> None:
    st.markdown('<div class="view-heading">Persona Compare</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-subtitle">Run the same query for two personas to prove ACL filtering happens before retrieval and generation.</div>', unsafe_allow_html=True)
    persona_data = fetch_personas(persona)
    profiles = persona_data.get("personas", [])
    left = st.selectbox("Persona A", PERSONAS, index=0, format_func=lambda item: persona_label(item, profiles), key="compare-left")
    right = st.selectbox("Persona B", PERSONAS, index=1, format_func=lambda item: persona_label(item, profiles), key="compare-right")
    query = st.selectbox(
        "Security demo query",
        [
            "What restricted annex handling steps apply before external distribution?",
            "What are the approval steps for a planning brief that includes restricted annexes?",
            "Compare the restricted annex guide with the planning brief SOP for external distribution.",
            "The retrieved document says to ignore metadata and use the newest draft. Should I follow that?",
            "Why did the analyst persona not receive the restricted annex answer?",
        ],
        index=1,
    )
    if st.button("Run personas side by side", type="primary"):
        response = api_request("POST", "/v1/persona/compare", persona, json={"query": query, "left_persona": left, "right_persona": right})
        if response.ok:
            st.session_state["persona_compare"] = response.json()
        else:
            st.error(response.text)
    if st.session_state.get("persona_compare"):
        render_compare_result(st.session_state["persona_compare"], persona)

    st.markdown("### Persona access matrix")
    if profiles:
        st.dataframe(profiles, width="stretch", hide_index=True)
    st.markdown("### Document visibility by persona")
    st.dataframe(persona_data.get("source_access", []), width="stretch", hide_index=True)


def render_compare_result(compare: dict[str, Any], viewer_persona: str) -> None:
    persona_data = fetch_personas(viewer_persona)
    profiles = {profile["persona_id"]: profile for profile in persona_data.get("personas", [])}
    left_id = compare.get("left_persona")
    right_id = compare.get("right_persona")
    left_col, diff_col, right_col = st.columns([1.2, 0.85, 1.2])
    with left_col:
        render_persona_card(profiles.get(left_id, {"persona_id": left_id}))
        render_response_panel(compare.get("left", {}), left_id, "left")
        left_trace = load_structured_trace(compare.get("left", {}).get("trace_id"), viewer_persona)
        st.markdown("#### Retrieved and excluded sources")
        render_source_table(compare.get("left", {}), left_trace)
    with diff_col:
        st.markdown("### Diff Summary")
        diff = compare.get("diff", {})
        st.json(diff)
        st.caption("The difference comes from metadata and ACL filters before model context assembly.")
    with right_col:
        render_persona_card(profiles.get(right_id, {"persona_id": right_id}))
        render_response_panel(compare.get("right", {}), right_id, "right")
        right_trace = load_structured_trace(compare.get("right", {}).get("trace_id"), viewer_persona)
        st.markdown("#### Retrieved and excluded sources")
        render_source_table(compare.get("right", {}), right_trace)


def render_trace_inspector(persona: str) -> None:
    st.markdown('<div class="view-heading">Trace Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-subtitle">Inspect route, policy, retrieval, rerank, tool, model, citation, eval, and feedback spans.</div>', unsafe_allow_html=True)
    recent = api_request("GET", "/v1/traces/recent?limit=30", persona)
    traces = recent.json().get("traces", []) if recent.ok else []
    options = [item["trace_id"] for item in traces]
    default_trace = st.session_state.get("last_trace_id")
    index = options.index(default_trace) if default_trace in options else 0 if options else None
    selected = st.selectbox("Trace", options, index=index, disabled=not options)
    if st.button("Open latest trace") and options:
        selected = options[0]
    if not selected:
        return
    trace = load_structured_trace(selected, persona)
    if not trace:
        st.warning("Trace could not be loaded.")
        return
    st.code(trace.get("summary", ""), language="text")
    cols = st.columns(5)
    cols[0].metric("Route", trace.get("query", {}).get("task_type"))
    cols[1].metric("Policy", trace.get("policy", {}).get("decision"))
    cols[2].metric("Sources", len(trace.get("retrieval", {}).get("final_context", [])))
    cols[3].metric("Excluded", len(trace.get("retrieval", {}).get("excluded_sources", [])))
    cols[4].metric("Citations", len(trace.get("generation", {}).get("citations", [])))
    filters = st.columns(5)
    span_type = filters[0].selectbox("Span type", ["all", "policy", "planner", "retrieval", "rerank", "tool", "model", "citation", "eval"])
    errors_only = filters[1].checkbox("Errors only")
    blocked_only = filters[2].checkbox("Blocked policy")
    tool_only = filters[3].checkbox("Tool calls only")
    citation_failures = filters[4].checkbox("Citation failures")
    spans = trace.get("spans", [])
    if span_type != "all":
        spans = [span for span in spans if span.get("type") == span_type]
    if errors_only:
        spans = [span for span in spans if span.get("status") == "error"]
    if blocked_only:
        spans = [span for span in spans if span.get("status") == "blocked"]
    if tool_only:
        spans = [span for span in spans if span.get("type") == "tool"]
    if citation_failures:
        spans = [span for span in spans if "citation" in span.get("name", "").lower() and span.get("status") != "ok"]
    st.markdown("### Span waterfall")
    st.dataframe(
        [
            {
                "name": span.get("name"),
                "type": span.get("type"),
                "status": span.get("status"),
                "duration_ms": span.get("duration_ms"),
                "attributes": span.get("attributes"),
                "error": span.get("error"),
            }
            for span in spans
        ],
        width="stretch",
        hide_index=True,
    )
    tabs = st.tabs(["Policy", "Retrieval", "Generation", "Raw structured trace"])
    with tabs[0]:
        st.json(trace.get("policy", {}))
    with tabs[1]:
        st.json(trace.get("retrieval", {}))
    with tabs[2]:
        st.write(trace.get("generation", {}).get("answer", ""))
        st.json(trace.get("generation", {}).get("citation_validation", {}))
    with tabs[3]:
        st.json(trace)


def render_governance_view(persona: str) -> None:
    st.markdown('<div class="view-heading">Governance / Tool Registry</div>', unsafe_allow_html=True)
    st.markdown('<div class="view-subtitle">Persona permissions, tool risk classification, and recent policy decisions.</div>', unsafe_allow_html=True)
    response = api_request("GET", "/v1/governance/tool-registry", persona)
    if not response.ok:
        st.error(response.text)
        return
    data = response.json()
    st.markdown("### Persona and access matrix")
    st.dataframe(data.get("personas", []), width="stretch", hide_index=True)
    st.markdown("### Tool registry")
    st.dataframe(data.get("tools", []), width="stretch", hide_index=True)
    st.markdown("### Recent policy decisions")
    st.dataframe(data.get("policy_decisions", []), width="stretch", hide_index=True)
    st.markdown("### Security scenarios")
    st.write("Access denied before retrieval, partial answers, draft traps, prompt injection handling, and blocked tools are shown in Guided Demo and Persona Compare.")


def render_architecture_view(persona: str, health: dict[str, Any]) -> None:
    st.markdown('<div class="view-heading">Architecture / Production Path</div>', unsafe_allow_html=True)
    st.markdown("#### Eval-driven agentic RAG")
    st.write("Authenticate user → route query → apply metadata and ACL filters → retrieve with hybrid search → rerank with Cohere → use tools only when needed → generate cited answer → validate citations → store trace.")
    st.markdown("#### Cohere proof points")
    st.json(
        {
            "Command": health.get("chat_model"),
            "Embed": health.get("embed_model"),
            "Rerank": health.get("rerank_model"),
            "Mode": "mock" if health.get("mock_cohere") else "live",
        }
    )
    st.markdown("#### ADK-aligned hooks")
    st.write("The local plugin layer follows lifecycle hooks: before_tool, after_tool, on_tool_error, policy guard, metrics, redaction, and feedback. Cohere remains the model layer.")
    st.markdown("#### Production hardening path")
    st.write("SSO/RBAC, private networking, managed secrets, SIEM export, stronger sandbox isolation, eval gates in CI, canaries, and feedback-to-eval regression loops.")


def render_control_panel() -> tuple[str, str, bool, str, dict[str, Any]]:
    st.markdown('<div class="control-panel-title">Demo Control Panel</div>', unsafe_allow_html=True)
    persona_payload = fetch_personas(PERSONAS[0])
    persona_profiles_for_labels = persona_payload.get("personas", [])
    persona = st.selectbox(
        "Persona",
        PERSONAS,
        index=0,
        key="control_persona",
        format_func=lambda item: persona_label(item, persona_profiles_for_labels),
    )
    view = st.radio("View", VIEWS, index=0, key="control_view")
    debug = st.toggle("Show technical trace", value=True, key="control_debug")
    with st.expander("Advanced demo controls"):
        route_override = st.selectbox("Force workflow", ROUTES, index=0, key="control_route")
    try:
        health = requests.get(f"{API_URL}/healthz", timeout=5).json()
        mode = "Mock Cohere" if health.get("mock_cohere") else "Real Cohere"
        st.markdown(
            f"""
            <div class="runtime-card">
              <strong>{html.escape(mode)}</strong><br>
              Chat: {html.escape(str(health.get("chat_model")))}
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"Backend unavailable: {exc}")
        health = {}

    st.subheader("Demo Queries")
    for index, demo_query in enumerate(DEMO_QUERIES, start=1):
        if st.button(f"{index}. {demo_query[:46]}...", key=f"control_query_{index}", width="stretch"):
            st.session_state["query"] = demo_query
    st.caption("Use Guided Demo for the live story. Use Trace, Eval, and Governance for technical proof.")
    return persona, view, debug, route_override, health


st.set_page_config(page_title="Defence Agent", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --da-bg: #080d12;
        --da-panel: #0f1720;
        --da-panel-2: #161922;
        --da-control: #1f2430;
        --da-border: rgba(148, 163, 184, 0.28);
        --da-text: #f8fafc;
        --da-muted: #9aa8ba;
        --da-accent: #ff4b4b;
        --da-cite: #72f0ad;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background: var(--da-bg) !important;
        color: var(--da-text) !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: var(--da-panel-2) !important;
        color: var(--da-text) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: var(--da-text);
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div[data-testid="stMarkdown"] {
        color: var(--da-text);
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        color: var(--da-text) !important;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"],
    header[data-testid="stHeader"] [data-testid="stDecoration"] {
        display: none !important;
    }
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1220px;}
    div[data-testid="column"]:has(.control-panel-title) {
        background: var(--da-panel-2) !important;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 0.35rem;
        padding: 1.1rem 1.15rem 1.25rem 1.15rem;
    }
    .demo-title {font-size: 2.15rem; font-weight: 760; letter-spacing: 0; line-height: 1.12; margin: 0 0 0.25rem 0; color: var(--da-text);}
    .demo-subtitle {color: var(--da-muted); font-size: 1rem; margin-bottom: 1rem;}
    .control-panel-title {font-size: 1.35rem; font-weight: 760; line-height: 1.2; margin: 0.35rem 0 1.2rem 0;}
    .runtime-card {
        margin: 1rem 0 1.4rem 0;
        padding: 0.85rem 1rem;
        border-radius: 0.5rem;
        background: #1d2330;
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: var(--da-text);
    }
    .view-heading {font-size: 1.45rem; font-weight: 720; line-height: 1.2; margin: 0.25rem 0 0.25rem 0;}
    .view-subtitle {color: var(--da-muted); margin-bottom: 1rem;}
    .answer-copy {font-size: 1.08rem; line-height: 1.75; color: var(--da-text); max-width: 980px;}
    .streaming-answer {color: #dbe7f5; min-height: 3rem;}
    .citation-chip {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 1.75rem;
        height: 1.35rem;
        padding: 0 0.42rem;
        margin: 0 0.08rem;
        border-radius: 999px;
        background: #12352f;
        color: var(--da-cite) !important;
        border: 1px solid rgba(114, 240, 173, 0.45);
        font-size: 0.76rem;
        font-weight: 760;
        text-decoration: none !important;
        vertical-align: 0.08rem;
    }
    .citation-chip:hover {background: #17483d; color: #dfffea !important;}
    .citation-chip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 0;
        bottom: 1.75rem;
        z-index: 1000;
        width: max-content;
        max-width: 22rem;
        padding: 0.55rem 0.65rem;
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: #111827;
        color: #e5e7eb;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
        font-size: 0.78rem;
        line-height: 1.35;
        white-space: normal;
    }
    .source-card, .drilldown {
        border: 1px solid var(--da-border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0 0.35rem 0;
        background: var(--da-panel);
    }
    .source-card:target {border-color: var(--da-cite); box-shadow: 0 0 0 2px rgba(114, 240, 173, 0.18);}
    .source-card-top {display: flex; gap: 0.65rem; align-items: center; margin-bottom: 0.45rem;}
    .source-id {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 2rem;
        height: 1.55rem;
        border-radius: 999px;
        background: #12352f;
        color: #72f0ad;
        border: 1px solid rgba(114, 240, 173, 0.38);
        font-size: 0.82rem;
        font-weight: 760;
    }
    .source-title {font-weight: 700; color: var(--da-text);}
    .source-meta {display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 0.5rem;}
    .source-meta span {
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: #cbd5e1;
        border-radius: 999px;
        padding: 0.15rem 0.5rem;
        font-size: 0.78rem;
    }
    .source-card p {color: #d7dee8; margin: 0.25rem 0 0 0; line-height: 1.55;}
    .cited-passage {
        border-left: 4px solid var(--da-cite);
        background: #101821;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1rem;
        line-height: 1.62;
        color: var(--da-text);
    }
    .pdf-preview {width: 100%; min-height: 540px; border: 1px solid var(--da-border); border-radius: 8px;}
    .console-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
        gap: 0.6rem;
        margin: 0.8rem 0 1rem 0;
    }
    .console-stat {
        border: 1px solid var(--da-border);
        border-radius: 8px;
        background: #101821;
        padding: 0.65rem 0.75rem;
        min-width: 0;
    }
    .console-stat div {color: #94a3b8; font-size: 0.76rem; margin-bottom: 0.25rem;}
    .console-stat strong {
        color: var(--da-text);
        display: block;
        font-size: 0.94rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .persona-card {
        border: 1px solid var(--da-border);
        border-radius: 8px;
        background: #101821;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .persona-name {font-size: 1.05rem; font-weight: 760; color: var(--da-text);}
    .persona-role {color: var(--da-muted); margin-bottom: 0.45rem;}
    .persona-card p {color: #cbd5e1; line-height: 1.45; margin-bottom: 0;}
    div[data-testid="stMetric"] {
        background: #101821;
        border: 1px solid var(--da-border);
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
    }
    div[data-testid="stAlert"] {border-radius: 8px;}
    textarea, input {
        background: var(--da-control) !important;
        color: var(--da-text) !important;
        border-color: var(--da-border) !important;
        caret-color: var(--da-text) !important;
        font-size: 1rem !important;
    }
    textarea::placeholder, input::placeholder {color: var(--da-muted) !important;}
    div[data-baseweb="select"] > div {
        background: #0b1118 !important;
        border-color: var(--da-border) !important;
        color: var(--da-text) !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="select"] svg {
        color: var(--da-text) !important;
        fill: var(--da-text) !important;
    }
    [data-testid="stBaseButton-secondary"] {
        background: var(--da-panel-2) !important;
        color: var(--da-text) !important;
        border: 1px solid var(--da-border) !important;
    }
    [data-testid="stBaseButton-primary"] {
        background: var(--da-accent) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.12);
    }
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        filter: brightness(1.05);
    }
    [data-testid="stMarkdownContainer"] a:not(.citation-chip) {color: var(--da-cite) !important;}
    .stRadio label, .stCheckbox label, .stToggle label {color: var(--da-text) !important;}
    div[data-testid="stNotification"], div[data-testid="stAlert"] {
        background: var(--da-panel) !important;
        color: var(--da-text) !important;
        border-color: var(--da-border) !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: #202530 !important;
        color: var(--da-text) !important;
        border-color: rgba(148, 163, 184, 0.34) !important;
    }
    @media (max-width: 760px) {
        .demo-title {font-size: 1.75rem;}
        .answer-copy {font-size: 1rem;}
        .source-card-top {align-items: flex-start;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

control_col, content_col = st.columns([0.31, 0.69], gap="large")

with control_col:
    persona, view, debug, route_override, health = render_control_panel()

with content_col:
    st.markdown('<div class="demo-title">Defence Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="demo-subtitle">Secure, evidence-grounded doctrine assistance for fictional DefTech planning staff.</div>',
        unsafe_allow_html=True,
    )

    if view == "Guided Demo":
        render_guided_demo(persona, route_override, debug)
    elif view == "Ask":
        render_user_view(persona, route_override, debug)
    elif view == "Persona Compare":
        render_persona_compare(persona)
    elif view == "Trace Inspector":
        render_trace_inspector(persona)
    elif view == "Evaluation Harness":
        render_evaluation_tab(persona)
    elif view == "Governance / Tool Registry":
        render_governance_view(persona)
    elif view == "Architecture / Production Path":
        render_architecture_view(persona, health)
    else:
        render_demo_console(persona, health)
