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
    "direct_rag",
    "version_comparison",
    "table_analysis",
    "multi_source_synthesis",
    "ambiguous_query",
    "restricted_access",
    "security_test",
    "human_review",
]
DEMO_QUERIES = [
    "What review steps should planning staff complete before approving a cross-unit planning request?",
    "Compare the 2024 and 2025 review gate procedure. What changed and what is the impact?",
    "Using the readiness review table, which units fall below the 80% readiness threshold?",
    "What does Restricted Annex B say about exception handling?",
    "Summarize the exception handling guidance from the test document.",
]


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
    linked_answer = re.sub(r"\[C(\d+)\]", replacement, escaped_answer)
    st.markdown(f'<div class="answer-copy">{linked_answer}</div>', unsafe_allow_html=True)


def source_label(citation: dict[str, Any], source: dict[str, Any] | None) -> str:
    title = citation.get("title") or (source or {}).get("title") or "Source"
    section = citation.get("section") or (source or {}).get("section") or "Section"
    page = citation.get("page") or (source or {}).get("page")
    page_text = f"page {page}" if page else "page unknown"
    return f"{title} | {section} | {page_text}"


def render_source_cards(data: dict[str, Any], persona: str) -> None:
    pairs = citation_source_pairs(data)
    if not pairs:
        return

    st.markdown("#### Sources")
    st.caption("Open a source to inspect the exact passage used for the answer.")
    selected = st.session_state.get("selected_source")
    for citation, source in pairs:
        citation_id = citation["id"]
        source = source or {}
        classification = citation.get("classification") or source.get("classification") or "unknown"
        version = citation.get("version") or source.get("version") or "unknown"
        effective_date = citation.get("effective_date") or source.get("effective_date") or "unknown"
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
                <span>Version {html.escape(str(version))}</span>
                <span>Effective {html.escape(str(effective_date))}</span>
              </div>
              <p>{html.escape(preview)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open source", key=f"open-{citation_id}-{source.get('chunk_id', '')}", width="content"):
            selected = {"citation": citation, "source": source}
            st.session_state["selected_source"] = selected
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


def run_query(query: str, persona: str, route_override: str, debug: bool, use_streaming: bool) -> None:
    payload = {"query": query, "route_override": route_override, "debug": debug}
    progress = st.empty()
    streamed_answer = st.empty()
    try:
        if use_streaming:
            final_data = None
            for event_name, event_data in api_stream("/v1/agent/stream", persona, payload):
                if event_name == "trace":
                    st.session_state["last_trace_id"] = event_data["trace_id"]
                elif event_name == "stage":
                    progress.info(event_data["label"])
                elif event_name == "route":
                    progress.info("Retrieving authorized evidence")
                elif event_name == "delta":
                    continue
                elif event_name == "final":
                    final_data = event_data
                elif event_name == "error":
                    progress.error(event_data["error"])
            if final_data:
                st.session_state["last_response"] = final_data
                st.session_state["last_trace_id"] = final_data["trace_id"]
                st.session_state.pop("selected_source", None)
        else:
            progress.info("Running route, retrieval, tools, generation, and safety checks")
            response = api_request("POST", "/v1/agent/query", persona, json=payload)
            if response.ok:
                data = response.json()
                st.session_state["last_response"] = data
                st.session_state["last_trace_id"] = data["trace_id"]
                st.session_state.pop("selected_source", None)
            else:
                st.error(response.text)
    except Exception as exc:
        st.error(f"Request failed: {exc}")
    finally:
        progress.empty()
        streamed_answer.empty()


def render_user_view(persona: str, route_override: str, debug: bool, use_streaming: bool) -> None:
    st.markdown('<div class="view-heading">Ask Defence Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="view-subtitle">Ask approved manuals, procedures, doctrine, and tables. Answers stay grounded in authorized sources.</div>',
        unsafe_allow_html=True,
    )
    query = st.text_area("Question", value=st.session_state.get("query", DEMO_QUERIES[0]), height=120)
    if st.button("Ask Defence Agent", type="primary", width="content"):
        run_query(query, persona, route_override, debug, use_streaming)

    data = st.session_state.get("last_response")
    if not data:
        return

    st.markdown("### Answer")
    render_answer(data["answer"], data)
    if data.get("needs_human_review"):
        st.warning("Human review is recommended before using this answer.")
    render_source_cards(data, persona)

    st.markdown("### Feedback")
    feedback_cols = st.columns([1, 1, 3])
    if feedback_cols[0].button("Answered my question", key="feedback-yes"):
        api_request("POST", "/v1/feedback", persona, json={"trace_id": data["trace_id"], "helpful": True})
        st.toast("Feedback logged")
    if feedback_cols[1].button("Needs work", key="feedback-no"):
        api_request("POST", "/v1/feedback", persona, json={"trace_id": data["trace_id"], "helpful": False})
        st.toast("Feedback logged")


def render_trace_tab(persona: str) -> None:
    data = st.session_state.get("last_response", {})
    stats = [
        ("Route", data.get("route", "not run")),
        ("Trace", str(data.get("trace_id", ""))[-8:] or "none"),
        ("Sources", len(data.get("sources", []))),
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
    if st.button("Run evals"):
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


st.set_page_config(page_title="Defence Agent", layout="wide")
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {display: none;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1220px;}
    .demo-title {font-size: 2.15rem; font-weight: 760; letter-spacing: 0; line-height: 1.12; margin: 0 0 0.25rem 0;}
    .demo-subtitle {color: #9aa8ba; font-size: 1rem; margin-bottom: 1rem;}
    .view-heading {font-size: 1.45rem; font-weight: 720; line-height: 1.2; margin: 0.25rem 0 0.25rem 0;}
    .view-subtitle {color: #9aa8ba; margin-bottom: 1rem;}
    .answer-copy {font-size: 1.08rem; line-height: 1.75; color: #f8fafc; max-width: 980px;}
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
        color: #72f0ad !important;
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
        border: 1px solid rgba(148, 163, 184, 0.26);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0 0.35rem 0;
        background: #0f1720;
    }
    .source-card:target {border-color: #72f0ad; box-shadow: 0 0 0 2px rgba(114, 240, 173, 0.18);}
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
    .source-title {font-weight: 700; color: #f8fafc;}
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
        border-left: 4px solid #72f0ad;
        background: #101821;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1rem;
        line-height: 1.62;
        color: #f8fafc;
    }
    .pdf-preview {width: 100%; min-height: 540px; border: 1px solid rgba(148, 163, 184, 0.28); border-radius: 8px;}
    .console-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
        gap: 0.6rem;
        margin: 0.8rem 0 1rem 0;
    }
    .console-stat {
        border: 1px solid rgba(148, 163, 184, 0.26);
        border-radius: 8px;
        background: #101821;
        padding: 0.65rem 0.75rem;
        min-width: 0;
    }
    .console-stat div {color: #94a3b8; font-size: 0.76rem; margin-bottom: 0.25rem;}
    .console-stat strong {
        color: #f8fafc;
        display: block;
        font-size: 0.94rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    div[data-testid="stMetric"] {
        background: #101821;
        border: 1px solid rgba(148, 163, 184, 0.26);
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
    }
    div[data-testid="stAlert"] {border-radius: 8px;}
    textarea {font-size: 1rem !important;}
    @media (max-width: 760px) {
        .demo-title {font-size: 1.75rem;}
        .answer-copy {font-size: 1rem;}
        .source-card-top {align-items: flex-start;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="demo-title">Defence Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="demo-subtitle">Secure, evidence-grounded doctrine assistance for fictional DefTech planning staff.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo Control Panel")
    persona = st.selectbox("Persona", PERSONAS, index=0)
    route_override = st.selectbox("Route override", ROUTES, index=0)
    debug = st.toggle("Debug trace", value=True)
    use_streaming = st.toggle("Stream answer", value=True)
    try:
        health = requests.get(f"{API_URL}/healthz", timeout=5).json()
        mode = "Mock Cohere" if health.get("mock_cohere") else "Real Cohere"
        st.info(f"{mode}\n\nChat: {health.get('chat_model')}")
    except Exception as exc:
        st.error(f"Backend unavailable: {exc}")
        health = {}

    st.subheader("Demo Queries")
    for index, demo_query in enumerate(DEMO_QUERIES, start=1):
        if st.button(f"{index}. {demo_query[:42]}...", width="stretch"):
            st.session_state["query"] = demo_query
    st.caption("Use User View for the product demo. Use Demo Console for trace, eval, security, and corpus details.")

mode = st.radio("View", ["User View", "Demo Console"], horizontal=True, label_visibility="collapsed")
if mode == "User View":
    render_user_view(persona, route_override, debug, use_streaming)
else:
    render_demo_console(persona, health)
