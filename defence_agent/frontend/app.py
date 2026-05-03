from __future__ import annotations

import os
import json
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


st.set_page_config(page_title="Defence Agent", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px;}
    div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e5e7eb; padding: 0.7rem 0.85rem; border-radius: 8px;}
    div[data-testid="stAlert"] {border-radius: 8px;}
    .demo-title {font-size: 2.1rem; font-weight: 750; letter-spacing: 0; margin-bottom: 0.1rem;}
    .demo-subtitle {color: #475569; margin-bottom: 1.1rem;}
    .flow-card {border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.75rem; background: #ffffff;}
    .source-meta {font-size: 0.9rem; color: #475569;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="demo-title">Defence Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="demo-subtitle">Secure, evaluated, observable agentic RAG for fictional DefTech planning staff.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo Controls")
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
    st.caption("Core flow: ask, inspect trace, switch persona for restricted access, run evals.")

tabs = st.tabs(["Ask Defence Agent", "Trace", "Evaluation", "Security", "Corpus"])

with tabs[0]:
    query = st.text_area("Question", value=st.session_state.get("query", DEMO_QUERIES[0]), height=110)
    ask_clicked = st.button("Ask Defence Agent", type="primary", width="content")
    if ask_clicked:
        payload = {"query": query, "route_override": route_override, "debug": debug}
        answer_placeholder = st.empty()
        stage_placeholder = st.empty()
        streamed_answer = ""
        try:
            if use_streaming:
                for event_name, event_data in api_stream("/v1/agent/stream", persona, payload):
                    if event_name == "trace":
                        st.session_state["last_trace_id"] = event_data["trace_id"]
                    elif event_name == "stage":
                        stage_placeholder.info(event_data["label"])
                    elif event_name == "route":
                        stage_placeholder.info(f"Route selected: {event_data['route']}")
                    elif event_name == "delta":
                        streamed_answer += event_data["text"]
                        answer_placeholder.markdown(streamed_answer)
                    elif event_name == "final":
                        st.session_state["last_response"] = event_data
                        st.session_state["last_trace_id"] = event_data["trace_id"]
                        stage_placeholder.success("Answer generated, citations validated, trace stored.")
                    elif event_name == "error":
                        stage_placeholder.error(event_data["error"])
            else:
                with st.status("Running route, retrieval, tools, generation, and safety checks...", expanded=False):
                    response = api_request("POST", "/v1/agent/query", persona, json=payload)
                if response.ok:
                    data = response.json()
                    st.session_state["last_response"] = data
                    st.session_state["last_trace_id"] = data["trace_id"]
                else:
                    st.error(response.text)
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    data = st.session_state.get("last_response")
    if data:
        st.subheader("Answer")
        st.markdown(data["answer"])
        cols = st.columns(4)
        cols[0].metric("Route", data["route"])
        cols[1].metric("Trace", data["trace_id"][-8:])
        cols[2].metric("Sources", len(data.get("sources", [])))
        cols[3].metric("Latency ms", int(data.get("latency_ms") or 0))
        if data.get("token_cost_estimate"):
            st.caption(
                f"Token estimate: {data['token_cost_estimate'].get('total_tokens_estimate')} | "
                f"Cost estimate: ${data['token_cost_estimate'].get('estimated_cost_usd', 0):.4f}"
            )

        if data.get("degradations"):
            st.warning("Degradations: " + ", ".join(data["degradations"]))
        if data.get("needs_human_review"):
            st.error("Human review required")

        st.subheader("Citations")
        if not data.get("citations"):
            st.caption("No citations returned because the system abstained or requested clarification.")
        for citation in data.get("citations", []):
            st.markdown(f"- `{citation['id']}` {citation['title']} | {citation['section']} | page {citation['page']}")

        st.subheader("Sources")
        for source in data.get("sources", []):
            score = source.get("rerank_score")
            score_text = f"rerank {score:.3f}" if isinstance(score, float) else "hybrid fallback"
            with st.expander(f"{source['title']} | {source['section']} | {score_text}"):
                meta_cols = st.columns(5)
                meta_cols[0].write(f"Class: `{source['classification']}`")
                meta_cols[1].write(f"Version: `{source['version']}`")
                meta_cols[2].write(f"Page: `{source['page']}`")
                meta_cols[3].write(f"Lexical: `{source['lexical_score']}`")
                meta_cols[4].write(f"Vector: `{source['vector_score']}`")
                st.write(source["text"])
                if source.get("table_markdown"):
                    st.markdown(source["table_markdown"])

        st.subheader("Feedback")
        feedback_cols = st.columns(2)
        if feedback_cols[0].button("Yes, this answered it"):
            api_request("POST", "/v1/feedback", persona, json={"trace_id": data["trace_id"], "helpful": True})
            st.success("Feedback logged")
        if feedback_cols[1].button("No, needs work"):
            api_request("POST", "/v1/feedback", persona, json={"trace_id": data["trace_id"], "helpful": False})
            st.success("Feedback logged")

with tabs[1]:
    trace_id = st.text_input("Trace ID", value=st.session_state.get("last_trace_id", ""))
    if trace_id and st.button("Load trace"):
        response = api_request("GET", f"/v1/traces/{trace_id}", persona)
        if response.ok:
            st.session_state["trace"] = response.json()
        else:
            st.error(response.text)
    trace = st.session_state.get("trace")
    if trace:
        response_preview = trace.get("response", {})
        st.json(
            {
                "route": trace.get("route"),
                "status": trace.get("status"),
                "duration_ms": trace.get("duration_ms"),
                "request": trace.get("request"),
                "plan": response_preview.get("plan", []),
                "tool_calls": [call.get("tool") for call in response_preview.get("tool_calls", [])],
                "token_cost_estimate": response_preview.get("token_cost_estimate", {}),
                "citations": response_preview.get("citations", []),
                "safety": response_preview.get("safety", {}),
            }
        )
        for span in trace.get("spans", []):
            with st.expander(f"{span['name']} | {span['status']} | {span.get('duration_ms', 0):.1f} ms"):
                st.json(span)

with tabs[2]:
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
    if evals:
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

with tabs[3]:
    response = api_request("GET", "/v1/auth/me", persona)
    if response.ok:
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
    else:
        st.error(response.text)

with tabs[4]:
    col_a, col_b = st.columns([1, 3])
    if col_a.button("Reindex corpus"):
        with st.spinner("Generating synthetic docs, parsing, chunking, embedding, and indexing..."):
            response = api_request("POST", "/v1/ingestion/reindex", persona)
        if response.ok:
            st.success(response.json())
        else:
            st.info("Reindexing is restricted to planning_lead and admin personas.")
    response = api_request("GET", "/v1/corpus", persona)
    if response.ok:
        corpus = response.json()
        col_b.metric("Chunks", corpus.get("chunk_count", 0))
        st.subheader("Classification Distribution")
        st.json(corpus.get("classification_distribution", {}))
        st.subheader("Documents")
        st.dataframe(corpus.get("documents", []), width="stretch")
    else:
        st.info("Corpus metadata is unavailable for this persona.")
