"""Streamlit UI for the Defence Agent demo."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import time

import streamlit as st

from defence_agent.session import AgentTurnResult
from defence_agent.ui import backend_bridge
from defence_agent.ui.backend_bridge import DEFAULT_TIMEOUT_SECONDS
from defence_agent.ui.eval_results import (
    filter_eval_rows,
    load_eval_rows,
    transcript_run_options,
    transcript_run_label,
)
from defence_agent.ui.source_preview import resolve_source_preview
from defence_agent.ui.source_catalog import (
    CATALOG_FILTERS,
    filter_source_rows,
    source_catalog_for_persona,
    source_catalog_table_rows,
)
from defence_agent.ui.view_model import (
    DEFAULT_UI_PERSONA_ID,
    UI_PERSONAS,
    CitationView,
    DefenceAgentViewModel,
    EvidencePageView,
    RuntimeStepView,
    SourceView,
    ToolCallView,
    UiPersona,
    build_view_model,
    persona_for_ui_id,
)


@dataclass(frozen=True)
class DemoQuery:
    prompt: str
    summary: str
    target_answer_language: str = "auto"
    live_only: bool = False


DEMO_QUERIES = {
    "Planning brief comparison": DemoQuery(
        "Compare how Canada's defence policy and the DND/CAF AI Strategy describe AI-enabled modernization for a planning brief, and cite the strongest source pages.",
        "Compare Canadian defence modernization sources with cited support.",
        target_answer_language="en",
    ),
    "Scanned manual retrieval": DemoQuery(
        "What is technical intelligence, what foreign materiel does it cover, and which objectives matter most for strategic and tactical planners?",
        "Retrieve a digitized physical defence manual excerpt as cited evidence.",
        target_answer_language="en",
    ),
    "Access boundary": DemoQuery(
        "For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?",
        "Show that restricted evidence is withheld from the unclassified persona.",
        target_answer_language="en",
    ),
    "Evidence gap: Arctic basing 2031": DemoQuery(
        "What does the corpus say about the approved Arctic submarine basing schedule for 2031?",
        "Show that Command A abstains when authorized pages do not support the claim.",
        target_answer_language="en",
    ),
    "French NATO doctrine answer": DemoQuery(
        "Quelles taches fondamentales l'OTAN attribue-t-elle a l'Alliance dans son Concept strategique, et pourquoi sont-elles importantes pour un brief de planification canadien?",
        "French answer that highlights English and French NATO source pages.",
        target_answer_language="fr",
    ),
    "Concise cited NATO answer": DemoQuery(
        "Write a 150-200 word cited planning answer explaining NATO's core tasks and why they matter for a Canadian planning brief.",
        "Compact cited answer using NATO source pages.",
        target_answer_language="en",
    ),
}
EXAMPLE_PROMPTS = {label: query.prompt for label, query in DEMO_QUERIES.items()}
DEFAULT_DEMO_QUERY = "Planning brief comparison"
RUN_MODES = {
    "demo": "Guided run",
    "live": "Live run",
}
PLANNING_TRANSCRIPT_DIR = (
    Path(__file__).resolve().parent
    / "defence_agent"
    / "data"
    / "transcripts"
    / "live_readiness_flagship_retry_20260511_013333"
)
DEMO_TRANSCRIPT_DIR = (
    Path(__file__).resolve().parent
    / "defence_agent"
    / "data"
    / "transcripts"
    / "live_readiness_final_20260511_013937"
)
INSUFFICIENT_EVIDENCE_TRANSCRIPT_DIR = (
    Path(__file__).resolve().parent
    / "defence_agent"
    / "data"
    / "transcripts"
    / "insufficient_evidence_strategy_realignment_20260511_131708"
)
EVAL_FULL_TRANSCRIPT_DIR = (
    Path(__file__).resolve().parent
    / "defence_agent"
    / "data"
    / "transcripts"
    / "eval_full_registry_20260514_020915"
)
GUIDED_STEP_DELAY_SECONDS = 1.25
LIVE_STATUS_POLL_INTERVAL_SECONDS = 0.75
ANSWER_STREAM_CHUNK_WORDS = 4
ANSWER_STREAM_DELAY_SECONDS = 0.07


@dataclass(frozen=True)
class DemoReplay:
    transcript_dir: Path
    transcript_name: str
    demo_point: str


DEMO_REPLAYS = {
    ("persona_a", "Planning brief comparison"): DemoReplay(
        PLANNING_TRANSCRIPT_DIR,
        "flagship_planning_brief_modernization.json",
        "Two-step planning question across defence policy and AI strategy.",
    ),
    ("persona_a", "Access boundary"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "acl_unclassified_sensor_fusion_release_rule.json",
        "Unclassified user is blocked from restricted sensor-fusion guidance.",
    ),
    ("persona_a", "Scanned manual retrieval"): DemoReplay(
        EVAL_FULL_TRANSCRIPT_DIR,
        "scanned_manual_technical_intelligence.json",
        "Digitized physical manual excerpt with cited scanned-source pages.",
    ),
    ("persona_a", "Evidence gap: Arctic basing 2031"): DemoReplay(
        INSUFFICIENT_EVIDENCE_TRANSCRIPT_DIR,
        "insufficient_evidence_planning_topic.json",
        "Authorized pages are reviewed, then the unsupported scheduled claim is refused.",
    ),
    ("persona_a", "French NATO doctrine answer"): DemoReplay(
        EVAL_FULL_TRANSCRIPT_DIR,
        "french_nato_doctrine_answer.json",
        "French cited answer over bilingual NATO Strategic Concept pages.",
    ),
    ("persona_a", "Concise cited NATO answer"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "natural_multilingual_nato_core_tasks.json",
        "Compact cited answer using NATO source pages.",
    ),
    ("persona_b", "Planning brief comparison"): DemoReplay(
        PLANNING_TRANSCRIPT_DIR,
        "flagship_planning_brief_modernization.json",
        "Two-step planning question across defence policy and AI strategy.",
    ),
    ("persona_b", "Access boundary"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "acl_secret_sensor_fusion_release_rule.json",
        "Cleared user receives the restricted workflow rule with traceability.",
    ),
    ("persona_b", "Scanned manual retrieval"): DemoReplay(
        EVAL_FULL_TRANSCRIPT_DIR,
        "scanned_manual_technical_intelligence.json",
        "Digitized physical manual excerpt with cited scanned-source pages.",
    ),
    ("persona_b", "Evidence gap: Arctic basing 2031"): DemoReplay(
        INSUFFICIENT_EVIDENCE_TRANSCRIPT_DIR,
        "insufficient_evidence_planning_topic.json",
        "Authorized pages are reviewed, then the unsupported scheduled claim is refused.",
    ),
    ("persona_b", "French NATO doctrine answer"): DemoReplay(
        EVAL_FULL_TRANSCRIPT_DIR,
        "french_nato_doctrine_answer.json",
        "French cited answer over bilingual NATO Strategic Concept pages.",
    ),
    ("persona_b", "Concise cited NATO answer"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "natural_multilingual_nato_core_tasks.json",
        "Compact cited answer using NATO source pages.",
    ),
}


class DemoReplayUnavailable(RuntimeError):
    """Raised when an option preview cannot be used for the selected UI state."""


def main() -> None:
    st.set_page_config(page_title="Defence Agent", layout="centered", initial_sidebar_state="expanded")
    _apply_styles()
    _init_state()

    st.markdown("<h1 class='da-title'>Defence Agent</h1>", unsafe_allow_html=True)
    st.caption("Ask cited questions over approved manuals, procedures, and doctrine.")

    ask_tab, trace_tab, eval_tab, database_tab = st.tabs(["Ask", "Trace", "Eval", "Database"])
    with ask_tab:
        _render_ask_tab()
    with trace_tab:
        _render_trace_tab()
    with eval_tab:
        _render_eval_tab()
    with database_tab:
        _render_database_tab()


def _init_state() -> None:
    defaults = {
        "ui_persona_id": DEFAULT_UI_PERSONA_ID,
        "_active_ui_persona_id": DEFAULT_UI_PERSONA_ID,
        "query_text": EXAMPLE_PROMPTS[DEFAULT_DEMO_QUERY],
        "selected_example": DEFAULT_DEMO_QUERY,
        "last_result": None,
        "selected_citation_id": None,
        "selected_page_key": None,
        "last_run_error": "",
        "run_active": False,
        "scroll_to_evidence": False,
        "run_mode": "demo",
        "stream_answer": True,
        "stream_answer_once": False,
        "session_id_by_persona": {},
        "show_sanitized_json": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_ask_tab() -> None:
    _render_query_controls()

    if st.session_state.get("last_run_error"):
        st.error(st.session_state["last_run_error"])

    view_model = st.session_state.get("last_result")
    if view_model is None:
        return

    _render_runtime_flow_panel(view_model)
    _render_answer(view_model)


def _render_query_controls() -> UiPersona:
    with st.sidebar:
        st.markdown("## Defence Agent")
        selected = st.radio(
            "Persona",
            options=tuple(UI_PERSONAS.keys()),
            format_func=lambda key: UI_PERSONAS[key].label,
            key="ui_persona_id",
        )
    persona = persona_for_ui_id(selected)
    _reset_result_when_persona_changes(persona)

    with st.sidebar:
        st.caption(persona.description)
        st.markdown("**Allowed evidence**")
        st.write(persona.visible_access_label)

        if st.session_state.get("selected_example") not in EXAMPLE_PROMPTS:
            st.session_state["selected_example"] = DEFAULT_DEMO_QUERY
            st.session_state["query_text"] = EXAMPLE_PROMPTS[DEFAULT_DEMO_QUERY]
        selected_example = st.selectbox(
            "Demo query",
            options=tuple(EXAMPLE_PROMPTS.keys()),
            key="selected_example",
            on_change=_sync_selected_demo_query,
        )
        _render_demo_query_context(selected_example, persona)

        with st.expander("Advanced run controls", expanded=False):
            run_mode = st.radio(
                "Run mode",
                options=tuple(RUN_MODES.keys()),
                format_func=lambda key: RUN_MODES[key],
                key="run_mode",
            )
            effective_run_mode = _effective_run_mode(selected_example, run_mode)
            selected_query = DEMO_QUERIES.get(selected_example)
            if selected_query is not None and selected_query.live_only:
                st.caption("This selected query runs live because no guided replay is bundled.")
            elif effective_run_mode == "demo":
                st.caption("Uses the curated walkthrough output when available.")
            else:
                st.caption("Runs the backend with Cohere and the agent loop.")
            st.checkbox("Stream answer display", key="stream_answer")
            st.caption("Streams the answer after the run completes.")

        _render_tool_card()

    st.caption(
        f"Persona: {persona.label} · Allowed evidence: {persona.visible_access_label} · "
        f"Run mode: {RUN_MODES[effective_run_mode]}"
    )

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_area(
            "Question",
            key="query_text",
            height=116,
            placeholder="Ask a question over manuals, procedures, and doctrine...",
        )
        submitted = st.form_submit_button("Run query", type="primary")

    if submitted:
        _run_query(query=query, persona=persona, selected_example=selected_example, run_mode=effective_run_mode)
    return persona


def _render_tool_card() -> None:
    st.markdown("**Model-facing tool**")
    st.markdown(
        f"""
        <div class="da-tool-card">
            <div class="da-tool-header">
                <span>Only tool</span>
                <code>search_documents</code>
            </div>
            <div class="da-tool-purpose">Persona-scoped evidence retrieval over approved doctrine pages.</div>
            <div class="da-tool-flow">
                <span>Access filter</span>
                <span>Embed v4</span>
                <span>Vector search</span>
                <span>Rerank v4</span>
                <span>Citation IDs</span>
            </div>
            <code class="da-tool-signature">query, top_k, status_filter, language</code>
            <div class="da-tool-timeout">Live hard stop: {DEFAULT_TIMEOUT_SECONDS}s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _reset_result_when_persona_changes(persona: UiPersona) -> None:
    if st.session_state["_active_ui_persona_id"] == persona.ui_id:
        return
    st.session_state["_active_ui_persona_id"] = persona.ui_id
    st.session_state["last_result"] = None
    st.session_state["selected_citation_id"] = None
    st.session_state["selected_page_key"] = None
    st.session_state["last_run_error"] = ""
    st.session_state["run_active"] = False


def _sync_selected_demo_query() -> None:
    selected = st.session_state.get("selected_example", DEFAULT_DEMO_QUERY)
    prompt = EXAMPLE_PROMPTS.get(selected)
    if prompt is None:
        selected = DEFAULT_DEMO_QUERY
        prompt = EXAMPLE_PROMPTS[selected]
        st.session_state["selected_example"] = selected
    st.session_state["query_text"] = prompt
    st.session_state["last_result"] = None
    st.session_state["selected_citation_id"] = None
    st.session_state["selected_page_key"] = None
    st.session_state["last_run_error"] = ""


def _run_query(*, query: str, persona: UiPersona, selected_example: str, run_mode: str) -> None:
    cleaned = query.strip()
    if not cleaned:
        st.warning("Enter a query before running the agent.")
        return
    if st.session_state.get("run_active"):
        st.warning("A live Defence Agent run is already active.")
        return

    session_id = st.session_state["session_id_by_persona"].get(persona.ui_id)
    started = time.monotonic()
    st.session_state["run_active"] = True
    st.session_state["last_run_error"] = ""
    run_mode = _effective_run_mode(selected_example, run_mode)
    target_answer_language = _target_answer_language(cleaned, selected_example)
    try:
        if _can_use_guided_replay(run_mode, selected_example, cleaned):
            status_slot = st.empty()
            result = _load_demo_result(
                selected_example=selected_example,
                persona=persona,
                query=cleaned,
                target_answer_language=target_answer_language,
            )
            elapsed = time.monotonic() - started
            view_model = build_view_model(result, ui_persona_id=persona.ui_id, run_elapsed_seconds=elapsed)
            with status_slot.status("Running Defence Agent", expanded=True):
                _reveal_audit_sequence(view_model, delay_seconds=GUIDED_STEP_DELAY_SECONDS)
                _store_result(view_model, result, persona)
            status_slot.empty()
            return

        if not _cohere_api_key_available():
            raise RuntimeError("COHERE_API_KEY is missing for live mode.")

        timer_slot = st.empty()
        status_slot = st.empty()
        with status_slot.status("Running Defence Agent", expanded=True):
            st.write("Applying persona access policy")
            st.write("Running live Cohere/ADK backend with read-only search access")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _run_turn_sync,
                    cleaned,
                    persona=persona,
                    session_id=session_id,
                    target_answer_language=target_answer_language,
                    progress=lambda _message: None,
                )
                while not future.done():
                    elapsed = time.monotonic() - started
                    timer_slot.caption(
                        f"Live run · {_format_elapsed(elapsed)} elapsed · "
                        f"hard stop {DEFAULT_TIMEOUT_SECONDS}s"
                    )
                    time.sleep(LIVE_STATUS_POLL_INTERVAL_SECONDS)
                result = future.result()

            elapsed = time.monotonic() - started
            view_model = build_view_model(result, ui_persona_id=persona.ui_id, run_elapsed_seconds=elapsed)
            _reveal_audit_sequence(view_model, delay_seconds=GUIDED_STEP_DELAY_SECONDS)
            _store_result(view_model, result, persona)
        status_slot.empty()
        timer_slot.empty()
    except Exception as exc:
        st.session_state["last_run_error"] = _friendly_run_error(exc)
    finally:
        st.session_state["run_active"] = False


def _reveal_audit_sequence(view_model: DefenceAgentViewModel, *, delay_seconds: float) -> None:
    for step in view_model.runtime_steps:
        message = f"{step.index}. {step.title}: {step.detail}"
        _write_status_step(message, delay_seconds=delay_seconds)


def _render_runtime_flow_panel(view_model: DefenceAgentViewModel) -> None:
    with st.expander("Runtime flow", expanded=False):
        st.markdown(_runtime_flow_html(view_model), unsafe_allow_html=True)


def _runtime_flow_html(view_model: DefenceAgentViewModel) -> str:
    rows = "".join(_runtime_flow_row_html(step) for step in view_model.runtime_steps)
    return f'<div class="da-runtime-flow">{rows}</div>'


def _runtime_flow_row_html(step: RuntimeStepView) -> str:
    status = step.status
    code = f'<code class="da-runtime-code">{html.escape(step.code)}</code>' if step.code else ""
    detail = f'<div class="da-runtime-step-text">{html.escape(step.detail)}</div>' if step.detail else ""
    markers = "".join(
        f'<span class="da-runtime-chip"><span class="da-runtime-model-dot da-runtime-model-dot--{css_class}"></span>'
        f"{html.escape(label)}</span>"
        for label, css_class in step.markers
    )
    metrics = "".join(
        f'<span class="da-runtime-metric"><strong>{html.escape(label)}</strong> {html.escape(value)}</span>'
        for label, value in step.metrics
    )
    return (
        f'<div class="da-runtime-step da-runtime-step--{status}">'
        f'<span class="da-runtime-status-dot da-runtime-status-dot--{status}" aria-hidden="true"></span>'
        '<div class="da-runtime-step-body">'
        f'<div class="da-runtime-step-title">{step.index}. {html.escape(step.title)}</div>'
        f"{code}"
        f"{detail}"
        '<div class="da-runtime-step-meta">'
        f"{markers}"
        f"{metrics}"
        "</div>"
        "</div>"
        "</div>"
    )


def _audit_status_steps(view_model: DefenceAgentViewModel) -> list[str]:
    return [f"{step.index}. {step.title}: {step.detail}" for step in view_model.runtime_steps]


def _write_status_step(message: str, *, delay_seconds: float = 0.16) -> None:
    st.write(message)
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def _can_use_guided_replay(run_mode: str, selected_example: str, query: str) -> bool:
    if run_mode != "demo":
        return False
    demo_query = DEMO_QUERIES.get(selected_example)
    if demo_query is None or demo_query.live_only:
        return False
    return query.strip() == EXAMPLE_PROMPTS.get(selected_example, "").strip()


def _effective_run_mode(selected_example: str, requested_run_mode: str) -> str:
    demo_query = DEMO_QUERIES.get(selected_example)
    if demo_query is not None and demo_query.live_only:
        return "live"
    return requested_run_mode if requested_run_mode in RUN_MODES else "demo"


def _target_answer_language(query: str, selected_example: str) -> str:
    demo_query = DEMO_QUERIES.get(selected_example)
    if demo_query is not None and query.strip() == demo_query.prompt.strip():
        return demo_query.target_answer_language
    return _detect_query_language(query)


def _detect_query_language(query: str) -> str:
    text = f" {query.lower()} "
    french_markers = {
        " quelles ",
        " quels ",
        " quelle ",
        " quel ",
        " pourquoi ",
        " sont-elles ",
        " attribue-t-elle ",
        " taches ",
        " tâches ",
        " l'otan ",
        " dans son ",
        " pour un ",
        " canadien ",
    }
    return "fr" if sum(1 for marker in french_markers if marker in text) >= 2 else "en"


def _render_demo_query_context(selected_example: str, persona: UiPersona) -> None:
    query = DEMO_QUERIES.get(selected_example)
    if query is not None:
        st.caption(query.summary)
    replay = DEMO_REPLAYS.get((persona.ui_id, selected_example))
    if replay is None and query is not None and query.live_only:
        st.caption("Runs live against the current model and index.")


def _store_result(view_model: DefenceAgentViewModel, result: AgentTurnResult, persona: UiPersona) -> None:
    st.session_state["last_result"] = view_model
    st.session_state["session_id_by_persona"][persona.ui_id] = result.session_id
    selected_citation = view_model.citations[0] if view_model.citations else None
    selected_page = _evidence_page_for_citation(view_model, selected_citation) if selected_citation else None
    if selected_page is None and view_model.evidence_pages:
        selected_page = view_model.evidence_pages[0]
    st.session_state["selected_citation_id"] = selected_citation.citation_id if selected_citation else None
    st.session_state["selected_page_key"] = selected_page.page_key if selected_page else None
    st.session_state["stream_answer_once"] = bool(st.session_state.get("stream_answer"))
    st.session_state["last_run_error"] = ""


def _load_demo_result(
    *,
    selected_example: str,
    persona: UiPersona,
    query: str,
    target_answer_language: str,
) -> AgentTurnResult:
    replay = DEMO_REPLAYS.get((persona.ui_id, selected_example))
    if replay is None:
        raise DemoReplayUnavailable("No guided run exists for this persona and query.")
    path = replay.transcript_dir / replay.transcript_name
    if not path.exists():
        raise DemoReplayUnavailable("Guided run data is missing.")

    data = json.loads(path.read_text(encoding="utf-8"))
    audit = dict(data.get("final_answer_audit", {}) or {})
    if not str(audit.get("retrieval_status", "") or "").strip():
        audit["retrieval_status"] = "retrieval_complete"
    retrieval = dict(audit.get("retrieval", {}) or {})
    _validate_replay_acl(audit, persona=persona)
    retrieval["allowed_access"] = list(persona.allowed_access)
    audit["retrieval"] = retrieval
    audit["persona_id"] = persona.backend_persona_id
    audit["user_id"] = persona.backend_persona_id
    audit["query"] = query
    generation = audit.get("generation", {}) if isinstance(audit.get("generation"), dict) else {}
    generation = dict(generation)
    generation["target_answer_language"] = target_answer_language
    audit["generation"] = generation
    return AgentTurnResult(
        session_id=str(audit.get("session_id", "demo_replay")),
        user_id=persona.backend_persona_id,
        persona_id=persona.backend_persona_id,
        answer=str(data.get("answer", "")),
        raw_answer=str(data.get("raw_answer") or data.get("answer", "")),
        events_seen=0,
        tool_calls=list(audit.get("tool_calls", []) or []),
        tool_responses=list(audit.get("tool_responses", []) or []),
        citations=list(audit.get("citations", []) or []),
        citation_mode=str(data.get("mode", generation.get("citation_mode", ""))),
        citation_validation=dict(data.get("citation_validation", generation.get("citation_resolution", {})) or {}),
        grounded_model=str(generation.get("model", "")),
        documents_sent_to_model=int(generation.get("document_count", 0) or 0),
        retrieval_status=str(audit.get("retrieval_status", "")),
        answer_audit=audit,
    )


def _validate_replay_acl(audit: dict[str, object], *, persona: UiPersona) -> None:
    allowed = set(persona.allowed_access)
    retrieval = audit.get("retrieval", {}) if isinstance(audit.get("retrieval"), dict) else {}
    for group in _replay_acl_source_groups(audit, retrieval):
        if not isinstance(group, list):
            continue
        for source in group:
            if not isinstance(source, dict):
                continue
            access_level = str(source.get("access_level", "") or "")
            if access_level and access_level not in allowed:
                raise DemoReplayUnavailable(
                    "Selected demo path contains evidence outside the selected persona access policy."
                )


def _replay_acl_source_groups(
    audit: dict[str, object],
    retrieval: dict[str, object],
) -> tuple[object, ...]:
    citation_sources: list[dict[str, object]] = []
    for citation in audit.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        citation_sources.extend(
            source for source in citation.get("sources", []) or [] if isinstance(source, dict)
        )

    return (
        retrieval.get("authorized_sources", []),
        retrieval.get("sources_sent_to_answer", []),
        citation_sources,
    )


def _cohere_api_key_available() -> bool:
    if os.getenv("COHERE_API_KEY", "").strip():
        return True
    env_path = Path(".env")
    if not env_path.exists():
        return False
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("COHERE_API_KEY="):
                return bool(line.split("=", 1)[1].strip().strip('"').strip("'"))
    except OSError:
        return False
    return False


def _run_turn_sync(
    query: str,
    *,
    persona: UiPersona,
    session_id: str | None,
    target_answer_language: str,
    progress,
) -> AgentTurnResult:
    return backend_bridge.run_turn_in_subprocess(
        query=query,
        persona_id=persona.backend_persona_id,
        user_id=persona.backend_persona_id,
        session_id=session_id,
        target_answer_language=target_answer_language,
        progress=progress,
    )


def _render_answer(view_model: DefenceAgentViewModel) -> None:
    with st.container(border=True):
        _render_answer_text(view_model)
        _render_multilingual_retrieval(view_model)
        _render_citation_pages(view_model)
        _render_selected_evidence(view_model)
        _render_citation_explainer(view_model)
        _render_answer_trace_panel(view_model)


def _render_answer_text(view_model: DefenceAgentViewModel) -> None:
    if st.session_state.get("stream_answer_once"):
        _stream_answer_display(view_model)
        st.session_state["stream_answer_once"] = False
        return
    st.markdown(_answer_html_with_inline_citations(view_model), unsafe_allow_html=True)


def _render_multilingual_retrieval(view_model: DefenceAgentViewModel) -> None:
    payload = _multilingual_retrieval_payload(view_model)
    if payload is None:
        return
    st.markdown(_multilingual_retrieval_html(payload), unsafe_allow_html=True)


def _multilingual_retrieval_payload(view_model: DefenceAgentViewModel) -> dict[str, object] | None:
    sent_sources = [source for source in view_model.sources.values() if source.sent_to_model]
    if not sent_sources:
        sent_sources = [source for source in view_model.sources.values() if source.authorized_hit]
    sent_counts = _language_page_counts(sent_sources)
    if not {"en", "fr"}.issubset(sent_counts):
        return None

    cited_source_ids = {
        source_id
        for citation in view_model.citations
        for source_id in citation.source_ids
    }
    cited_sources = [source for source_id, source in view_model.sources.items() if source_id in cited_source_ids]
    cited_counts = _language_page_counts(cited_sources)
    return {
        "answer_language": _language_label(view_model.target_answer_language),
        "sent_counts": sent_counts,
        "cited_counts": cited_counts,
    }


def _language_page_counts(sources: list[SourceView]) -> dict[str, int]:
    page_keys_by_language: dict[str, set[str]] = {}
    for source in sources:
        language = _normalized_source_language(source.language)
        if language not in {"en", "fr"}:
            continue
        key = f"{source.doc_id}:{source.page}:{source.access_level}:{language}"
        page_keys_by_language.setdefault(language, set()).add(key)
    return {language: len(keys) for language, keys in page_keys_by_language.items() if keys}


def _multilingual_retrieval_html(payload: dict[str, object]) -> str:
    sent_counts = payload.get("sent_counts", {})
    cited_counts = payload.get("cited_counts", {})
    if not isinstance(sent_counts, dict) or not isinstance(cited_counts, dict):
        return ""
    sent = _language_count_phrase(sent_counts)
    cited = _language_count_phrase(cited_counts) or "none"
    return (
        '<div class="da-language-strip">'
        '<span class="da-language-title">Language mix</span>'
        f'<span>Answer language: {html.escape(str(payload.get("answer_language", "Query language")))}</span>'
        f"<span>Candidate pages: {html.escape(sent)}</span>"
        f"<span>Cited pages: {html.escape(cited)}</span>"
        "</div>"
    )


def _language_count_phrase(counts: dict[object, object]) -> str:
    pieces: list[str] = []
    for language in ("fr", "en"):
        count = int(counts.get(language, 0) or 0)
        if count:
            noun = "page" if count == 1 else "pages"
            pieces.append(f"{_source_language_label(language)} {count} {noun}")
    return " + ".join(pieces) if pieces else "none"


def _render_answer_trace_panel(view_model: DefenceAgentViewModel) -> None:
    with st.expander("Answer trace", expanded=not view_model.citations):
        st.caption(
            "ADK orchestrates the search_documents call. Cohere handles Embed v4, Rerank v4, "
            "Command A generation, and native citation spans."
        )

        st.markdown(
            _trace_meta_html(
                [
                    ("persona", view_model.persona_label),
                    ("allowed", view_model.visible_access_label),
                    ("language", _language_label(view_model.target_answer_language)),
                    ("decision", view_model.answerability.lower()),
                    ("docs", view_model.documents_sent_to_model),
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            _json_trace_block("Request", _request_trace_payload(view_model), open_block=False),
            unsafe_allow_html=True,
        )
        st.markdown(
            _json_trace_block("Agent tool calls", _tool_trace_payload(view_model), open_block=True),
            unsafe_allow_html=True,
        )
        st.markdown(
            _json_trace_block("Retrieval", _retrieval_trace_payload(view_model), open_block=False),
            unsafe_allow_html=True,
        )
        st.markdown(
            _json_trace_block("Evidence check", _answerability_trace_payload(view_model), open_block=True),
            unsafe_allow_html=True,
        )
        st.markdown(
            _json_trace_block("Cohere generation", _generation_trace_payload(view_model), open_block=False),
            unsafe_allow_html=True,
        )
        st.markdown(
            _json_trace_block("Citation map", _citation_map_payload(view_model), open_block=False),
            unsafe_allow_html=True,
        )
        st.markdown(
            _json_trace_block("Sanitized JSON", view_model.sanitized_answer_audit, open_block=False),
            unsafe_allow_html=True,
        )


def _json_trace_block(label: str, payload: object, *, open_block: bool) -> str:
    opened = " open" if open_block else ""
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    line_count = rendered.count("\n") + 1
    summary = f"{line_count} lines"
    return (
        f'<details class="da-json-trace"{opened}>'
        f"<summary><span>{html.escape(label)}</span><code>{html.escape(summary)}</code></summary>"
        f'<pre><code>{html.escape(rendered)}</code></pre>'
        "</details>"
    )


def _trace_meta_html(items: list[tuple[str, object]]) -> str:
    chips = "".join(
        f"<span><em>{html.escape(label)}</em>{html.escape(_render_trace_value(value))}</span>"
        for label, value in items
    )
    return f'<div class="da-trace-meta">{chips}</div>'


def _request_trace_payload(view_model: DefenceAgentViewModel) -> dict[str, object]:
    return {
        "persona": view_model.persona_label,
        "allowed_evidence": list(view_model.allowed_access),
        "target_answer_language": _language_label(view_model.target_answer_language),
        "query": view_model.query,
    }


def _tool_trace_payload(view_model: DefenceAgentViewModel) -> dict[str, object]:
    if not view_model.tool_calls:
        return {
            "orchestration": "google_adk",
            "tool_calls": [],
            "tool_results": [],
        }

    tool_calls: list[dict[str, object]] = []
    tool_results: list[dict[str, object]] = []
    for call in view_model.tool_calls:
        call_id = f"adk_{call.tool_name}_{call.call_index}"
        arguments = dict(call.arguments)
        if call.query and "query" not in arguments:
            arguments["query"] = call.query
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    "arguments": arguments,
                },
            }
        )
        tool_results.append(
            {
                "tool_call_id": call_id,
                "status": call.status,
                "filters_applied": call.filters,
                "summary": {
                    "authorized_source_count": call.authorized_source_count,
                    "sources_sent_to_answer_count": call.sources_sent_to_answer_count,
                    "excluded_source_count": call.excluded_source_count,
                    "per_page_vector_match": call.vector_score,
                    "vector_match_formula": "1 / (1 + distance)",
                    "per_page_rerank_relevance": call.rerank_score,
                    "embedding_backend": call.embedding_backend or "recorded_in_index_audit",
                    "rerank_backend": call.rerank_backend or "recorded_in_index_audit",
                },
            }
        )
    return {
        "orchestration": "google_adk",
        "cohere_shape_reference": "tool_plan/tool_calls/tool_results",
        "tool_plan": "Run search_documents, apply persona and retrieval filters, then send only authorized pages to generation.",
        "tool_calls": tool_calls,
        "tool_results": tool_results,
    }


def _retrieval_trace_payload(view_model: DefenceAgentViewModel) -> dict[str, object]:
    retrieval = view_model.answer_audit.get("retrieval", {}) if isinstance(view_model.answer_audit, dict) else {}
    searches = retrieval.get("searches", []) if isinstance(retrieval.get("searches", []), list) else []
    return {
        "allowed_access": list(retrieval.get("allowed_access", []) or []),
        "filters_applied": list(retrieval.get("filters_applied", []) or []),
        "search_count": retrieval.get("search_count", len(view_model.tool_calls)),
        "searches": searches,
        "authorized_source_count": len(retrieval.get("authorized_sources", []) or []),
        "sources_sent_to_answer_count": len(retrieval.get("sources_sent_to_answer", []) or []),
        "excluded_source_groups": view_model.excluded_source_summary,
    }


def _answerability_trace_payload(view_model: DefenceAgentViewModel) -> dict[str, object]:
    return {
        "decision": view_model.answerability.lower(),
        "reason": _answerability_reason_label(view_model.answerability_reason),
        "detail": _answerability_detail(view_model),
        "zero_doc_generation": view_model.documents_sent_to_model == 0,
    }


def _generation_trace_payload(view_model: DefenceAgentViewModel) -> dict[str, object]:
    generation = view_model.answer_audit.get("generation", {}) if isinstance(view_model.answer_audit, dict) else {}
    return {
        "model": generation.get("model", "") or "none",
        "document_count": generation.get("document_count", view_model.documents_sent_to_model),
        "cohere_document_ids": list(generation.get("cohere_document_ids", []) or []),
        "citation_mode": generation.get("citation_mode", view_model.citation_mode) or "none",
        "target_answer_language": generation.get("target_answer_language", view_model.target_answer_language),
    }


def _citation_map_payload(view_model: DefenceAgentViewModel) -> dict[str, object]:
    citations: list[dict[str, object]] = []
    for citation in view_model.citations:
        sources: list[dict[str, object]] = []
        for source_id in citation.source_ids:
            source = view_model.sources.get(source_id)
            if source is None:
                sources.append({"source_id": source_id})
                continue
            sources.append(
                {
                    "source_id": source.source_id,
                    "doc_id": source.doc_id,
                    "title": source.title,
                    "page": source.page,
                    "language": source.language,
                    "access_level": source.access_level,
                }
            )
        citations.append(
            {
                "citation": citation.marker,
                "start": citation.answer_start,
                "end": citation.answer_end,
                "text": citation.answer_text,
                "sources": sources,
            }
        )
    return {
        "citation_object_shape": "start/end/text/sources",
        "citations": citations,
    }


def _stream_answer_display(view_model: DefenceAgentViewModel) -> None:
    text = _inline_answer_base_text(view_model)
    if not text:
        st.markdown(_answer_html_with_inline_citations(view_model), unsafe_allow_html=True)
        return

    placeholder = st.empty()
    words = text.split()
    chunk_size = max(1, int(ANSWER_STREAM_CHUNK_WORDS))
    for end in range(chunk_size, len(words) + chunk_size, chunk_size):
        partial = " ".join(words[: min(end, len(words))])
        placeholder.markdown(_plain_answer_html(partial), unsafe_allow_html=True)
        if ANSWER_STREAM_DELAY_SECONDS > 0:
            time.sleep(ANSWER_STREAM_DELAY_SECONDS)
    placeholder.markdown(_answer_html_with_inline_citations(view_model), unsafe_allow_html=True)


def _plain_answer_html(text: str) -> str:
    body = html.escape(str(text or "").strip())
    body = body.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f'<div class="da-answer da-answer-streaming"><p>{body}</p></div>'


def _answer_html_with_inline_citations(view_model: DefenceAgentViewModel) -> str:
    text = _inline_answer_base_text(view_model)
    if not text:
        return "<p><em>No answer returned.</em></p>"

    markers_by_end: dict[int, list[str]] = {}
    for citation in view_model.citations:
        end = _inline_citation_end(view_model, text, citation)
        if end is None:
            continue
        href = _citation_href(citation.citation_id)
        tooltip = html.escape(_citation_tooltip(view_model, citation), quote=True)
        page_key = _single_page_key_for_citation(view_model, citation)
        page_key_attr = (
            f'data-citation-page-key="{html.escape(page_key, quote=True)}" '
            if page_key
            else ""
        )
        markers_by_end.setdefault(end, []).append(
            f'<a class="da-inline-cite" href="{href}" target="_self" '
            f'data-citation-id="{html.escape(citation.citation_id, quote=True)}" '
            f'{page_key_attr}data-citation-marker="{html.escape(citation.marker, quote=True)}" data-tooltip="{tooltip}" '
            f'aria-label="Show evidence for citation {citation.display_index}">'
            f"{html.escape(citation.marker)}</a>"
        )

    pieces: list[str] = []
    previous = 0
    for end in sorted(markers_by_end):
        pieces.append(html.escape(text[previous:end]))
        pieces.append("".join(markers_by_end[end]))
        previous = end
    pieces.append(html.escape(text[previous:]))

    body = "".join(pieces).strip()
    body = body.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f'<div class="da-answer"><p>{body}</p></div>'


def _citation_href(citation_id: str) -> str:
    return "#selected-evidence-anchor"


def _single_page_key_for_citation(
    view_model: DefenceAgentViewModel,
    citation: CitationView,
) -> str:
    page_keys = [
        page.page_key
        for page in view_model.evidence_pages
        if citation.citation_id in page.citation_ids
    ]
    unique_page_keys = list(dict.fromkeys(page_keys))
    return unique_page_keys[0] if len(unique_page_keys) == 1 else ""


def _inline_answer_base_text(view_model: DefenceAgentViewModel) -> str:
    raw = view_model.raw_answer or ""
    if raw and "[C" not in raw:
        return raw.strip()
    return view_model.display_answer.strip()


def _inline_citation_end(
    view_model: DefenceAgentViewModel,
    text: str,
    citation: CitationView,
) -> int | None:
    raw = view_model.raw_answer or ""
    if raw and "[C" not in raw and citation.answer_end is not None and 0 <= citation.answer_end <= len(text):
        return citation.answer_end
    return _find_citation_span_end(text, citation.answer_text) or len(text)


def _citation_tooltip(view_model: DefenceAgentViewModel, citation: CitationView) -> str:
    pages: list[str] = []
    seen: set[str] = set()
    for source_id in citation.source_ids:
        source = view_model.sources.get(source_id)
        if source is None:
            continue
        page_key = f"{source.doc_id}:{source.page}:{source.access_level}"
        if page_key in seen:
            continue
        seen.add(page_key)
        title = source.title or source.doc_id or "Source"
        page = f"p.{source.page}" if source.page else "page unknown"
        access = source.access_level or "unknown access"
        pages.append(f"{title} · {page} · {access}")

    visible_pages = pages[:2]
    if len(pages) > 2:
        visible_pages.append(f"+{len(pages) - 2} more")

    span = " ".join((citation.answer_text or "").split())
    if len(span) > 96:
        span = span[:93].rstrip() + "..."
    parts = visible_pages or ["Evidence source unresolved"]
    if span:
        parts.append(f"Span: {span}")
    return " | ".join(parts)


def _find_citation_span_end(text: str, span: str) -> int | None:
    clean_span = " ".join(str(span or "").split()).strip()
    if not clean_span:
        return None
    normalized_text = " ".join(text.split())
    index = normalized_text.lower().find(clean_span.lower())
    if index < 0:
        return None
    # The normalized string may collapse whitespace, so fall back to the first
    # exact-ish match in the original text when possible.
    original_index = text.lower().find(clean_span.lower())
    if original_index >= 0:
        return original_index + len(clean_span)
    return min(len(text), index + len(clean_span))


def _trace_cards_html(items: list[tuple[str, object]]) -> str:
    cards = "".join(_trace_card_html(label, value) for label, value in items)
    return f'<div class="da-trace-card-grid">{cards}</div>'


def _trace_card_html(label: str, value: object) -> str:
    rendered = _render_trace_value(value)
    return (
        '<div class="da-trace-card">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(rendered)}</strong>"
        "</div>"
    )


def _tool_call_card_html(call: ToolCallView) -> str:
    args = dict(call.arguments)
    filters = dict(call.filters)
    details = [
        ("Tool", call.tool_name),
        ("Query", call.query or args.get("query", "")),
        ("top_k", args.get("top_k", "")),
        ("status_filter", args.get("status_filter", filters.get("status", ""))),
        ("language", args.get("language", filters.get("language", ""))),
        ("authorized sources", call.authorized_source_count),
        ("sent to model", call.sources_sent_to_answer_count),
        ("excluded sources", call.excluded_source_count),
    ]
    return (
        '<div class="da-trace-call">'
        f"<h4>Tool call {call.call_index}</h4>"
        f"{_trace_cards_html(details)}"
        "</div>"
    )


def _retrieval_trace_html(view_model: DefenceAgentViewModel) -> str:
    retrieval = view_model.answer_audit.get("retrieval", {}) if isinstance(view_model.answer_audit, dict) else {}
    filters = retrieval.get("filters_applied", []) if isinstance(retrieval.get("filters_applied", []), list) else []
    first_filters = filters[0] if filters and isinstance(filters[0], dict) else {}
    calls = list(view_model.tool_calls)
    embedding = next((call.embedding_backend for call in calls if call.embedding_backend), "")
    rerank = next((call.rerank_backend for call in calls if call.rerank_backend), "")
    authorized_count = len(retrieval.get("authorized_sources", []) or [])
    sent_count = len(retrieval.get("sources_sent_to_answer", []) or [])
    excluded_count = sum(int(row.get("count", 0) or 0) for row in view_model.excluded_source_summary)
    return _trace_cards_html(
        [
            ("Searches", retrieval.get("search_count", len(calls))),
            ("Access filter", first_filters.get("access_level", view_model.allowed_access)),
            ("Status filter", first_filters.get("status", "")),
            ("Language filter", first_filters.get("language", "")),
            ("Embed backend", embedding or "recorded in index audit"),
            ("Rerank backend", rerank or "recorded in index audit"),
            ("Authorized pages", authorized_count),
            ("Pages sent to Command A", sent_count),
            ("Denied-source groups", excluded_count),
        ]
    )


def _answerability_trace_html(view_model: DefenceAgentViewModel) -> str:
    reason_detail = _answerability_detail(view_model)
    zero_doc = "yes" if view_model.documents_sent_to_model == 0 else "no"
    return _trace_cards_html(
        [
            ("Decision", view_model.answerability.lower()),
            ("Reason", _answerability_reason_label(view_model.answerability_reason)),
            ("Detail", reason_detail),
            ("Zero-doc generation", zero_doc),
        ]
    )


def _answerability_reason_label(reason: str) -> str:
    labels = {
        "authorized_evidence_sent": "authorized evidence sent",
        "authorized_sources_available": "authorized evidence available",
        "insufficient_authorized_evidence": "evidence not sufficient",
        "denied_source_matches_query": "access denied before generation",
        "no_authorized_sources": "no authorized sources",
    }
    normalized = str(reason or "").strip()
    return labels.get(normalized, normalized.replace("_", " ") or "not recorded")


def _generation_trace_html(view_model: DefenceAgentViewModel) -> str:
    generation = view_model.answer_audit.get("generation", {}) if isinstance(view_model.answer_audit, dict) else {}
    document_ids = generation.get("cohere_document_ids", []) if isinstance(generation.get("cohere_document_ids", []), list) else []
    return _trace_cards_html(
        [
            ("Model", generation.get("model", view_model.citation_mode) or "none"),
            ("Documents", view_model.documents_sent_to_model),
            ("Cohere document IDs", ", ".join(str(item) for item in document_ids[:6]) or "none"),
            ("Citation mode", view_model.citation_mode or "none"),
        ]
    )


def _citation_trace_card_html(view_model: DefenceAgentViewModel, citation: CitationView) -> str:
    source_labels: list[str] = []
    for source_id in citation.source_ids:
        source = view_model.sources.get(source_id)
        if source is None:
            source_labels.append(source_id)
            continue
        title = source.title or source.doc_id or source_id
        page = f"p.{source.page}" if source.page else "page unknown"
        language = source.language or "language unknown"
        source_labels.append(f"{title} {page} {language}")
    return (
        '<div class="da-trace-call">'
        f"<h4>{html.escape(citation.marker)} {html.escape(_compact_label(citation.answer_text, 96))}</h4>"
        f"{_trace_cards_html([('Linked source pages', '; '.join(source_labels) or 'none')])}"
        "</div>"
    )


def _render_citation_explainer(view_model: DefenceAgentViewModel) -> None:
    if not view_model.citations and not view_model.evidence_pages:
        return
    st.caption("Answer trace below keeps the full citation-span map and audit payload.")


def _answerability_detail(view_model: DefenceAgentViewModel) -> str:
    retrieval = view_model.answer_audit.get("retrieval", {}) if isinstance(view_model.answer_audit, dict) else {}
    records = [item for item in retrieval.get("answerability", []) or [] if isinstance(item, dict)]
    for record in records:
        if record.get("reason") == "denied_source_matches_query":
            denied_count = sum(int(row.get("count", 0) or 0) for row in view_model.excluded_source_summary)
            return (
                f"Restricted evidence matched the query but was withheld by persona policy "
                f"({denied_count} denied source group(s)); restricted text is not shown."
            )
        unsupported = record.get("unsupported_specificity", {})
        if isinstance(unsupported, dict) and (unsupported.get("evidence_gap") or unsupported.get("hard_refusal")):
            missing = list(unsupported.get("missing_year_terms", []) or []) + list(
                unsupported.get("missing_scheduled_fact_terms", []) or []
            )
            return "Missing support for: " + ", ".join(str(item) for item in missing)
    if view_model.answerability == "ANSWERED":
        return "Authorized evidence was sent to Command A for a grounded answer."
    return "No additional answerability detail was recorded."


def _language_label(language: str) -> str:
    labels = {"en": "English", "fr": "French", "auto": "Query language"}
    return labels.get(str(language or "auto").lower(), str(language or "auto"))


def _render_trace_value(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value) or "none"
    if value in ("", None, [], ()):
        return "none"
    return str(value)


def _render_citation_pages(view_model: DefenceAgentViewModel) -> None:
    if not view_model.evidence_pages:
        if not view_model.citations:
            st.caption("No citations were returned for this answer.")
        return

    st.markdown("**Citations**")
    st.caption("Source pages linked from inline citation markers. Each page lists the citation markers that point to it.")
    citations_by_id = {citation.citation_id: citation for citation in view_model.citations}
    for page in view_model.evidence_pages:
        span_count = len(page.citation_ids)
        span_label = "1 cited span" if span_count == 1 else f"{span_count} cited spans"
        button_label = f"{page.display_index}. {_evidence_page_button_label(page)}"
        if span_count:
            button_label += f" · {span_label}"
        st.markdown(
            _evidence_page_card_html(page, button_label, citations_by_id),
            unsafe_allow_html=True,
        )


def _evidence_page_card_html(
    page: EvidencePageView,
    label: str,
    citations_by_id: dict[str, CitationView],
) -> str:
    attributes = [
        'class="da-list-button da-evidence-link"',
        'href="#selected-evidence-anchor"',
        'target="_self"',
        f'data-page-key="{html.escape(page.page_key, quote=True)}"',
    ]
    page_link = f"<a {' '.join(attributes)}>{html.escape(label)}</a>"
    return (
        '<div class="da-evidence-card">'
        f"{page_link}"
        f"{_evidence_page_citation_map_html(page, citations_by_id)}"
        "</div>"
    )


def _evidence_page_citation_map_html(
    page: EvidencePageView,
    citations_by_id: dict[str, CitationView],
) -> str:
    chips: list[str] = []
    for citation_id in page.citation_ids:
        citation = citations_by_id.get(citation_id)
        marker = citation.marker if citation else citation_id
        label = f"{marker} {_compact_label(citation.answer_text, 96)}" if citation else marker
        attributes = [
            'class="da-citation-chip"',
            'href="#selected-evidence-anchor"',
            'target="_self"',
            f'data-citation-id="{html.escape(citation_id, quote=True)}"',
            f'data-citation-page-key="{html.escape(page.page_key, quote=True)}"',
            f'title="{html.escape(label, quote=True)}"',
            f'aria-label="Show evidence for {html.escape(marker, quote=True)} on this source page"',
        ]
        chips.append(f"<a {' '.join(attributes)}>{html.escape(marker)}</a>")
    if not chips:
        return ""
    return '<div class="da-citation-map"><span>Citation markers</span>' + "".join(chips) + "</div>"


def _evidence_page_button_label(page: EvidencePageView) -> str:
    title = page.source.title or page.source.doc_id or "Untitled source"
    source_page = f"p.{page.source.page}" if page.source.page else "page unknown"
    language = _source_language_label(page.source.language)
    return f"{title} · {source_page} · {language}"


def _source_language_label(language: str) -> str:
    normalized = _normalized_source_language(language)
    labels = {"en": "English", "fr": "French"}
    return labels.get(normalized, str(language or "language unknown"))


def _normalized_source_language(language: str) -> str:
    return str(language or "").strip().lower()


def _render_selected_evidence(view_model: DefenceAgentViewModel) -> None:
    selected_page = _selected_evidence_page(view_model)
    if selected_page is None:
        return

    selected_citation = _citation_for_evidence_page(view_model, selected_page)
    payloads = _evidence_page_payloads(view_model)
    selected_payload = payloads.get(selected_page.page_key)
    if selected_payload is None:
        st.caption("Selected evidence source metadata was not resolved in the answer audit.")
        return

    st.markdown("<div id='selected-evidence-anchor'></div>", unsafe_allow_html=True)
    st.markdown("**Selected evidence**")
    st.markdown(
        f'<div data-da-selected-evidence-panel>{selected_payload["html"]}</div>',
        unsafe_allow_html=True,
    )
    _install_citation_interaction_script(
        payloads,
        selected_page_key=selected_page.page_key,
        selected_citation_id=selected_citation.citation_id if selected_citation else "",
    )


def _evidence_page_payloads(view_model: DefenceAgentViewModel) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    previews: dict[str, object] = {}
    citations_by_id = {citation.citation_id: citation for citation in view_model.citations}
    for page in view_model.evidence_pages:
        source = page.source
        preview = previews.get(source.source_id)
        if preview is None:
            preview = resolve_source_preview(source, backend_persona_id=view_model.backend_persona_id)
            previews[source.source_id] = preview
        citation = _citation_for_evidence_page(view_model, page)
        html_by_citation_id = {
            citation_id: _selected_evidence_card_html(citations_by_id[citation_id], source, preview)
            for citation_id in page.citation_ids
            if citation_id in citations_by_id
        }
        payloads[page.page_key] = {
            "page_key": page.page_key,
            "citation_ids": list(page.citation_ids),
            "primary_citation_id": citation.citation_id if citation else "",
            "html": _selected_evidence_card_html(citation, source, preview),
            "html_by_citation_id": html_by_citation_id,
        }
    return payloads


def _selected_evidence_card_html(citation: CitationView | None, source: SourceView, preview) -> str:
    span = html.escape(citation.answer_text or "") if citation else ""
    cited_span = (
        f'{html.escape(citation.marker)} cited span: {span}'
        if citation
        else "Evidence page"
    )
    title = html.escape(source.title or source.doc_id or "Source")
    caption = html.escape(_source_caption(source, preview))
    actions = _source_actions_html(preview)
    preview_html = _source_preview_html(source, preview)
    metadata = html.escape(_compact_source_metadata(source, preview))
    page_text = _page_text_html(preview.page_text)
    return (
        '<div class="da-selected-evidence-card">'
        f'<p class="da-selected-span">{cited_span}</p>'
        f"<h3>{title}</h3>"
        f'<p class="da-source-caption">{caption}</p>'
        f"{actions}"
        f"{preview_html}"
        f'<p class="da-source-metadata">{metadata}</p>'
        f"{_source_note_html(source, preview)}"
        f"{page_text}"
        "</div>"
    )


def _source_actions_html(preview) -> str:
    pdf_label = (
        "Open normalized PDF page"
        if preview.source_format == "docx" and preview.normalized_format == "pdf"
        else "Open official PDF page"
    )
    links = [
        (pdf_label, preview.official_pdf_page_url),
        ("Open original Word document", preview.original_docx_url),
    ]
    visible = [(label, url) for label, url in links if url]
    if not visible:
        return ""
    items = [
        f'<a class="da-source-action" href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
        f"{html.escape(label)}</a>"
        for label, url in visible
    ]
    return '<div class="da-source-actions">' + "".join(items) + "</div>"


def _source_preview_html(source: SourceView, preview) -> str:
    if preview.page_image:
        encoded = base64.b64encode(preview.page_image).decode("ascii")
        page = html.escape(str(preview.page or source.page or "unknown"))
        return (
            '<figure class="da-page-preview">'
            f'<img src="data:image/png;base64,{encoded}" alt="Rendered source page {page}">'
            f"<figcaption>Rendered page {page} from the normalized source PDF.</figcaption>"
            "</figure>"
        )
    if not preview.authorized:
        return '<p class="da-source-muted">Preview blocked by persona access policy.</p>'
    if not preview.available:
        return '<p class="da-source-muted">Source preview is unavailable. The citation metadata above is still available.</p>'
    return ""


def _compact_source_metadata(source: SourceView, preview) -> str:
    values = {
        "title": source.title or source.doc_id,
        "doc ID": source.doc_id,
        "page": source.page,
        "version": source.version,
        "effective date": source.effective_date,
        "access": source.access_level,
        "language": source.language,
        "retrieved": preview.retrieved_date or source.retrieved_date,
    }
    return " · ".join(f"{key}: {value}" for key, value in values.items() if value not in ("", None))


def _source_note_html(source: SourceView, preview) -> str:
    notes: list[str] = []
    if preview.source_format == "docx" and preview.normalized_format == "pdf":
        notes.append("Word-origin source, normalized to PDF for page preview.")
    if preview.provenance_note or source.provenance_note:
        notes.append(preview.provenance_note or source.provenance_note)
    return "".join(f'<p class="da-source-muted">{html.escape(note)}</p>' for note in notes)


def _page_text_html(page_text: str) -> str:
    if not page_text:
        return ""
    return (
        '<details class="da-page-text">'
        "<summary>Page text</summary>"
        f"<pre>{html.escape(page_text)}</pre>"
        "</details>"
    )


def _install_citation_interaction_script(
    payloads: dict[str, dict[str, object]],
    *,
    selected_page_key: str,
    selected_citation_id: str,
) -> None:
    st.iframe(
        _citation_interaction_script(
            payloads,
            selected_page_key=selected_page_key,
            selected_citation_id=selected_citation_id,
        ),
        height=1,
    )


def _citation_interaction_script(
    payloads: dict[str, dict[str, object]],
    *,
    selected_page_key: str,
    selected_citation_id: str,
) -> str:
    payload_json = json.dumps(payloads).replace("</", "<\\/")
    initial_page_json = json.dumps(selected_page_key)
    initial_citation_json = json.dumps(selected_citation_id)
    return f"""
        <script>
        const parentWin = window.parent;
        const parentDoc = parentWin.document;
        parentWin.__defenceAgentEvidencePages = {payload_json};
        parentWin.__defenceAgentEvidenceRecordForCitation = (citationId) => {{
            const evidence = parentWin.__defenceAgentEvidencePages || {{}};
            for (const record of Object.values(evidence)) {{
                if ((record.citation_ids || []).includes(citationId)) {{
                    return record;
                }}
            }}
            return null;
        }};
        parentWin.__defenceAgentSelectPage = (pageKey, preferredCitationId, shouldScroll = true) => {{
            const evidence = parentWin.__defenceAgentEvidencePages || {{}};
            const record = evidence[pageKey];
            if (!record) return;
            const citationIds = record.citation_ids || [];
            const activeCitationId = citationIds.includes(preferredCitationId)
                ? preferredCitationId
                : (record.primary_citation_id || citationIds[0] || "");
            const htmlByCitationId = record.html_by_citation_id || {{}};
            const selectedHtml = activeCitationId && htmlByCitationId[activeCitationId]
                ? htmlByCitationId[activeCitationId]
                : record.html;
            const panel = parentDoc.querySelector("[data-da-selected-evidence-panel]");
            if (panel) {{
                panel.innerHTML = selectedHtml;
            }}
            parentDoc.querySelectorAll("[data-page-key]").forEach((node) => {{
                const active = node.dataset.pageKey === pageKey;
                node.classList.toggle("da-active-page", active);
                if (active) {{
                    node.setAttribute("aria-current", "true");
                }} else {{
                    node.removeAttribute("aria-current");
                }}
            }});
            parentDoc.querySelectorAll("[data-citation-id]").forEach((node) => {{
                const chipPageKey = node.dataset.citationPageKey || "";
                const active = Boolean(activeCitationId)
                    && node.dataset.citationId === activeCitationId
                    && (!chipPageKey || chipPageKey === pageKey);
                node.classList.toggle("da-active-citation", active);
                if (active) {{
                    node.setAttribute("aria-current", "true");
                }} else {{
                    node.removeAttribute("aria-current");
                }}
            }});
            const anchor = parentDoc.getElementById("selected-evidence-anchor");
            if (shouldScroll && anchor) {{
                parentWin.history.replaceState(null, "", "#selected-evidence-anchor");
                anchor.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
        }};
        parentWin.__defenceAgentSelectCitation = (citationId) => {{
            const activePage = parentDoc.querySelector("[data-page-key].da-active-page");
            if (activePage) {{
                const activeRecord = (parentWin.__defenceAgentEvidencePages || {{}})[activePage.dataset.pageKey];
                if (activeRecord && (activeRecord.citation_ids || []).includes(citationId)) {{
                    parentWin.__defenceAgentSelectPage(activeRecord.page_key, citationId);
                    return;
                }}
            }}
            const record = parentWin.__defenceAgentEvidenceRecordForCitation(citationId);
            if (!record) return;
            parentWin.__defenceAgentSelectPage(record.page_key, citationId);
        }};
        if (!parentWin.__defenceAgentCitationClickInstalled) {{
            parentDoc.addEventListener("click", (event) => {{
                const target = event.target;
                if (!(target instanceof parentWin.Element)) return;
                const citationTarget = target.closest("[data-citation-id]");
                if (citationTarget) {{
                    const citationId = citationTarget.dataset.citationId;
                    const evidence = parentWin.__defenceAgentEvidencePages || {{}};
                    const pageKey = citationTarget.dataset.citationPageKey;
                    if (!citationId || !parentWin.__defenceAgentEvidenceRecordForCitation(citationId)) return;
                    event.preventDefault();
                    if (pageKey && evidence[pageKey]) {{
                        parentWin.__defenceAgentSelectPage(pageKey, citationId);
                    }} else {{
                        parentWin.__defenceAgentSelectCitation(citationId);
                    }}
                    return;
                }}
                const pageTarget = target.closest("[data-page-key]");
                if (pageTarget) {{
                    const pageKey = pageTarget.dataset.pageKey;
                    const evidence = parentWin.__defenceAgentEvidencePages || {{}};
                    if (!pageKey || !evidence[pageKey]) return;
                    event.preventDefault();
                    parentWin.__defenceAgentSelectPage(pageKey, evidence[pageKey].primary_citation_id || "");
                }}
            }}, true);
            parentWin.__defenceAgentCitationClickInstalled = true;
        }}
        parentWin.__defenceAgentSelectPage({initial_page_json}, {initial_citation_json}, false);
        </script>
        """


def _selected_citation(view_model: DefenceAgentViewModel) -> CitationView | None:
    if not view_model.citations:
        return None
    selected_id = st.session_state.get("selected_citation_id")
    for citation in view_model.citations:
        if citation.citation_id == selected_id:
            return citation
    return view_model.citations[0]


def _selected_evidence_page(view_model: DefenceAgentViewModel) -> EvidencePageView | None:
    if not view_model.evidence_pages:
        return None
    selected_page_key = st.session_state.get("selected_page_key")
    for page in view_model.evidence_pages:
        if page.page_key == selected_page_key:
            return page
    selected_citation = _selected_citation(view_model)
    if selected_citation:
        page = _evidence_page_for_citation(view_model, selected_citation)
        if page:
            return page
    return view_model.evidence_pages[0]


def _citation_for_evidence_page(
    view_model: DefenceAgentViewModel,
    page: EvidencePageView,
) -> CitationView | None:
    selected = _selected_citation(view_model)
    if selected and selected.citation_id in page.citation_ids:
        return selected
    citations = {citation.citation_id: citation for citation in view_model.citations}
    for citation_id in page.citation_ids:
        citation = citations.get(citation_id)
        if citation:
            return citation
    return selected


def _evidence_page_for_citation(
    view_model: DefenceAgentViewModel,
    citation: CitationView,
) -> EvidencePageView | None:
    for page in view_model.evidence_pages:
        if citation.citation_id in page.citation_ids:
            return page
    return view_model.evidence_pages[0] if view_model.evidence_pages else None


def _source_caption(source: SourceView, preview) -> str:
    pieces = [
        source.doc_id,
        f"p.{source.page}" if source.page else "",
        source.access_level,
        _source_language_label(source.language),
    ]
    organization = preview.source_organization or source.source_organization
    if organization:
        pieces.append(organization)
    return " · ".join(piece for piece in pieces if piece)


def _friendly_run_error(exc: Exception) -> str:
    if isinstance(exc, DemoReplayUnavailable):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return f"Live run timed out after {DEFAULT_TIMEOUT_SECONDS}s. No transcript fallback was used."
    if "COHERE_API_KEY is missing" in str(exc):
        return "COHERE_API_KEY is missing for live mode."
    if isinstance(exc, backend_bridge.BackendRunError):
        return _friendly_backend_error(str(exc))
    return "Defence Agent run failed. Check the Streamlit server logs for details."


def _friendly_backend_error(message: str) -> str:
    text = str(message or "").lower()
    no_fallback = "No transcript fallback was used."
    if _contains_any_error_text(text, ("rate limit", "too many requests", "429")):
        return f"Cohere rate limit hit during live mode. Wait briefly, then rerun the same case. {no_fallback}"
    if _contains_any_error_text(
        text,
        ("authentication", "unauthorized", "invalid api key", "invalid token", "401", "403", "forbidden"),
    ):
        return f"Cohere authentication failed in live mode. Check COHERE_API_KEY. {no_fallback}"
    if _contains_any_error_text(
        text,
        (
            "remoteprotocolerror",
            "incomplete chunked read",
            "peer closed connection",
            "connection reset",
            "readtimeout",
            "timed out",
            "timeout",
            "502",
            "503",
            "504",
        ),
    ):
        return (
            "Cohere transport error during the live run. Retry the same case before treating it as a "
            f"retrieval or citation regression. {no_fallback}"
        )
    if _contains_any_error_text(
        text,
        ("index_build.lock", "no such collection", "chroma", "manifest", "sqlite"),
    ):
        return (
            "Local retrieval index failed during live mode. Check index artifacts and locks before rerunning. "
            f"{no_fallback}"
        )
    return f"Backend worker failed during live mode. Check the Streamlit server logs for details. {no_fallback}"


def _contains_any_error_text(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "00:00"
    whole = max(0, int(seconds))
    minutes, remainder = divmod(whole, 60)
    return f"{minutes:02d}:{remainder:02d}"


def _compact_label(value: str, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "..."


def _scroll_to_selected_evidence() -> None:
    st.iframe(_scroll_to_selected_evidence_script(), height=1)


def _scroll_to_selected_evidence_script() -> str:
    return """
        <script>
        const parentDoc = window.parent.document;
        let attempts = 0;
        const scrollToEvidence = () => {
            const target = parentDoc.getElementById("selected-evidence-anchor");
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
                parentDoc.__defenceAgentPendingCitationScroll = false;
                return;
            }
            attempts += 1;
            if (attempts < 30) {
                window.setTimeout(scrollToEvidence, 80);
            }
        };
        window.requestAnimationFrame(() => window.setTimeout(scrollToEvidence, 80));
        </script>
        """


def _render_trace_tab() -> None:
    view_model = st.session_state.get("last_result")
    st.subheader("Execution Trace")
    if view_model is None:
        st.info("Run a query in Ask to populate the trace.")
        return

    _render_trace_story(view_model)

    st.markdown("**Detailed audit tables**")
    st.markdown("**Request and access policy**")
    st.dataframe(_request_rows(view_model), width="stretch", hide_index=True)

    st.markdown("**Tool calls**")
    st.dataframe([call.as_row() for call in view_model.tool_calls], width="stretch", hide_index=True)

    st.markdown("**Authorized evidence**")
    st.dataframe(view_model.authorized_source_rows, width="stretch", hide_index=True)

    st.markdown("**Excluded evidence summary**")
    if view_model.excluded_source_summary:
        st.dataframe(view_model.excluded_source_summary, width="stretch", hide_index=True)
    else:
        st.caption("No excluded restricted-source metadata was reported for this turn.")

    st.markdown("**Citation resolution**")
    st.dataframe(_citation_resolution_rows(view_model), width="stretch", hide_index=True)

    st.checkbox("Show sanitized technical JSON", key="show_sanitized_json")
    if st.session_state.get("show_sanitized_json"):
        with st.expander("Sanitized answer audit", expanded=False):
            st.json(view_model.sanitized_answer_audit)


def _render_trace_story(view_model: DefenceAgentViewModel) -> None:
    st.markdown("**Execution timeline**")
    st.markdown(_trace_timeline_html(view_model), unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].metric("Retrieval status", _compact_label(view_model.retrieval_status or "unknown", 22))
    cols[1].metric("Answerability", view_model.answerability.lower())
    cols[2].metric("Searches", str(len(view_model.tool_calls)))
    cols[3].metric("Citations", str(len(view_model.citations)))
    cols = st.columns(3)
    cols[0].metric("Persona", view_model.persona_label.replace("Persona ", "P"))
    cols[1].metric("Allowed evidence", view_model.visible_access_label)
    cols[2].metric("Evidence pages", str(len(view_model.evidence_pages)))

    st.markdown("**Full query**")
    st.text_area(
        "Full query",
        value=view_model.query,
        height=104,
        disabled=True,
        key=f"trace_query_{view_model.session_id}",
        label_visibility="collapsed",
    )
    if view_model.thinking_blocks:
        with st.expander("Cohere thinking blocks", expanded=False):
            st.caption("Returned only by reasoning-capable Cohere models; separate from the action audit above.")
            for block in view_model.thinking_blocks:
                st.text_area(
                    f"Thinking block {block.get('index', '')}",
                    value=str(block.get("thinking", "")),
                    height=180,
                    disabled=True,
                    key=f"thinking_{view_model.session_id}_{block.get('index', len(str(block)))}",
                )


def _trace_timeline_html(view_model: DefenceAgentViewModel) -> str:
    steps: list[tuple[str, str]] = [
        (
            "1. Access scope set",
            f"{view_model.persona_label} can use {view_model.visible_access_label} evidence.",
        )
    ]

    next_index = 2
    if view_model.tool_calls:
        for call in view_model.tool_calls:
            query = call.query or "document search"
            filters = _trace_filter_summary(call.filters)
            detail = query
            if filters:
                detail += f" | {filters}"
            steps.append((f"{next_index}. Search {call.call_index}", detail))
            next_index += 1
    else:
        steps.append((f"{next_index}. Search", "No document search was required for this turn."))
        next_index += 1

    excluded_count = len(view_model.excluded_source_summary)
    evidence_count = len(view_model.evidence_pages)
    filtered_detail = f"{evidence_count} source pages moved forward"
    if excluded_count:
        filtered_detail += f"; {excluded_count} restricted source groups excluded"
    steps.append((f"{next_index}. Evidence filtered and ranked", filtered_detail))
    next_index += 1

    steps.append(
        (
            f"{next_index}. Answer generated",
            f"{_generation_model_label(view_model)} used {view_model.documents_sent_to_model} evidence documents for the grounded response.",
        )
    )
    next_index += 1

    steps.append(
        (
            f"{next_index}. Citations resolved",
            f"{len(view_model.citations)} cited answer spans mapped back to {evidence_count} source pages.",
        )
    )

    items = "".join(_trace_step_html(title, detail) for title, detail in steps)
    return f'<div class="da-trace-timeline">{items}</div>'


def _generation_model_label(view_model: DefenceAgentViewModel) -> str:
    generation = view_model.answer_audit.get("generation", {}) if isinstance(view_model.answer_audit, dict) else {}
    model = str(generation.get("model", "") or "")
    if not model:
        return "Cohere"
    if "reasoning" in model:
        return "Command A Reasoning"
    if "command-a" in model:
        return "Command A"
    return model


def _trace_step_html(title: str, detail: str) -> str:
    return (
        "<div>"
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(detail)}</span>"
        "</div>"
    )


def _trace_filter_summary(filters: dict[str, object]) -> str:
    if not filters:
        return ""
    pieces: list[str] = []
    for key in ("access_level", "language", "status"):
        value = filters.get(key)
        if value in ("", None, [], ()):
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        pieces.append(f"{key}: {rendered}")
    return "; ".join(pieces)


def _request_rows(view_model: DefenceAgentViewModel) -> list[dict[str, object]]:
    return [
        {"field": "session_id", "value": view_model.session_id},
        {"field": "persona", "value": view_model.persona_label},
        {"field": "backend_persona_id", "value": view_model.backend_persona_id},
        {"field": "allowed_access", "value": ", ".join(view_model.allowed_access)},
        {"field": "target_answer_language", "value": view_model.target_answer_language},
        {"field": "answerability", "value": view_model.answerability.lower()},
        {"field": "retrieval_status", "value": view_model.retrieval_status},
        {"field": "citation_mode", "value": view_model.citation_mode},
    ]


def _render_eval_tab() -> None:
    st.subheader("Demo Readiness")
    options = transcript_run_options()
    if not options:
        st.info("No saved transcript runs found under defence_agent/data/transcripts.")
        return

    selected_run = st.selectbox(
        "Transcript run",
        options=options,
        index=0,
        format_func=transcript_run_label,
    )
    rows = load_eval_rows(selected_run)
    eval_filter = st.radio(
        "Filter",
        ("all", "failed only", "ACL", "citation", "refusal", "bilingual"),
        horizontal=True,
    )
    filtered = filter_eval_rows(rows, eval_filter)

    _render_eval_metrics(rows)
    st.dataframe(filtered, width="stretch", hide_index=True)


def _render_eval_metrics(rows: list[dict[str, object]]) -> None:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    refusal_cases = sum(1 for row in rows if row.get("final_answerability") == "refused")
    zero_doc_refusals = sum(1 for row in rows if row.get("zero_doc_refusal"))
    citation_cases = sum(1 for row in rows if int(row.get("citation_count") or 0) > 0)
    cols = st.columns(5)
    cols[0].metric("Cases", str(total))
    cols[1].metric("Pass rate", f"{round((passed / total) * 100)}%" if total else "n/a")
    cols[2].metric("Refusals", str(refusal_cases))
    cols[3].metric("Zero-doc refusals", str(zero_doc_refusals))
    cols[4].metric("Cited cases", str(citation_cases))


def _render_database_tab() -> None:
    st.subheader("Database")
    st.caption("Approved source catalog filtered by persona access.")

    persona_id = str(st.session_state.get("ui_persona_id", DEFAULT_UI_PERSONA_ID))
    catalog = source_catalog_for_persona(persona_id)

    cols = st.columns(4)
    cols[0].metric("Visible docs", str(len(catalog.rows)))
    cols[1].metric("Visible pages", str(catalog.visible_pages))
    cols[2].metric("Datasets", str(catalog.dataset_count))
    cols[3].metric("Withheld docs", str(catalog.withheld_count))

    filter_cols = st.columns([0.48, 0.52])
    with filter_cols[0]:
        dataset_filter = st.radio(
            "Catalog filter",
            CATALOG_FILTERS,
            horizontal=True,
            key="source_catalog_filter",
        )
    with filter_cols[1]:
        search_query = st.text_input(
            "Search catalog",
            key="source_catalog_search",
            placeholder="Search title, document ID, owner, or dataset",
        )

    rows = filter_source_rows(catalog.rows, dataset_filter=dataset_filter, search_query=search_query)
    if not rows:
        st.info("No visible source rows match the current catalog filter.")
        return

    st.dataframe(
        source_catalog_table_rows(rows),
        width="stretch",
        hide_index=True,
    )


def _citation_resolution_rows(view_model: DefenceAgentViewModel) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    allowed_access = set(view_model.allowed_access)
    for citation in view_model.citations:
        for source_id in citation.source_ids:
            source = view_model.sources.get(source_id)
            rows.append(
                {
                    "citation": citation.marker,
                    "answer_text": citation.answer_text,
                    "source": source.title if source else source_id,
                    "doc_id": source.doc_id if source else "",
                    "resolved": source is not None,
                    "authorized": bool(source and source.access_level in allowed_access),
                    "support_check": "manual review required",
                }
            )
    return rows


def _apply_styles() -> None:
    st.markdown(_app_css(), unsafe_allow_html=True)


def _app_css() -> str:
    return """
        <style>
        :root {
            --da-green: #062c22;
            --da-near-black: #061324;
            --da-accent: #00a04d;
            --da-off-white: #f0eee9;
            --da-panel: #f8f6ef;
            --da-pale-green: #f1fdea;
            --da-aqua: #b8f9f3;
            --da-selected-border: #287e78;
            --da-coral: #da532c;
            --da-gray: #a4a4a4;
            --da-border: #d6d1c8;
            --da-border-strong: #a9b8ad;
            --da-radius: 6px;
            --da-radius-sm: 4px;
            --da-shadow: 0 1px 0 rgba(6, 19, 36, 0.06);
            --da-font: CohereText, CohereVariable, "Unica77 Cohere Web", Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            --da-mono: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }
        html {
            scroll-behavior: smooth;
        }
        .block-container {
            max-width: 920px;
            padding-top: 3rem;
        }
        .da-title {
            font-size: 2.65rem;
            line-height: 1.05;
            margin-bottom: 0.35rem;
            letter-spacing: 0;
        }
        section[data-testid="stSidebar"] h2 {
            font-size: 1.35rem;
            letter-spacing: 0;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1rem;
        }
        div[data-testid="stButton"] button {
            justify-content: flex-start;
            font-size: 0.92rem;
        }
        div[data-testid="stAlert"] {
            font-size: 0.94rem;
        }
        .da-tool-card {
            margin: 0.35rem 0 1rem;
            padding: 0.72rem;
            border: 1px solid rgba(248, 246, 239, 0.24);
            border-radius: var(--da-radius);
            background: rgba(248, 246, 239, 0.09);
            color: var(--da-off-white);
        }
        .da-tool-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.45rem;
        }
        .da-tool-header span {
            color: rgba(240, 238, 233, 0.72);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .da-tool-header code,
        .da-tool-signature {
            font-family: var(--da-mono);
            font-size: 0.74rem;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        .da-tool-purpose {
            margin-bottom: 0.52rem;
            color: rgba(240, 238, 233, 0.9);
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .da-tool-flow {
            display: flex;
            flex-wrap: wrap;
            gap: 0.28rem;
            margin-bottom: 0.56rem;
        }
        .da-tool-flow span {
            padding: 0.14rem 0.34rem;
            border: 1px solid rgba(184, 249, 243, 0.36);
            border-radius: var(--da-radius-sm);
            background: rgba(184, 249, 243, 0.12);
            color: var(--da-aqua);
            font-size: 0.72rem;
            font-weight: 650;
            line-height: 1.25;
        }
        .da-tool-signature {
            display: block;
            margin-bottom: 0.38rem;
            color: rgba(240, 238, 233, 0.86);
        }
        .da-tool-timeout {
            color: rgba(240, 238, 233, 0.64);
            font-size: 0.72rem;
        }
        .da-list-button {
            display: block;
            width: 100%;
            margin: 0.34rem 0;
            padding: 0.55rem 0.64rem;
            border: 1px solid var(--da-border);
            border-radius: var(--da-radius-sm);
            background: var(--da-panel);
            color: var(--da-near-black) !important;
            font-size: 0.86rem;
            font-weight: 620;
            line-height: 1.35;
            text-align: left;
            text-decoration: none !important;
            box-shadow: var(--da-shadow);
        }
        .da-list-button:hover,
        .da-list-button:focus {
            border-color: var(--da-accent);
            background: var(--da-pale-green);
            color: var(--da-green) !important;
        }
        .da-list-button.da-active-citation,
        .da-list-button.da-active-page {
            border-color: var(--da-selected-border);
            background: var(--da-aqua);
            color: var(--da-near-black) !important;
        }
        .da-evidence-card {
            margin: 0.42rem 0 0.72rem;
        }
        .da-evidence-card .da-list-button {
            margin-bottom: 0.24rem;
        }
        .da-citation-map {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.32rem;
            margin: 0 0 0.32rem;
            color: #5d6762;
            font-size: 0.78rem;
            line-height: 1.3;
        }
        .da-citation-map span {
            font-weight: 650;
        }
        .da-citation-chip {
            display: inline-flex;
            align-items: center;
            min-height: 1.35rem;
            padding: 0.03rem 0.34rem;
            border: 1px solid var(--da-border);
            border-radius: var(--da-radius-sm);
            background: var(--da-off-white);
            color: var(--da-near-black) !important;
            font-size: 0.82rem;
            font-weight: 700;
            text-decoration: none !important;
        }
        .da-list-button:focus-visible,
        .da-inline-cite:focus-visible,
        .da-citation-chip:focus-visible {
            outline: 2px solid var(--da-selected-border) !important;
            outline-offset: 2px;
        }
        .da-answer {
            font-size: 1rem;
            line-height: 1.62;
        }
        .da-answer p {
            margin: 0 0 0.85rem 0;
        }
        .da-language-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.25rem 0 0.9rem;
        }
        .da-language-strip span {
            display: inline-flex;
            align-items: center;
            min-height: 1.75rem;
            padding: 0.22rem 0.52rem;
            border: 1px solid var(--da-border);
            border-radius: var(--da-radius-sm);
            background: var(--da-panel);
            color: var(--da-near-black);
            box-shadow: var(--da-shadow);
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1.2;
        }
        .da-language-strip .da-language-title {
            border-color: var(--da-border-strong);
            background: var(--da-off-white);
            color: var(--da-near-black);
        }
        .da-inline-cite {
            position: relative;
            display: inline-flex;
            align-items: center;
            margin-left: 0.18rem;
            padding: 0.05rem 0.34rem;
            border: 1px solid var(--da-border);
            border-radius: var(--da-radius-sm);
            background: var(--da-off-white);
            color: var(--da-near-black) !important;
            font-size: 0.72em;
            font-weight: 750;
            line-height: 1.25;
            text-decoration: none !important;
            transform: translateY(-0.04rem);
        }
        .da-inline-cite:hover,
        .da-inline-cite:focus,
        .da-citation-chip:hover,
        .da-citation-chip:focus {
            border-color: var(--da-accent);
            background: var(--da-pale-green);
            color: var(--da-green) !important;
        }
        .da-inline-cite.da-active-citation,
        .da-citation-chip.da-active-citation {
            border-color: var(--da-selected-border);
            background: var(--da-aqua);
            color: var(--da-near-black) !important;
        }
        .da-inline-cite[data-tooltip]:hover::after,
        .da-inline-cite[data-tooltip]:focus::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 50%;
            bottom: calc(100% + 0.52rem);
            z-index: 1000;
            width: min(21rem, 72vw);
            transform: translateX(-50%);
            padding: 0.62rem 0.72rem;
            border: 1px solid #d6d6d2;
            border-radius: 0.55rem;
            background: #fffefa;
            box-shadow: 0 0.5rem 1.4rem rgba(24, 24, 27, 0.12);
            color: #2e2f3a;
            font-size: 0.82rem;
            font-weight: 500;
            line-height: 1.35;
            text-align: left;
            white-space: normal;
        }
        .da-inline-cite[data-tooltip]:hover::before,
        .da-inline-cite[data-tooltip]:focus::before {
            content: "";
            position: absolute;
            left: 50%;
            bottom: calc(100% + 0.22rem);
            z-index: 1001;
            transform: translateX(-50%);
            border: 0.34rem solid transparent;
            border-top-color: #fffefa;
        }
        .da-selected-evidence-card {
            border: 1px solid #deded9;
            border-radius: 0.6rem;
            padding: 1rem;
            background: #fffefa;
        }
        .da-selected-evidence-card h3 {
            margin: 0.35rem 0 0.35rem;
            font-size: 1.05rem;
            line-height: 1.3;
        }
        .da-selected-span,
        .da-source-caption,
        .da-source-metadata,
        .da-source-muted {
            margin: 0.2rem 0 0.7rem;
            color: #747782;
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .da-source-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
            gap: 0.55rem;
            margin: 0.8rem 0 1rem;
        }
        .da-source-action {
            display: block;
            padding: 0.62rem 0.76rem;
            border: 1px solid #d8d8d4;
            border-radius: 0.5rem;
            color: #2e2f3a !important;
            font-size: 0.9rem;
            font-weight: 550;
            text-align: center;
            text-decoration: none !important;
        }
        .da-source-action:hover,
        .da-source-action:focus {
            border-color: #82b6a1;
            background: #f4fbf8;
            color: #185b45 !important;
        }
        .da-page-preview {
            margin: 0.8rem 0 0.9rem;
        }
        .da-page-preview img {
            display: block;
            width: 100%;
            max-width: 760px;
            margin: 0 auto;
            border: 1px solid #e3e3de;
            border-radius: 0.55rem;
            background: #ffffff;
        }
        .da-page-preview figcaption {
            margin-top: 0.45rem;
            color: #747782;
            font-size: 0.84rem;
            text-align: center;
        }
        .da-page-text {
            margin-top: 0.75rem;
            border: 1px solid #deded9;
            border-radius: 0.5rem;
            padding: 0.55rem 0.7rem;
        }
        .da-page-text summary {
            cursor: pointer;
            font-weight: 600;
        }
        .da-page-text pre {
            margin: 0.7rem 0 0;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            font-family: inherit;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .da-trace-timeline {
            display: grid;
            gap: 0.52rem;
            margin: 0.4rem 0 1rem 0;
        }
        .da-trace-timeline > div {
            border: 1px solid #deded9;
            border-radius: 0.55rem;
            padding: 0.65rem 0.75rem;
            background: #fffefa;
        }
        .da-trace-timeline strong,
        .da-trace-timeline span {
            display: block;
        }
	        .da-trace-timeline span {
	            margin-top: 0.12rem;
	            color: #6f727c;
	            font-size: 0.88rem;
	            line-height: 1.35;
	        }
	        .da-trace-card-grid {
	            display: grid;
	            grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr));
	            gap: 0.5rem;
	            margin: 0.45rem 0 0.85rem;
	        }
	        .da-trace-card,
	        .da-trace-call {
	            border: 1px solid #deded9;
	            border-radius: 0.55rem;
	            background: #fffefa;
	        }
	        .da-trace-card {
	            min-height: 4rem;
	            padding: 0.62rem 0.72rem;
	        }
	        .da-trace-card span {
	            display: block;
	            margin-bottom: 0.2rem;
	            color: #747782;
	            font-size: 0.78rem;
	            font-weight: 650;
	        }
	        .da-trace-card strong {
	            display: block;
	            color: #2e2f3a;
	            font-size: 0.9rem;
	            font-weight: 600;
	            line-height: 1.32;
	            overflow-wrap: anywhere;
	        }
	        .da-trace-call {
	            margin: 0.45rem 0 0.75rem;
	            padding: 0.7rem 0.78rem 0.1rem;
	        }
	        .da-trace-call h4 {
	            margin: 0 0 0.35rem;
	            font-size: 0.95rem;
	            line-height: 1.3;
	        }
	        .da-trace-meta {
	            display: flex;
	            flex-wrap: wrap;
	            gap: 0.35rem;
	            margin: 0.45rem 0 0.65rem;
	        }
	        .da-trace-meta span {
	            display: inline-flex;
	            align-items: center;
	            gap: 0.35rem;
	            border: 1px solid #deded9;
	            border-radius: 999px;
	            background: #fffefa;
	            padding: 0.22rem 0.52rem;
	            color: #2e2f3a;
	            font-size: 0.78rem;
	            line-height: 1.2;
	        }
	        .da-trace-meta em {
	            color: #747782;
	            font-style: normal;
	            font-weight: 700;
	        }
	        .da-json-trace {
	            margin: 0.55rem 0;
	            border: 1px solid #deded9;
	            border-radius: 0.55rem;
	            background: #fffefa;
	            overflow: hidden;
	        }
	        .da-json-trace summary {
	            display: flex;
	            align-items: center;
	            justify-content: space-between;
	            gap: 0.75rem;
	            cursor: pointer;
	            padding: 0.58rem 0.72rem;
	            color: #2e2f3a;
	            font-weight: 700;
	        }
	        .da-json-trace summary code {
	            color: #777b85;
	            background: #f5f4ef;
	            border-radius: 999px;
	            padding: 0.12rem 0.45rem;
	            font-size: 0.72rem;
	            font-weight: 600;
	        }
	        .da-json-trace pre {
	            margin: 0;
	            padding: 0.72rem 0.82rem;
	            border-top: 1px solid #ebe9e2;
	            background: #faf9f4;
	            color: #282a33;
	            font-size: 0.78rem;
	            line-height: 1.34;
	            max-height: 18rem;
	            overflow: auto;
	            white-space: pre-wrap;
	            overflow-wrap: anywhere;
	        }
	        .da-json-trace pre code {
	            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
	        }
        .da-runtime-flow {
            display: grid;
            gap: 0.42rem;
            margin: 0.2rem 0 0.1rem;
        }
        .da-runtime-step {
            display: grid;
            grid-template-columns: 0.9rem 1fr;
            gap: 0.5rem;
            align-items: start;
            padding: 0.58rem 0.68rem;
            border: 1px solid #deded9;
            border-radius: 0.55rem;
            background: #fffefa;
        }
        .da-runtime-status-dot,
        .da-runtime-model-dot {
            display: inline-block;
            flex: 0 0 auto;
            border-radius: 999px;
        }
        .da-runtime-status-dot {
            width: 0.52rem;
            height: 0.52rem;
            margin-top: 0.34rem;
            box-shadow: 0 0 0 0.18rem rgba(15, 31, 25, 0.04);
        }
        .da-runtime-status-dot--success {
            background: #00b368;
        }
        .da-runtime-status-dot--denied {
            background: #d94f3d;
        }
        .da-runtime-status-dot--neutral {
            background: #a4a4a4;
        }
        .da-runtime-step--denied {
            border-color: rgba(217, 79, 61, 0.42);
            background: #fff8f5;
        }
        .da-runtime-step-title {
            color: #151922;
            font-size: 0.84rem;
            font-weight: 750;
            line-height: 1.28;
            overflow-wrap: anywhere;
        }
        .da-runtime-step-text {
            color: #2e2f3a;
            font-size: 0.82rem;
            line-height: 1.38;
            margin-top: 0.12rem;
            overflow-wrap: anywhere;
        }
        .da-runtime-code {
            display: block;
            width: fit-content;
            max-width: 100%;
            margin: 0.28rem 0 0.2rem;
            padding: 0.32rem 0.46rem;
            border: 1px solid #d8d4c8;
            border-radius: 0.34rem;
            background: #eef7f4;
            color: #102c26;
            font-family: var(--da-mono);
            font-size: 0.78rem;
            line-height: 1.35;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        .da-runtime-step-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.32rem;
            margin-top: 0.4rem;
        }
        .da-runtime-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
            min-height: 1.35rem;
            padding: 0.08rem 0.42rem;
            border: 1px solid #deded9;
            border-radius: 999px;
            background: #f8f6ef;
            color: #2e2f3a;
            font-size: 0.72rem;
            font-weight: 650;
            line-height: 1.2;
        }
        .da-runtime-metric {
            display: inline-flex;
            align-items: center;
            gap: 0.24rem;
            min-height: 1.35rem;
            padding: 0.08rem 0.42rem;
            border: 1px solid #ece8dc;
            border-radius: 0.38rem;
            background: #fffdfa;
            color: #3b3d47;
            font-size: 0.72rem;
            line-height: 1.2;
        }
        .da-runtime-metric strong {
            color: #171b24;
            font-weight: 750;
        }
        .da-runtime-model-dot {
            width: 0.46rem;
            height: 0.46rem;
        }
        .da-runtime-model-dot--command {
            background: #c77de8;
        }
        .da-runtime-model-dot--tool {
            background: #5b78f0;
        }
        .da-runtime-model-dot--embed,
        .da-runtime-model-dot--rerank {
            background: #ff7759;
        }
        .da-runtime-model-dot--search {
            background: #0f8f83;
        }
        .da-runtime-model-dot--access {
            background: #00b368;
        }
        .stApp,
        .stApp p,
        .stApp label,
        .stApp input,
        .stApp textarea,
        .stApp button,
        .stApp [data-testid="stMarkdownContainer"] {
            font-family: var(--da-font);
        }
        .stApp {
            background: var(--da-off-white);
            color: var(--da-near-black);
        }
        .block-container {
            max-width: 1040px;
            padding-top: 1.35rem;
            padding-bottom: 2.5rem;
        }
        .da-title {
            margin: 0 0 0.1rem;
            color: var(--da-near-black);
            font-size: 1.58rem;
            font-weight: 720;
            line-height: 1.12;
            letter-spacing: 0;
        }
        .stApp h2,
        .stApp h3,
        .stApp h4 {
            color: var(--da-near-black);
            letter-spacing: 0;
        }
        .stApp h3 {
            font-size: 1.02rem;
            line-height: 1.25;
        }
        .stApp [data-testid="stCaptionContainer"] {
            color: #4f5a55;
            font-size: 0.78rem;
            line-height: 1.35;
        }
        section[data-testid="stSidebar"] {
            background: var(--da-green);
            border-right: 1px solid #0d4033;
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
            color: var(--da-off-white);
        }
        section[data-testid="stSidebar"] h2 {
            font-size: 1.02rem;
            font-weight: 720;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #cbd8d1;
        }
        div[data-testid="stTabs"] button {
            border-radius: var(--da-radius-sm) var(--da-radius-sm) 0 0;
            color: #33433d;
            font-size: 0.86rem;
            font-weight: 680;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--da-green);
            border-bottom-color: var(--da-accent);
        }
        div[data-testid="stButton"] button,
        div[data-testid="stFormSubmitButton"] button,
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            min-height: 2.15rem;
            border: 1px solid var(--da-border-strong);
            border-radius: var(--da-radius-sm);
            background: var(--da-panel);
            color: var(--da-near-black);
            box-shadow: none;
            font-size: 0.86rem;
            font-weight: 680;
        }
        div[data-testid="stFormSubmitButton"] button[kind="primary"] {
            border-color: var(--da-green);
            background: var(--da-green);
            color: var(--da-off-white);
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            border-color: var(--da-accent);
            color: var(--da-green);
        }
        div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background: #0a3b2e;
            color: var(--da-off-white);
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="radio"] [role="radio"] {
            border-radius: var(--da-radius-sm);
        }
        textarea,
        input {
            color: var(--da-near-black);
            font-size: 0.9rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--da-border);
            border-radius: var(--da-radius);
            background: var(--da-panel);
            box-shadow: var(--da-shadow);
        }
        div[data-testid="stMetric"] {
            min-height: 4.2rem;
            padding: 0.55rem 0.65rem;
            border: 1px solid var(--da-border);
            border-radius: var(--da-radius);
            background: var(--da-panel);
            box-shadow: var(--da-shadow);
        }
        div[data-testid="stMetricLabel"] {
            color: #52645d;
            font-size: 0.72rem;
            font-weight: 720;
        }
        div[data-testid="stMetricValue"] {
            color: var(--da-near-black);
            font-size: 1rem;
            font-weight: 720;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border: 1px solid var(--da-border);
            border-radius: var(--da-radius);
            background: var(--da-panel);
            box-shadow: var(--da-shadow);
            overflow: hidden;
        }
        div[data-testid="stAlert"] {
            border-radius: var(--da-radius);
            border-color: var(--da-border);
            background: var(--da-panel);
            color: var(--da-near-black);
            font-size: 0.88rem;
        }
        .da-answer {
            padding: 0.15rem 0 0.25rem;
            color: var(--da-near-black);
            font-size: 0.95rem;
            line-height: 1.58;
        }
        .da-selected-evidence-card,
        .da-trace-timeline > div,
        .da-trace-card,
        .da-trace-call,
        .da-json-trace,
        .da-page-text {
            border-color: var(--da-border);
            border-radius: var(--da-radius);
            background: var(--da-panel);
            box-shadow: var(--da-shadow);
        }
        .da-selected-evidence-card {
            padding: 0.85rem;
        }
        .da-selected-evidence-card h3 {
            color: var(--da-near-black);
            font-size: 0.98rem;
        }
        .da-selected-span {
            color: var(--da-green);
            font-weight: 680;
        }
        .da-source-caption,
        .da-source-metadata,
        .da-source-muted,
        .da-trace-timeline span,
        .da-trace-card span {
            color: #5d6762;
        }
        .da-source-action {
            border-color: var(--da-border);
            border-radius: var(--da-radius-sm);
            background: var(--da-off-white);
            color: var(--da-near-black) !important;
            font-size: 0.84rem;
        }
        .da-source-action:hover,
        .da-source-action:focus {
            border-color: var(--da-accent);
            background: var(--da-pale-green);
            color: var(--da-green) !important;
        }
        .da-page-preview img {
            border-color: var(--da-border);
            border-radius: var(--da-radius-sm);
        }
        .da-trace-meta span {
            border-color: var(--da-border);
            border-radius: var(--da-radius-sm);
            background: var(--da-off-white);
            color: var(--da-near-black);
        }
        .da-trace-meta em {
            color: var(--da-green);
        }
        .da-json-trace summary {
            color: var(--da-near-black);
            font-size: 0.86rem;
        }
        .da-json-trace summary code {
            border-radius: var(--da-radius-sm);
            background: var(--da-off-white);
            color: #52645d;
            font-family: var(--da-mono);
        }
        .da-json-trace pre {
            border-top-color: var(--da-border);
            background: #fbfaf5;
            color: var(--da-near-black);
            font-family: var(--da-mono);
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #dce5df;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            background: var(--da-off-white);
            color: var(--da-near-black);
            opacity: 1;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] button span {
            color: var(--da-near-black) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button:disabled {
            border-color: rgba(240, 238, 233, 0.56);
            background: rgba(240, 238, 233, 0.16);
            color: #e2e7e3;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button:disabled p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] button:disabled span {
            color: #e2e7e3 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            border-color: rgba(240, 238, 233, 0.42);
            background: rgba(240, 238, 233, 0.1);
            box-shadow: none;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary span {
            color: var(--da-off-white) !important;
            opacity: 1;
        }
        section[data-testid="stSidebar"] code {
            background: rgba(184, 249, 243, 0.45);
            color: var(--da-green);
        }
        @media (max-width: 720px) {
            .block-container {
                padding: 0.85rem 0.78rem 2rem;
            }
            .da-title {
                font-size: 1.35rem;
            }
            .da-source-actions {
                grid-template-columns: 1fr;
            }
            .da-trace-card-grid {
                grid-template-columns: 1fr;
            }
        }
	        </style>
        """


if __name__ == "__main__":
    main()
