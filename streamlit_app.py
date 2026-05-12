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
from defence_agent.ui.view_model import (
    DEFAULT_UI_PERSONA_ID,
    UI_PERSONAS,
    CitationView,
    DefenceAgentViewModel,
    EvidencePageView,
    SourceView,
    UiPersona,
    build_view_model,
    citation_chip_label,
    persona_for_ui_id,
)


EXAMPLE_PROMPTS = {
    "Cited lookup": "Write a 150-200 word cited planning answer explaining NATO's core tasks and why they matter for a Canadian planning brief.",
    "Access control": "For the new sensor-fusion release workflow, what rule should planning staff follow before sharing a candidate observation?",
    "Multi-step comparison": "First find the DND/CAF AI Strategy's main lines of effort. Then search for Canada's defence-policy modernization priorities and compare the overlap for a planning audience.",
    "Bilingual query": "Quelles taches fondamentales l'OTAN attribue-t-elle a l'Alliance dans son Concept strategique, et pourquoi sont-elles importantes pour un brief de planification canadien?",
    "Refusal / unsupported": "What does the corpus say about the approved Arctic submarine basing schedule for 2031?",
}
RUN_MODES = {
    "demo": "Demo replay",
    "live": "Live Cohere/ADK",
}
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


@dataclass(frozen=True)
class DemoReplay:
    transcript_dir: Path
    transcript_name: str
    demo_point: str


DEMO_REPLAYS = {
    ("persona_a", "Cited lookup"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "natural_multilingual_nato_core_tasks.json",
        "Cited planning answer using English and French NATO source pages.",
    ),
    ("persona_a", "Access control"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "acl_unclassified_sensor_fusion_release_rule.json",
        "Unauthorized persona receives a refusal with restricted evidence excluded.",
    ),
    ("persona_a", "Multi-step comparison"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "flagship_planning_brief_modernization.json",
        "Multi-document planning comparison with public source evidence.",
    ),
    ("persona_a", "Bilingual query"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "natural_multilingual_nato_core_tasks.json",
        "Language-agnostic retrieval across English and French NATO pages.",
    ),
    ("persona_a", "Refusal / unsupported"): DemoReplay(
        INSUFFICIENT_EVIDENCE_TRANSCRIPT_DIR,
        "insufficient_evidence_planning_topic.json",
        "Unsupported query refuses without falling back to speculation.",
    ),
    ("persona_b", "Cited lookup"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "natural_multilingual_nato_core_tasks.json",
        "Cleared user cites authorized unclassified doctrine pages.",
    ),
    ("persona_b", "Access control"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "acl_secret_sensor_fusion_release_rule.json",
        "Cleared user receives the restricted workflow rule with traceability.",
    ),
    ("persona_b", "Multi-step comparison"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "flagship_planning_brief_modernization.json",
        "Cleared user cites authorized public planning evidence.",
    ),
    ("persona_b", "Bilingual query"): DemoReplay(
        DEMO_TRANSCRIPT_DIR,
        "natural_multilingual_nato_core_tasks.json",
        "Cleared user retrieves authorized English and French source pages.",
    ),
    ("persona_b", "Refusal / unsupported"): DemoReplay(
        INSUFFICIENT_EVIDENCE_TRANSCRIPT_DIR,
        "insufficient_evidence_planning_topic.json",
        "Cleared user still receives refusal when no evidence supports the claim.",
    ),
}


class DemoReplayUnavailable(RuntimeError):
    """Raised when a saved replay cannot be used for the selected UI state."""


def main() -> None:
    st.set_page_config(page_title="Defence Agent", layout="centered", initial_sidebar_state="expanded")
    _apply_styles()
    _init_state()

    st.markdown("<h1 class='da-title'>Defence Agent</h1>", unsafe_allow_html=True)
    st.caption("Ask cited questions over approved manuals, procedures, and doctrine.")

    ask_tab, trace_tab, eval_tab = st.tabs(["Ask", "Trace", "Eval"])
    with ask_tab:
        _render_ask_tab()
    with trace_tab:
        _render_trace_tab()
    with eval_tab:
        _render_eval_tab()


def _init_state() -> None:
    defaults = {
        "ui_persona_id": DEFAULT_UI_PERSONA_ID,
        "_active_ui_persona_id": DEFAULT_UI_PERSONA_ID,
        "query_text": EXAMPLE_PROMPTS["Cited lookup"],
        "selected_example": "Cited lookup",
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
    persona = _render_query_controls()

    if st.session_state.get("last_run_error"):
        st.error(st.session_state["last_run_error"])

    view_model = st.session_state.get("last_result")
    if view_model is None:
        st.info("Run a query to see a cited answer and the evidence used.")
        return

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

        selected_example = st.selectbox(
            "Example prompt",
            options=tuple(EXAMPLE_PROMPTS.keys()),
            key="selected_example",
        )
        if st.button("Use example", width="stretch"):
            st.session_state["query_text"] = EXAMPLE_PROMPTS[selected_example]

        with st.expander("Advanced run controls", expanded=False):
            run_mode = st.radio(
                "Run mode",
                options=tuple(RUN_MODES.keys()),
                format_func=lambda key: RUN_MODES[key],
                key="run_mode",
            )
            if run_mode == "demo":
                st.caption("Demo replay loads saved live runs immediately. It does not call Cohere.")
            else:
                st.caption("Live Cohere/ADK makes a fresh backend call and shows the elapsed timer.")
            st.checkbox("Stream answer display", key="stream_answer")
            st.caption("Streams the answer after the run completes. Backend calls are still replay or live as selected.")

        st.markdown("**Runtime**")
        st.caption(f"`run_turn` · `search_documents` · {DEFAULT_TIMEOUT_SECONDS}s timeout")

    st.caption(
        f"Persona: {persona.label} · Allowed evidence: {persona.visible_access_label} · "
        f"Run mode: {RUN_MODES[run_mode]}"
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
        _run_query(query=query, persona=persona, selected_example=selected_example, run_mode=run_mode)
    return persona


def _reset_result_when_persona_changes(persona: UiPersona) -> None:
    if st.session_state["_active_ui_persona_id"] == persona.ui_id:
        return
    st.session_state["_active_ui_persona_id"] = persona.ui_id
    st.session_state["last_result"] = None
    st.session_state["selected_citation_id"] = None
    st.session_state["selected_page_key"] = None
    st.session_state["last_run_error"] = ""
    st.session_state["run_active"] = False


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
    try:
        if run_mode == "demo":
            expected_query = EXAMPLE_PROMPTS[selected_example].strip()
            if cleaned != expected_query:
                st.warning(
                    "Demo replay uses the saved run for the selected example. "
                    "Switch to Live Cohere/ADK for edited prompts."
                )
                return
            status_slot = st.empty()
            with status_slot.status("Running Defence Agent", expanded=True):
                _write_status_step("Applying persona access policy")
                _write_status_step("Loading saved live run")
                result = _load_demo_result(selected_example=selected_example, persona=persona, query=cleaned)
                _write_status_step("Resolving citations")
                elapsed = time.monotonic() - started
                view_model = build_view_model(result, ui_persona_id=persona.ui_id, run_elapsed_seconds=elapsed)
                _write_status_step("Building answer audit")
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
                    progress=lambda _message: None,
                )
                while not future.done():
                    elapsed = time.monotonic() - started
                    timer_slot.caption(
                        f"Live Cohere/ADK run · {_format_elapsed(elapsed)} elapsed · "
                        f"hard stop {DEFAULT_TIMEOUT_SECONDS}s"
                    )
                    time.sleep(0.75)
                result = future.result()

            elapsed = time.monotonic() - started
            st.write("Resolving citations and building answer audit")
        view_model = build_view_model(result, ui_persona_id=persona.ui_id, run_elapsed_seconds=elapsed)
        _store_result(view_model, result, persona)
        status_slot.empty()
        timer_slot.empty()
    except Exception as exc:
        st.session_state["last_run_error"] = _friendly_run_error(exc)
    finally:
        st.session_state["run_active"] = False


def _write_status_step(message: str, *, delay_seconds: float = 0.16) -> None:
    st.write(message)
    time.sleep(delay_seconds)


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


def _load_demo_result(*, selected_example: str, persona: UiPersona, query: str) -> AgentTurnResult:
    replay = DEMO_REPLAYS.get((persona.ui_id, selected_example))
    if replay is None:
        raise DemoReplayUnavailable("No saved replay exists for this persona and example. Switch to Live Cohere/ADK.")
    path = replay.transcript_dir / replay.transcript_name
    if not path.exists():
        raise DemoReplayUnavailable("Saved replay file is missing. Switch to Live Cohere/ADK.")

    data = json.loads(path.read_text(encoding="utf-8"))
    audit = dict(data.get("final_answer_audit", {}) or {})
    if not str(audit.get("retrieval_status", "") or "").strip():
        audit["retrieval_status"] = "replay verified"
    retrieval = dict(audit.get("retrieval", {}) or {})
    _validate_replay_acl(audit, persona=persona)
    retrieval["allowed_access"] = list(persona.allowed_access)
    audit["retrieval"] = retrieval
    audit["persona_id"] = persona.backend_persona_id
    audit["user_id"] = persona.backend_persona_id
    audit["query"] = query
    generation = audit.get("generation", {}) if isinstance(audit.get("generation"), dict) else {}
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
                    "Saved replay contains evidence outside the selected persona access policy."
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
    progress,
) -> AgentTurnResult:
    return backend_bridge.run_turn_in_subprocess(
        query=query,
        persona_id=persona.backend_persona_id,
        user_id=persona.backend_persona_id,
        session_id=session_id,
        progress=progress,
    )


def _render_answer(view_model: DefenceAgentViewModel) -> None:
    with st.container(border=True):
        _render_answer_text(view_model)
        _render_citation_controls(view_model)
        _render_source_pages_used(view_model)
        _render_selected_evidence(view_model)


def _render_answer_text(view_model: DefenceAgentViewModel) -> None:
    if st.session_state.get("stream_answer_once"):
        _stream_answer_display(view_model)
        st.session_state["stream_answer_once"] = False
        return
    st.markdown(_answer_html_with_inline_citations(view_model), unsafe_allow_html=True)


def _stream_answer_display(view_model: DefenceAgentViewModel) -> None:
    text = _inline_answer_base_text(view_model)
    if not text:
        st.markdown(_answer_html_with_inline_citations(view_model), unsafe_allow_html=True)
        return

    placeholder = st.empty()
    words = text.split()
    chunk_size = 10
    for end in range(chunk_size, len(words) + chunk_size, chunk_size):
        partial = " ".join(words[: min(end, len(words))])
        placeholder.markdown(_plain_answer_html(partial), unsafe_allow_html=True)
        time.sleep(0.018)
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
        markers_by_end.setdefault(end, []).append(
            f'<a class="da-inline-cite" href="{href}" target="_self" '
            f'data-citation-id="{html.escape(citation.citation_id, quote=True)}" '
            f'data-citation-marker="{html.escape(citation.marker, quote=True)}" data-tooltip="{tooltip}" '
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


def _render_citation_controls(view_model: DefenceAgentViewModel) -> None:
    if not view_model.citations:
        st.caption("No citations were returned for this answer.")
        return

    st.markdown("**Citations**")
    st.caption("A citation is a Cohere-linked answer span mapped to authorized source pages.")
    for citation in view_model.citations:
            _render_anchor_button(
                label=citation_chip_label(citation, view_model.sources),
                href=_citation_href(citation.citation_id),
                css_class="da-citation-link",
                citation_id=citation.citation_id,
        )


def _render_source_pages_used(view_model: DefenceAgentViewModel) -> None:
    if not view_model.evidence_pages:
        return

    label = "Evidence page used" if len(view_model.evidence_pages) == 1 else "Evidence pages used"
    st.markdown(f"**{label}**")
    st.caption("Repeated citations from the same document page are grouped here.")
    for page in view_model.evidence_pages:
        marker_list = ", ".join(
            citation.marker for citation in view_model.citations if citation.citation_id in page.citation_ids
        )
        button_label = f"{page.display_index}. {page.label}"
        if marker_list:
            button_label += f" · {marker_list}"
        _render_anchor_button(
            label=button_label,
            href="#selected-evidence-anchor",
            css_class="da-evidence-link",
            page_key=page.page_key,
        )


def _render_anchor_button(
    *,
    label: str,
    href: str,
    css_class: str,
    citation_id: str | None = None,
    page_key: str | None = None,
) -> None:
    attributes = [
        f'class="da-list-button {html.escape(css_class, quote=True)}"',
        f'href="{html.escape(href, quote=True)}"',
        'target="_self"',
    ]
    if citation_id:
        attributes.append(f'data-citation-id="{html.escape(citation_id, quote=True)}"')
    if page_key:
        attributes.append(f'data-page-key="{html.escape(page_key, quote=True)}"')
    st.markdown(
        f"<a {' '.join(attributes)}>{html.escape(label)}</a>",
        unsafe_allow_html=True,
    )


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
    for page in view_model.evidence_pages:
        source = page.source
        preview = previews.get(source.source_id)
        if preview is None:
            preview = resolve_source_preview(source, backend_persona_id=view_model.backend_persona_id)
            previews[source.source_id] = preview
        citation = _citation_for_evidence_page(view_model, page)
        payloads[page.page_key] = {
            "page_key": page.page_key,
            "citation_ids": list(page.citation_ids),
            "primary_citation_id": citation.citation_id if citation else "",
            "html": _selected_evidence_card_html(citation, source, preview),
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
    payload_json = json.dumps(payloads).replace("</", "<\\/")
    initial_page_json = json.dumps(selected_page_key)
    initial_citation_json = json.dumps(selected_citation_id)
    st.iframe(
        f"""
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
            const panel = parentDoc.querySelector("[data-da-selected-evidence-panel]");
            if (panel) {{
                panel.innerHTML = record.html;
            }}
            const citationIds = record.citation_ids || [];
            const activeCitationId = citationIds.includes(preferredCitationId)
                ? preferredCitationId
                : (record.primary_citation_id || citationIds[0] || "");
            parentDoc.querySelectorAll("[data-page-key]").forEach((node) => {{
                node.classList.toggle("da-active-page", node.dataset.pageKey === pageKey);
            }});
            parentDoc.querySelectorAll("[data-citation-id]").forEach((node) => {{
                node.classList.toggle("da-active-citation", Boolean(activeCitationId) && node.dataset.citationId === activeCitationId);
            }});
            const anchor = parentDoc.getElementById("selected-evidence-anchor");
            if (shouldScroll && anchor) {{
                parentWin.history.replaceState(null, "", "#selected-evidence-anchor");
                anchor.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
        }};
        parentWin.__defenceAgentSelectCitation = (citationId) => {{
            const record = parentWin.__defenceAgentEvidenceRecordForCitation(citationId);
            if (!record) return;
            parentWin.__defenceAgentSelectPage(record.page_key, citationId);
        }};
        if (!parentWin.__defenceAgentCitationClickInstalled) {{
            parentDoc.addEventListener("click", (event) => {{
                const pageTarget = event.target.closest("[data-page-key]");
                if (pageTarget) {{
                    const pageKey = pageTarget.dataset.pageKey;
                    const evidence = parentWin.__defenceAgentEvidencePages || {{}};
                    if (!pageKey || !evidence[pageKey]) return;
                    event.preventDefault();
                    parentWin.__defenceAgentSelectPage(pageKey, evidence[pageKey].primary_citation_id || "");
                    return;
                }}
                const citationTarget = event.target.closest("[data-citation-id]");
                if (!citationTarget) return;
                const citationId = citationTarget.dataset.citationId;
                if (!citationId || !parentWin.__defenceAgentEvidenceRecordForCitation(citationId)) return;
                event.preventDefault();
                parentWin.__defenceAgentSelectCitation(citationId);
            }}, true);
            parentWin.__defenceAgentCitationClickInstalled = true;
        }}
        parentWin.__defenceAgentSelectPage({initial_page_json}, {initial_citation_json}, false);
        </script>
        """,
        height=1,
    )


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
        source.language,
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
    st.markdown(
        """
        <div class="da-trace-timeline">
          <div><strong>1. Access policy applied</strong><span>Persona-derived evidence boundary was set before retrieval.</span></div>
          <div><strong>2. Search executed</strong><span>The agent used read-only document search.</span></div>
          <div><strong>3. Evidence filtered</strong><span>Only authorized evidence was sent forward.</span></div>
          <div><strong>4. Answer generated</strong><span>Cohere generated a grounded answer from the selected evidence.</span></div>
          <div><strong>5. Citations resolved</strong><span>Cohere citation objects were mapped back to source pages.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Retrieval status", _compact_label(view_model.retrieval_status or "unknown", 22))
    cols[1].metric("Answerability", view_model.answerability.lower())
    cols[2].metric("Tool calls", str(len(view_model.tool_calls)))
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
    st.caption(
        "Automated checks verify retrieval flow and citation-source resolution. "
        "They do not replace human source review."
    )


def _request_rows(view_model: DefenceAgentViewModel) -> list[dict[str, object]]:
    return [
        {"field": "session_id", "value": view_model.session_id},
        {"field": "persona", "value": view_model.persona_label},
        {"field": "backend_persona_id", "value": view_model.backend_persona_id},
        {"field": "allowed_access", "value": ", ".join(view_model.allowed_access)},
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
    st.markdown(
        """
        <style>
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
        .da-list-button {
            display: block;
            width: 100%;
            margin: 0.42rem 0;
            padding: 0.66rem 0.78rem;
            border: 1px solid #d8d8d4;
            border-radius: 0.5rem;
            background: #ffffff;
            color: #2e2f3a !important;
            font-size: 0.92rem;
            font-weight: 500;
            line-height: 1.35;
            text-align: left;
            text-decoration: none !important;
        }
        .da-list-button:hover,
        .da-list-button:focus {
            border-color: #82b6a1;
            background: #f4fbf8;
            color: #185b45 !important;
        }
        .da-list-button.da-active-citation,
        .da-list-button.da-active-page,
        .da-inline-cite.da-active-citation {
            border-color: #2f8f68;
            background: #e4f5ee;
            color: #124735 !important;
        }
        .da-answer {
            font-size: 1rem;
            line-height: 1.62;
        }
        .da-answer p {
            margin: 0 0 0.85rem 0;
        }
        .da-inline-cite {
            position: relative;
            display: inline-flex;
            align-items: center;
            margin-left: 0.18rem;
            padding: 0.05rem 0.34rem;
            border: 1px solid #b9d7cc;
            border-radius: 999px;
            background: #edf8f3;
            color: #185b45 !important;
            font-size: 0.76em;
            font-weight: 650;
            line-height: 1.25;
            text-decoration: none !important;
            transform: translateY(-0.04rem);
        }
        .da-inline-cite:hover,
        .da-inline-cite:focus {
            border-color: #82b6a1;
            background: #f4fbf8;
            color: #185b45 !important;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
