from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from defence_agent.session import AgentTurnResult
from defence_agent.ui.view_model import build_view_model
import streamlit_app


ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = ROOT / "defence_agent" / "data" / "transcripts" / "live_readiness_final_20260511_013937"

pytestmark = pytest.mark.streamlit


@pytest.fixture(autouse=True)
def _disable_ui_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(streamlit_app, "GUIDED_STEP_DELAY_SECONDS", 0)
    monkeypatch.setattr(streamlit_app, "GUIDED_MIN_RUN_SECONDS_BY_ROUTE", {})
    monkeypatch.setattr(streamlit_app, "LIVE_STATUS_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(streamlit_app, "ANSWER_STREAM_DELAY_SECONDS", 0)
    monkeypatch.setattr(streamlit_app, "ANSWER_STREAM_CHUNK_WORDS", 200)


def _result_from_transcript(name: str) -> AgentTurnResult:
    data = json.loads((TRANSCRIPTS / name).read_text(encoding="utf-8"))
    audit = data["final_answer_audit"]
    generation = audit.get("generation", {})
    return AgentTurnResult(
        session_id=str(audit.get("session_id", "test_session")),
        user_id=str(audit.get("user_id", data.get("persona_id", ""))),
        persona_id=str(audit.get("persona_id", data.get("persona_id", ""))),
        answer=str(data.get("answer", "")),
        raw_answer=str(data.get("answer", "")),
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


def test_scanned_manual_demo_query_uses_guided_replay_when_unchanged() -> None:
    prompt = streamlit_app.EXAMPLE_PROMPTS["Scanned manual retrieval"]

    assert streamlit_app._effective_run_mode("Scanned manual retrieval", "demo") == "demo"
    assert streamlit_app._target_answer_language(prompt, "Scanned manual retrieval") == "en"
    assert streamlit_app._can_use_guided_replay("demo", "Scanned manual retrieval", prompt)


def test_streamlit_demo_replay_renders_single_column_result_without_backend_call() -> None:
    with patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess") as backend:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)
        app.main.button[0].click()
        app.run(timeout=20)

    assert not app.exception
    backend.assert_not_called()
    rendered_markdown = "\n".join(item.value for item in app.markdown)
    assert "Our North, Strong and Free" in rendered_markdown
    assert app.main.expander[0].label == "Runtime flow"
    assert "da-runtime-flow" in rendered_markdown
    assert "da-runtime-status-dot--success" in rendered_markdown
    assert "da-runtime-model-dot--command" in rendered_markdown
    assert "Embed v4" in rendered_markdown
    assert "Rerank v4" in rendered_markdown
    assert "da-runtime-state" not in rendered_markdown
    assert "da-runtime-model-dot--database" not in rendered_markdown
    assert "COMPLETE" not in rendered_markdown
    assert "1. User query" in rendered_markdown
    assert "Agent runtime and session context" in rendered_markdown
    assert "Model runtime tool plan" in rendered_markdown
    assert "search_documents" in rendered_markdown
    assert "da-runtime-code" in rendered_markdown
    assert (
        "search_documents(query=&quot;Canada&#x27;s defence policy AI-enabled modernization&quot;, top_k=8)"
        in rendered_markdown
    )
    assert (
        "search_documents(query=&quot;DND/CAF AI Strategy AI-enabled modernization&quot;, top_k=8)"
        in rendered_markdown
    )
    assert "Reviewer sub-agent trust check" in rendered_markdown
    assert "1.0" in rendered_markdown
    assert "Authorized pages" in rendered_markdown
    assert "Vector match" not in rendered_markdown
    assert "Rerank relevance" not in rendered_markdown
    assert "Chroma" not in rendered_markdown
    assert "Language any" not in rendered_markdown
    assert "Status filter approved" not in rendered_markdown
    assert "Excluded pages 0" not in rendered_markdown
    assert "Retrieval status" not in rendered_markdown
    assert "Reason authorized_evidence_sent" not in rendered_markdown
    assert "confidence" not in rendered_markdown.lower()
    assert "Evidence check" in rendered_markdown
    assert "Grounded generation" in rendered_markdown
    assert "UI output" in rendered_markdown
    assert rendered_markdown.find("1. User query") < rendered_markdown.find('class="da-answer"')
    assert "da-inline-cite" in rendered_markdown
    assert 'href="#selected-evidence-anchor"' in rendered_markdown
    assert 'target="_self"' in rendered_markdown
    assert 'data-citation-id="citation_1"' in rendered_markdown
    assert "da-citation-chip" in rendered_markdown
    assert 'data-citation-page-key=' in rendered_markdown
    evidence_links = re.findall(r'<a class="da-list-button da-evidence-link"[^>]*>', rendered_markdown)
    assert evidence_links
    assert all("data-page-key=" in link for link in evidence_links)
    assert all("data-citation-id=" not in link for link in evidence_links)
    assert "Source Inspector" not in rendered_markdown
    assert "Source details" not in rendered_markdown
    assert "Sources used" not in rendered_markdown
    assert "Citations" in rendered_markdown
    assert "Cited answer spans" not in rendered_markdown
    assert "Evidence pages used" not in rendered_markdown
    assert "Source pages used" not in rendered_markdown
    assert "Selected evidence" in rendered_markdown
    assert "Claims cited" not in rendered_markdown
    assert "ANSWERED" not in rendered_markdown
    assert "[C1" not in rendered_markdown
    assert "Open publisher page" not in rendered_markdown
    assert "Loaded saved live run" not in rendered_markdown
    assert "Defence Agent completed" not in rendered_markdown
    trace_text = rendered_markdown + "\n" + "\n".join(item.value for item in app.caption)
    assert "google_adk" in trace_text
    assert "tool_calls" in trace_text
    captions = "\n".join(item.value for item in app.caption)
    assert "Source pages linked from inline citation markers" in captions
    assert "Answer trace below keeps the full citation-span map" in captions
    assert "Cohere-linked answer span" not in captions
    assert app.sidebar.radio[0].label == "Persona"
    assert app.sidebar.selectbox[0].label == "Demo query"
    assert len(app.sidebar.button) == 0
    assert app.sidebar.radio[1].label == "Run mode"


def test_streamlit_renders_core_tabs_without_reports() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.run(timeout=20)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Ask", "Trace", "Eval", "Database"]


def test_database_tab_renders_without_backend_call() -> None:
    with patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess") as backend:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)

    assert not app.exception
    backend.assert_not_called()
    rendered = "\n".join(item.value for item in list(app.subheader) + list(app.caption))
    assert "Database" in rendered
    assert "Approved source catalog filtered by persona access." in rendered
    assert any(metric.label == "Visible docs" and metric.value == "8" for metric in app.metric)
    assert any(metric.label == "Visible pages" and metric.value == "211" for metric in app.metric)
    assert any(metric.label == "Withheld docs" and metric.value == "2" for metric in app.metric)


def test_database_tab_does_not_expose_restricted_document_names_for_persona_a() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.run(timeout=20)

    assert not app.exception
    rendered = "\n".join(
        [item.value for item in list(app.markdown) + list(app.caption)]
        + [str(frame.value) for frame in app.dataframe]
    )
    assert "Fusion Model Release Control Procedure" not in rendered
    assert "Fusion Model Restricted Routing Annex" not in rendered
    assert "SYN-FUSION-S-RELEASE-001" not in rendered
    assert "SYN-FUSION-TS-ANNEX-002" not in rendered


def test_streamlit_live_mode_calls_backend() -> None:
    result = _result_from_transcript("natural_multilingual_nato_core_tasks.json")

    with (
        patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess", return_value=result) as backend,
        patch("streamlit_app._cohere_api_key_available", return_value=True),
    ):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)
        app.sidebar.radio[1].set_value("live")
        app.run(timeout=20)
        app.main.button[0].click()
        app.run(timeout=20)

    assert not app.exception
    backend.assert_called_once()
    assert backend.call_args.kwargs["run_mode"] == streamlit_app.REVIEWED_AGENT
    assert backend.call_args.kwargs["retrieval_mode"] == "hybrid"
    assert backend.call_args.kwargs["chunk_strategy"] == "page"
    assert app.main.expander[0].label == "Runtime flow"
    rendered_markdown = "\n".join(item.value for item in app.markdown)
    assert "da-runtime-flow" in rendered_markdown
    assert "da-runtime-status-dot--success" in rendered_markdown
    assert "1. User query" in rendered_markdown
    assert "Agent runtime and session context" in rendered_markdown
    assert "Evidence check" in rendered_markdown
    assert "search_documents" in rendered_markdown
    assert "UI output" in rendered_markdown


def test_streamlit_scanned_manual_option_uses_guided_replay_without_backend_call() -> None:
    with patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess") as backend:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)
        app.sidebar.selectbox[0].set_value("Scanned manual retrieval")
        app.run(timeout=20)
        app.main.button[0].click()
        app.run(timeout=20)

    assert not app.exception
    backend.assert_not_called()
    rendered = "\n".join(item.value for item in app.markdown)
    assert "technical intelligence" in rendered.lower()


def test_streamlit_edited_prompt_routes_to_live_backend_without_replay_warning() -> None:
    result = _result_from_transcript("natural_multilingual_nato_core_tasks.json")

    with (
        patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess", return_value=result) as backend,
        patch("streamlit_app._cohere_api_key_available", return_value=True),
    ):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)
        app.text_area[0].set_value("Edited prompt that does not match the saved run.")
        app.main.button[0].click()
        app.run(timeout=20)

    assert not app.exception
    backend.assert_called_once()
    warnings = "\n".join(item.value for item in app.warning)
    assert "Demo path" not in warnings


def test_streamlit_live_submit_shows_concise_timeout_error() -> None:
    with (
        patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess", side_effect=TimeoutError("raw payload")),
        patch("streamlit_app._cohere_api_key_available", return_value=True),
    ):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)
        app.sidebar.radio[1].set_value("live")
        app.run(timeout=20)
        app.main.button[0].click()
        app.run(timeout=20)

    assert not app.exception
    errors = "\n".join(item.value for item in app.error)
    assert "live run timed out" in errors.lower()
    assert "no transcript fallback was used" in errors.lower()
    assert "raw payload" not in errors


def test_inline_answer_html_uses_numeric_links_and_hover_tooltips() -> None:
    result = _result_from_transcript("natural_multilingual_nato_core_tasks.json")
    view_model = build_view_model(result, ui_persona_id="persona_a")

    rendered = streamlit_app._answer_html_with_inline_citations(view_model)

    assert "da-inline-cite" in rendered
    assert 'data-citation-trust="unreviewed"' in rendered
    assert "Trust: not checked" in rendered
    assert 'href="#selected-evidence-anchor"' in rendered
    assert 'target="_self"' in rendered
    assert 'data-citation-id="citation_1"' in rendered
    assert 'data-citation-marker="[1]"' in rendered
    assert "data-tooltip=" in rendered
    assert "NATO 2022 Strategic Concept" in rendered
    assert "p.4" in rendered
    assert "[1]" in rendered
    assert "[C1" not in rendered


def test_multilingual_retrieval_summary_counts_english_and_french_pages() -> None:
    result = _result_from_transcript("natural_multilingual_nato_core_tasks.json")
    view_model = build_view_model(result, ui_persona_id="persona_a")

    payload = streamlit_app._multilingual_retrieval_payload(view_model)
    assert payload is not None
    assert payload["sent_counts"] == {"en": 3, "fr": 5}
    assert payload["cited_counts"] == {"en": 3, "fr": 5}

    rendered = streamlit_app._multilingual_retrieval_html(payload)
    assert "Language mix" in rendered
    assert "Candidate pages" in rendered
    assert "Cited pages" in rendered
    assert "French 5 pages" in rendered
    assert "English 3 pages" in rendered


def test_evidence_page_buttons_show_source_language_tags() -> None:
    result = _result_from_transcript("natural_multilingual_nato_core_tasks.json")
    view_model = build_view_model(result, ui_persona_id="persona_a")

    labels = [streamlit_app._evidence_page_button_label(page) for page in view_model.evidence_pages]

    assert any("NATO 2022 Strategic Concept" in label and "English" in label for label in labels)
    assert any("Concept stratégique 2022 de l'OTAN" in label and "French" in label for label in labels)


def test_evidence_page_card_maps_citation_chips_to_page_key() -> None:
    audit = {
        "session_id": "chip_map_session",
        "user_id": "clearance_unclassified",
        "persona_id": "clearance_unclassified",
        "query": "Summarize cited claims.",
        "retrieval": {
            "allowed_access": ["unclassified"],
            "sources_sent_to_answer": [
                {
                    "source_id": "SOURCE-1_page_001",
                    "doc_id": "SOURCE-1",
                    "title": "Mapped Source",
                    "page": "1",
                    "access_level": "unclassified",
                    "language": "en",
                },
            ],
        },
        "generation": {"citation_mode": "cohere_native_accurate_default", "document_count": 1},
        "citations": [
            {
                "type": "TEXT_CONTENT",
                "start": 0,
                "end": 11,
                "text": "First claim",
                "sources": [{"source_id": "SOURCE-1_page_001", "doc_id": "SOURCE-1", "page": "1"}],
            },
            {
                "type": "TEXT_CONTENT",
                "start": 12,
                "end": 24,
                "text": "Second claim",
                "sources": [{"source_id": "SOURCE-1_page_001", "doc_id": "SOURCE-1", "page": "1"}],
            },
        ],
    }
    result = AgentTurnResult(
        session_id="chip_map_session",
        user_id="clearance_unclassified",
        persona_id="clearance_unclassified",
        answer="First claim. Second claim.",
        raw_answer="First claim. Second claim.",
        events_seen=0,
        citations=audit["citations"],
        citation_mode="cohere_native_accurate_default",
        documents_sent_to_model=1,
        answer_audit=audit,
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")
    page = view_model.evidence_pages[0]

    rendered = streamlit_app._evidence_page_card_html(
        page,
        "1. Mapped Source",
        {citation.citation_id: citation for citation in view_model.citations},
    )

    assert rendered.count("da-citation-chip") == 2
    assert 'data-citation-id="citation_1"' in rendered
    assert 'data-citation-id="citation_2"' in rendered
    assert rendered.count(f'data-citation-page-key="{page.page_key}"') == 2
    assert 'data-page-key="' in rendered
    assert 'data-citation-id="citation_1" data-citation-page-key=' in rendered
    assert "data-page-card-key" not in rendered


def test_evidence_payloads_are_keyed_by_page_for_multi_page_citation() -> None:
    audit = {
        "session_id": "multi_page_session",
        "user_id": "clearance_unclassified",
        "persona_id": "clearance_unclassified",
        "query": "Summarize the two-page source.",
        "retrieval": {
            "allowed_access": ["unclassified"],
            "sources_sent_to_answer": [
                {
                    "source_id": "SOURCE-1_page_001",
                    "doc_id": "SOURCE-1",
                    "title": "Two Page Source",
                    "page": "1",
                    "access_level": "unclassified",
                    "language": "en",
                },
                {
                    "source_id": "SOURCE-1_page_002",
                    "doc_id": "SOURCE-1",
                    "title": "Two Page Source",
                    "page": "2",
                    "access_level": "unclassified",
                    "language": "en",
                },
            ],
        },
        "generation": {"citation_mode": "cohere_native_accurate_default", "document_count": 2},
        "citations": [
            {
                "type": "TEXT_CONTENT",
                "start": 0,
                "end": 22,
                "text": "Two-page cited claim",
                "sources": [
                    {
                        "source_id": "SOURCE-1_page_001",
                        "doc_id": "SOURCE-1",
                        "title": "Two Page Source",
                        "page": "1",
                        "access_level": "unclassified",
                    },
                    {
                        "source_id": "SOURCE-1_page_002",
                        "doc_id": "SOURCE-1",
                        "title": "Two Page Source",
                        "page": "2",
                        "access_level": "unclassified",
                    },
                ],
            }
        ],
    }
    result = AgentTurnResult(
        session_id="multi_page_session",
        user_id="clearance_unclassified",
        persona_id="clearance_unclassified",
        answer="Two-page cited claim.",
        raw_answer="Two-page cited claim.",
        events_seen=0,
        citations=audit["citations"],
        citation_mode="cohere_native_accurate_default",
        documents_sent_to_model=2,
        answer_audit=audit,
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")

    def fake_preview(source, *, backend_persona_id):
        return SimpleNamespace(
            source_format="pdf",
            normalized_format="pdf",
            official_pdf_page_url="",
            original_docx_url="",
            page_image=b"",
            page=source.page,
            authorized=True,
            available=True,
            source_organization="",
            retrieved_date="",
            provenance_note="",
            page_text=f"page text {source.page}",
        )

    with patch("streamlit_app.resolve_source_preview", side_effect=fake_preview):
        payloads = streamlit_app._evidence_page_payloads(view_model)

    assert len(payloads) == 2
    assert "SOURCE-1:page:1:access:unclassified" in payloads
    assert "SOURCE-1:page:2:access:unclassified" in payloads
    assert payloads["SOURCE-1:page:1:access:unclassified"]["citation_ids"] == ["citation_1"]
    assert payloads["SOURCE-1:page:2:access:unclassified"]["citation_ids"] == ["citation_1"]
    assert "page text 1" in str(payloads["SOURCE-1:page:1:access:unclassified"]["html"])
    assert "page text 2" in str(payloads["SOURCE-1:page:2:access:unclassified"]["html"])
    assert payloads["SOURCE-1:page:1:access:unclassified"]["html"] != payloads["SOURCE-1:page:2:access:unclassified"]["html"]


def test_evidence_payloads_keep_citation_specific_html_on_shared_page() -> None:
    audit = {
        "session_id": "shared_page_session",
        "user_id": "clearance_unclassified",
        "persona_id": "clearance_unclassified",
        "query": "Summarize cited claims.",
        "retrieval": {
            "allowed_access": ["unclassified"],
            "sources_sent_to_answer": [
                {
                    "source_id": "SOURCE-1_page_021",
                    "doc_id": "SOURCE-1",
                    "title": "Shared Page Source",
                    "page": "21",
                    "access_level": "unclassified",
                    "language": "en",
                },
            ],
        },
        "generation": {"citation_mode": "cohere_native_accurate_default", "document_count": 1},
        "citations": [
            {
                "type": "TEXT_CONTENT",
                "start": 0,
                "end": 11,
                "text": "First claim",
                "sources": [{"source_id": "SOURCE-1_page_021", "doc_id": "SOURCE-1", "page": "21"}],
            },
            {
                "type": "TEXT_CONTENT",
                "start": 12,
                "end": 25,
                "text": "Second claim",
                "sources": [{"source_id": "SOURCE-1_page_021", "doc_id": "SOURCE-1", "page": "21"}],
            },
        ],
    }
    result = AgentTurnResult(
        session_id="shared_page_session",
        user_id="clearance_unclassified",
        persona_id="clearance_unclassified",
        answer="First claim. Second claim.",
        raw_answer="First claim. Second claim.",
        events_seen=0,
        citations=audit["citations"],
        citation_mode="cohere_native_accurate_default",
        documents_sent_to_model=1,
        answer_audit=audit,
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")

    def fake_preview(source, *, backend_persona_id):
        return SimpleNamespace(
            source_format="pdf",
            normalized_format="pdf",
            official_pdf_page_url="",
            original_docx_url="",
            page_image=b"",
            page=source.page,
            authorized=True,
            available=True,
            source_organization="",
            retrieved_date="",
            provenance_note="",
            page_text="shared page text",
        )

    with patch("streamlit_app.resolve_source_preview", side_effect=fake_preview):
        payloads = streamlit_app._evidence_page_payloads(view_model)

    payload = payloads["SOURCE-1:page:21:access:unclassified"]
    assert "[1] cited span: First claim" in payload["html"]
    assert "[1] cited span: First claim" in payload["html_by_citation_id"]["citation_1"]
    assert "[2] cited span: Second claim" in payload["html_by_citation_id"]["citation_2"]
    assert "[1] cited span: First claim" not in payload["html_by_citation_id"]["citation_2"]


def test_page_text_html_preserves_raw_line_breaks() -> None:
    raw_text = "CONTEXT\nA < B\n  indented line\n\n- bullet"

    rendered = streamlit_app._page_text_html(raw_text)

    assert "<pre>" in rendered
    assert "CONTEXT\nA &lt; B\n  indented line\n\n- bullet" in rendered
    assert "da-page-text-body" not in rendered
    assert "<p>" not in rendered


def test_citation_css_separates_hover_focus_from_selected_state() -> None:
    css = streamlit_app._app_css()

    hover_rule = re.search(
        r"\.da-inline-cite:hover,\s*"
        r"\.da-inline-cite:focus,\s*"
        r"\.da-citation-chip:hover,\s*"
        r"\.da-citation-chip:focus\s*\{(?P<body>[^}]+)\}",
        css,
    )
    active_rule = re.search(
        r"\.da-inline-cite\.da-active-citation,\s*"
        r"\.da-citation-chip\.da-active-citation\s*\{(?P<body>[^}]+)\}",
        css,
    )
    page_active_rule = re.search(
        r"\.da-list-button\.da-active-citation,\s*"
        r"\.da-list-button\.da-active-page\s*\{(?P<body>[^}]+)\}",
        css,
    )

    assert "--da-selected-border: #287e78" in css
    assert hover_rule
    assert "var(--da-accent)" in hover_rule.group("body")
    assert "var(--da-pale-green)" in hover_rule.group("body")
    assert "var(--da-green)" in hover_rule.group("body")
    assert active_rule
    assert "var(--da-selected-border)" in active_rule.group("body")
    assert "var(--da-aqua)" in active_rule.group("body")
    assert page_active_rule
    assert "var(--da-selected-border)" in page_active_rule.group("body")
    assert "var(--da-aqua)" in page_active_rule.group("body")
    assert ".da-inline-cite:hover,\n        .da-inline-cite:focus,\n        .da-inline-cite.da-active-citation" not in css


def test_citation_interaction_script_sets_semantic_and_page_specific_state() -> None:
    payloads = {
        "SOURCE-1:page:1:access:unclassified": {
            "page_key": "SOURCE-1:page:1:access:unclassified",
            "citation_ids": ["citation_1"],
            "primary_citation_id": "citation_1",
            "html": "<div>page one</div>",
            "html_by_citation_id": {"citation_1": "<div>page one citation one</div>"},
        },
        "SOURCE-1:page:2:access:unclassified": {
            "page_key": "SOURCE-1:page:2:access:unclassified",
            "citation_ids": ["citation_1"],
            "primary_citation_id": "citation_1",
            "html": "<div>page two</div>",
            "html_by_citation_id": {"citation_1": "<div>page two citation one</div>"},
        },
    }

    script = streamlit_app._citation_interaction_script(
        payloads,
        selected_page_key="SOURCE-1:page:1:access:unclassified",
        selected_citation_id="citation_1",
    )

    assert "target instanceof parentWin.Element" in script
    assert 'node.setAttribute("aria-current", "true")' in script
    assert 'node.removeAttribute("aria-current")' in script
    assert "const chipPageKey = node.dataset.citationPageKey ||" in script
    assert "&& (!chipPageKey || chipPageKey === pageKey)" in script
    assert "const htmlByCitationId = record.html_by_citation_id ||" in script
    assert "const selectedHtml = activeCitationId && htmlByCitationId[activeCitationId]" in script
    assert "panel.innerHTML = selectedHtml" in script
    assert "parentWin.__defenceAgentEvidenceRecordsForCitation" in script
    assert "parentWin.__defenceAgentActiveEvidence" in script
    assert "const activeRecord = records.find((record) => record.page_key === activePageKey)" in script
    assert "da-selected-evidence-stack" not in script
    assert "records.map((record) => record.page_key)" not in script
    assert 'anchor.scrollIntoView({ behavior: "auto", block: "start" })' in script
    assert "event.stopImmediatePropagation" in script
    assert "parentWin.__defenceAgentSelectPage(pageKey, citationId)" in script
    assert "parentWin.__defenceAgentSelectCitation(citationId)" in script
    assert "data-page-card-key" not in script


def test_demo_replay_sets_clear_trace_status_for_saved_runs() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    result = streamlit_app._load_demo_result(
        selected_example="Concise cited NATO answer",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Concise cited NATO answer"],
        target_answer_language="en",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        retrieval_mode="hybrid",
        chunk_strategy="page",
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")

    assert result.retrieval_status == "retrieval_complete"
    assert view_model.retrieval_status == "retrieval_complete"
    assert "unknown" not in {str(row["value"]).lower() for row in streamlit_app._request_rows(view_model)}


def test_sidebar_contains_demo_controls_without_moving_answer_workflow() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.run(timeout=20)

    assert not app.exception
    assert app.sidebar.radio[0].label == "Persona"
    assert app.sidebar.selectbox[0].label == "Demo query"
    assert len(app.sidebar.button) == 0
    assert app.sidebar.radio[1].label == "Run mode"
    assert app.sidebar.selectbox[1].label == "Autonomy mode"
    assert app.sidebar.selectbox[2].label == "Retrieval mode"
    assert app.sidebar.selectbox[3].label == "Chunk strategy"
    assert app.sidebar.checkbox[0].label == "Stream answer display"
    assert app.sidebar.checkbox[0].value is True
    sidebar_markdown = "\n".join(item.value for item in app.sidebar.markdown)
    assert "Allowed evidence" not in sidebar_markdown
    assert "Can retrieve and cite unclassified documents only." not in sidebar_markdown
    assert "Compare Canadian defence modernization sources with cited support." not in sidebar_markdown
    assert "Demo status" not in sidebar_markdown
    assert "Demo trajectory" not in sidebar_markdown
    assert "Cited answer spans" not in sidebar_markdown
    assert "Selected evidence" not in sidebar_markdown
    assert "citation-support scoring" not in sidebar_markdown
    assert "Model-facing tool" not in sidebar_markdown
    assert "Multi-agent over persona-scoped approved doctrine pages." not in sidebar_markdown
    assert "Only tool" not in sidebar_markdown
    assert "search_documents" not in sidebar_markdown
    assert "Access filter" not in sidebar_markdown
    assert "Embed v4" not in sidebar_markdown
    assert "Rerank v4" not in sidebar_markdown
    assert "Citation IDs" not in sidebar_markdown
    assert "run_turn" not in sidebar_markdown
    assert app.main.text_area[0].label == "Question"
    assert app.main.button[0].label == "Run query"


def test_answer_screen_exposes_follow_up_question_box() -> None:
    with patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess") as backend:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)
        app.main.button[0].click()
        app.run(timeout=20)

    assert not app.exception
    backend.assert_not_called()
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Follow-up question" in rendered
    assert app.main.text_area[0].label == "Follow-up question"
    assert all(text_area.label != "Question" for text_area in app.main.text_area)
    assert any(button.label == "Ask follow-up" for button in app.main.button)
    assert any(button.label == "Start new question" for button in app.main.button)


def test_access_boundary_result_uses_single_active_input_box() -> None:
    with patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess") as backend:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)
        app.sidebar.selectbox[0].set_value("Access boundary")
        app.run(timeout=20)
        app.main.button[0].click()
        app.run(timeout=20)

    assert not app.exception
    backend.assert_not_called()
    assert any(text_area.label == "Follow-up question" for text_area in app.main.text_area)
    assert all(text_area.label != "Question" for text_area in app.main.text_area)


def test_demo_query_selection_loads_prompt_text_without_button() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.run(timeout=20)
    app.sidebar.selectbox[0].set_value("Concise cited NATO answer")
    app.run(timeout=20)

    assert not app.exception
    assert app.main.text_area[0].value == streamlit_app.EXAMPLE_PROMPTS["Concise cited NATO answer"]


def test_route_preview_explains_auto_only_priority_controls() -> None:
    locked = streamlit_app._route_preview_caption(
        query="Compare two policy sources and cite pages.",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        accuracy_priority=5,
        latency_priority=1,
    )
    auto = streamlit_app._route_preview_caption(
        query="Compare two policy sources and cite pages.",
        autonomy_mode=streamlit_app.AUTO,
        accuracy_priority=5,
        latency_priority=1,
    )

    assert "Route locked" in locked
    assert "Auto priority sliders are not applied" in locked
    assert "Auto preview: Multi-agent" in auto


def test_demo_replay_registry_covers_visible_persona_examples() -> None:
    for persona_id in streamlit_app.UI_PERSONAS:
        for example, query in streamlit_app.DEMO_QUERIES.items():
            if query.live_only:
                assert (
                    streamlit_app._demo_replay(
                        ui_persona_id=persona_id,
                        selected_example=example,
                        autonomy_mode=streamlit_app.REVIEWED_AGENT,
                        retrieval_mode="hybrid",
                        chunk_strategy="page",
                    )
                    is None
                )
            else:
                assert (
                    streamlit_app._demo_replay(
                        ui_persona_id=persona_id,
                        selected_example=example,
                        autonomy_mode=streamlit_app.REVIEWED_AGENT,
                        retrieval_mode="hybrid",
                        chunk_strategy="page",
                    )
                    is not None
                )


def test_planning_option_uses_two_search_trace_candidate() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    replay = streamlit_app.DEMO_REPLAYS[
        streamlit_app._replay_key("persona_a", "Planning brief comparison", streamlit_app.REVIEWED_AGENT)
    ]
    result = streamlit_app._load_demo_result(
        selected_example="Planning brief comparison",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Planning brief comparison"],
        target_answer_language="en",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        retrieval_mode="hybrid",
        chunk_strategy="page",
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")
    queries = [call.query for call in view_model.tool_calls]
    budget = streamlit_app._context_budget(view_model)

    assert replay.transcript_dir == streamlit_app.ROUTE_REVIEWED_TRANSCRIPT_DIR
    assert replay.transcript_name == "readiness_hybrid_multidoc_modernization.json"
    assert result.answer_audit["retrieval"]["search_count"] == 2
    assert result.tool_calls == ["search_documents", "search_documents"]
    assert queries == [
        "Canada's defence policy AI-enabled modernization",
        "DND/CAF AI Strategy AI-enabled modernization",
    ]
    assert len(set(queries)) == 2
    assert view_model.citations
    assert budget["method"] == "guided_replay_page_estimate"
    assert budget["prompt_tokens_estimate"] > 0
    assert budget["context_window_used_pct"] > 0
    search_steps = [step for step in view_model.runtime_steps if step.title.startswith("Tool call")]
    assert len(search_steps) == 2
    assert search_steps[0].code == (
        'search_documents(query="Canada\'s defence policy AI-enabled modernization", top_k=8)'
    )
    assert search_steps[1].code == (
        'search_documents(query="DND/CAF AI Strategy AI-enabled modernization", top_k=8)'
    )
    for step in search_steps:
        assert ("Authorized pages", "8") in step.metrics
        assert any(label == "Top rerank" and value for label, value in step.metrics)
        assert any(label == "Top vector" and value for label, value in step.metrics)


def test_trace_generation_label_distinguishes_reasoning_model() -> None:
    view_model = SimpleNamespace(answer_audit={"generation": {"model": "command-a-reasoning-08-2025"}})

    assert streamlit_app._generation_model_label(view_model) == "Command A Reasoning"


def test_refusal_trace_details_distinguish_access_boundary_from_evidence_gap() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    access_result = streamlit_app._load_demo_result(
        selected_example="Access boundary",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Access boundary"],
        target_answer_language="en",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        retrieval_mode="hybrid",
        chunk_strategy="page",
    )
    access_view = build_view_model(access_result, ui_persona_id="persona_a")

    gap_result = streamlit_app._load_demo_result(
        selected_example="Evidence gap: Arctic basing 2031",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Evidence gap: Arctic basing 2031"],
        target_answer_language="en",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        retrieval_mode="hybrid",
        chunk_strategy="page",
    )
    gap_view = build_view_model(gap_result, ui_persona_id="persona_a")

    assert "denied source group" in streamlit_app._answerability_detail(access_view)
    access_html = streamlit_app._runtime_flow_html(access_view)
    assert "da-runtime-status-dot--denied" in access_html
    assert "da-runtime-code" in access_html
    assert "Evidence check" in access_html
    assert "Zero-doc generation path" in access_html
    assert "Vector match" not in access_html
    assert "Rerank relevance" not in access_html
    assert "Chroma" not in access_html
    assert "Status filter" not in access_html
    assert "Language any" not in access_html
    assert "Excluded pages" not in access_html
    assert "authorized_evidence_sent" not in access_html
    assert "confidence" not in access_html.lower()
    assert "Fusion Model Release Control Procedure" not in access_html
    assert "Database" not in access_html
    assert "COMPLETE" not in access_html
    assert "Citations</span>" not in access_html
    assert "Audit</span>" not in access_html
    gap_detail = streamlit_app._answerability_detail(gap_view)
    gap_html = streamlit_app._runtime_flow_html(gap_view)
    assert "2031" in gap_detail
    assert "basing" in gap_detail
    assert "Evidence check" in gap_html
    assert "reviewed 8 authorized page(s)" in gap_html
    assert "COMPLETE" not in gap_html
    assert gap_view.documents_sent_to_model == 8


def test_bilingual_example_prompt_matches_saved_replay_topic() -> None:
    prompt = streamlit_app.EXAMPLE_PROMPTS["French NATO doctrine answer"].lower()
    query = streamlit_app.DEMO_QUERIES["French NATO doctrine answer"]

    assert "otan" in prompt
    assert "taches fondamentales" in prompt
    assert query.target_answer_language == "fr"
    assert query.live_only is False
    assert (
        streamlit_app._demo_replay(
            ui_persona_id="persona_a",
            selected_example="French NATO doctrine answer",
            autonomy_mode=streamlit_app.REVIEWED_AGENT,
            retrieval_mode="hybrid",
            chunk_strategy="page",
        )
        is not None
    )
    assert streamlit_app._target_answer_language(streamlit_app.EXAMPLE_PROMPTS["French NATO doctrine answer"], "French NATO doctrine answer") == "fr"
    assert streamlit_app._can_use_guided_replay("demo", "French NATO doctrine answer", streamlit_app.EXAMPLE_PROMPTS["French NATO doctrine answer"]) is True


def test_persona_b_demo_replays_build_without_generic_value_error() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_b")
    for example in ("Concise cited NATO answer", "Access boundary"):
        result = streamlit_app._load_demo_result(
            selected_example=example,
            persona=persona,
            query=streamlit_app.EXAMPLE_PROMPTS[example],
            target_answer_language="en",
            autonomy_mode=streamlit_app.REVIEWED_AGENT,
            retrieval_mode="hybrid",
            chunk_strategy="page",
        )
        view_model = build_view_model(result, ui_persona_id="persona_b")
        assert view_model.backend_persona_id == "clearance_top_secret"
        assert view_model.citations
    assert set(view_model.allowed_access) == {"unclassified", "secret", "top_secret"}


def test_docx_origin_demo_replay_covers_normalized_source() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    result = streamlit_app._load_demo_result(
        selected_example="DOCX-origin ASOEM responsibilities",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["DOCX-origin ASOEM responsibilities"],
        target_answer_language="en",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        retrieval_mode="hybrid",
        chunk_strategy="page",
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")

    assert "safe operating environment" in result.answer.lower()
    assert {source.doc_id for source in view_model.sources.values()} == {"UK-MOD-ASOEM-2023-EN"}
    assert view_model.citations


def test_reviewer_gate_demo_replay_surfaces_human_review() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    result = streamlit_app._load_demo_result(
        selected_example="Reviewer catches unsupported claims",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Reviewer catches unsupported claims"],
        target_answer_language="en",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        retrieval_mode="hybrid",
        chunk_strategy="page",
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")

    assert view_model.critic["status"] == "needs_human_review"
    assert view_model.critic["credibility_score"] == 0.5
    assert view_model.critic["release_gate"] == "human_continue_or_stop_required"
    assert len(view_model.tool_calls) >= 2
    runtime_html = streamlit_app._runtime_flow_html(view_model)
    trace_html = streamlit_app._trace_timeline_html(view_model)
    quality_html = streamlit_app._quality_gate_html(view_model)
    trust_html = streamlit_app._reviewer_value_html(view_model)
    iteration_html = streamlit_app._review_iteration_html(view_model)
    answer_html = streamlit_app._answer_html_with_inline_citations(view_model)
    reviewer_rows = streamlit_app._reviewer_citation_review_rows(view_model.critic)

    assert "Reviewer sub-agent trust check" in runtime_html
    assert "Escalation path" in runtime_html
    assert "Reviewer sub-agent trust check" in trace_html
    assert "Escalation path" in trace_html
    assert "verified citations: 4" in trace_html
    assert "unverified citations: 4" in trace_html
    assert "top rerank score" in trace_html
    assert "Threshold" in runtime_html
    assert "Unverified" in runtime_html
    assert "human_continue_or_stop_required" in runtime_html
    assert "Trust score" in quality_html
    assert "System action" in quality_html
    assert "Human review" in quality_html
    assert "Reviewer trust layer" in trust_html
    assert "Low-trust answer needs human review" in trust_html
    assert "Verified" in trust_html
    assert "Needs review" in trust_html
    assert "Low-trust" in trust_html
    assert "Not checked" in trust_html
    assert "out of trusted support" in trust_html
    assert "Reviewer iteration loop" in iteration_html
    assert "Iteration 1" in iteration_html
    assert "Iteration 2" in iteration_html
    assert "Final generation" in iteration_html
    assert "da-citation-trust--unverified" in answer_html
    assert 'data-citation-trust="unverified"' in answer_html
    assert "Trust: low trust" in answer_html
    assert any(row["verdict"] == "unverified" for row in reviewer_rows)
    assert any("NORAD modernization" in row["answer_span"] for row in reviewer_rows)


def test_reviewer_improvement_demo_surfaces_feedback_loop() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    result = streamlit_app._load_demo_result(
        selected_example="Reviewer improves weak answer",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Reviewer improves weak answer"],
        target_answer_language="en",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        retrieval_mode="hybrid",
        chunk_strategy="page",
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")
    cycle_rows = streamlit_app._review_cycle_rows(view_model)
    iteration_html = streamlit_app._review_iteration_html(view_model)
    runtime_html = streamlit_app._runtime_flow_html(view_model)

    assert view_model.critic["status"] == "approved"
    assert view_model.critic["credibility_score"] == 0.875
    assert view_model.critic["credibility_score"] >= view_model.critic["threshold"]
    assert view_model.answer_audit["review_control"]["completed_review_cycles"] == 2
    assert cycle_rows[0]["reviewer_status"] == "needs_revision"
    assert cycle_rows[0]["feedback_to_research_agent"] == "yes"
    assert "Revise the answer" in cycle_rows[0]["feedback"]
    assert "weak_citations" in view_model.answer_audit["review_control"]["cycles"][0]
    assert cycle_rows[1]["reviewer_status"] == "approved"
    assert "Review cycle 1" in runtime_html
    assert "sent to Research sub-agent" in runtime_html
    assert "Review cycle 2" in runtime_html
    assert "Reviewer iteration loop" in iteration_html
    assert "Iteration 1" in iteration_html
    assert "Iteration 2" in iteration_html
    assert "Final generation" in iteration_html
    assert "The final answer is visible with caveats" in iteration_html
    assert "not checked" in iteration_html


def test_rerun_query_includes_reviewer_feedback_instruction() -> None:
    query = streamlit_app._rerun_query_with_reviewer_feedback(
        "Original query.",
        "Remove unsupported C5ISR claims.",
    )

    assert "Original query." in query
    assert "Remove unsupported C5ISR claims." in query
    assert "Return only claims directly supported" in query
    assert "Remove unsupported citations" in query


def test_citation_trace_card_can_show_one_span_linked_to_multiple_pages() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    result = streamlit_app._load_demo_result(
        selected_example="Concise cited NATO answer",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Concise cited NATO answer"],
        target_answer_language="en",
        autonomy_mode=streamlit_app.REVIEWED_AGENT,
        retrieval_mode="hybrid",
        chunk_strategy="page",
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")

    assert len(view_model.citations[0].source_ids) > 1
    rendered = streamlit_app._citation_trace_card_html(view_model, view_model.citations[0])
    assert "Linked source pages" in rendered
    assert "NATO 2022 Strategic Concept" in rendered


def test_missing_demo_replay_uses_clear_user_facing_error() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    with patch.dict(streamlit_app.DEMO_REPLAYS, {}, clear=True):
        try:
            streamlit_app._load_demo_result(
                selected_example="Concise cited NATO answer",
                persona=persona,
                query="demo",
                target_answer_language="en",
                autonomy_mode=streamlit_app.REVIEWED_AGENT,
                retrieval_mode="hybrid",
                chunk_strategy="page",
            )
        except Exception as exc:  # noqa: BLE001 - verifying UI-facing sanitizer.
            message = streamlit_app._friendly_run_error(exc)
        else:
            raise AssertionError("Expected missing replay to fail")

    assert message == "No guided run exists for this persona and query."


def test_demo_replay_acl_validation_checks_citation_sources() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    audit = {
        "retrieval": {
            "authorized_sources": [
                {
                    "source_id": "PUBLIC_page_001",
                    "doc_id": "PUBLIC",
                    "page": "1",
                    "access_level": "unclassified",
                }
            ],
            "sources_sent_to_answer": [
                {
                    "source_id": "PUBLIC_page_001",
                    "doc_id": "PUBLIC",
                    "page": "1",
                    "access_level": "unclassified",
                }
            ],
        },
        "citations": [
            {
                "text": "restricted claim",
                "sources": [
                    {
                        "source_id": "SECRET_page_001",
                        "doc_id": "SECRET",
                        "page": "1",
                        "access_level": "secret",
                    }
                ],
            }
        ],
    }

    with pytest.raises(streamlit_app.DemoReplayUnavailable, match="outside the selected persona"):
        streamlit_app._validate_replay_acl(audit, persona=persona)


def test_inline_citation_links_are_stable_and_do_not_match_button_text() -> None:
    href = streamlit_app._citation_href("citation_15")
    script = streamlit_app._scroll_to_selected_evidence_script()

    assert href == "#selected-evidence-anchor"
    assert "querySelectorAll(\"button\")" not in script
    assert "selected-evidence-anchor" in script
    assert 'scrollIntoView({ behavior: "auto", block: "start" })' in script
    assert "setTimeout" in script


def test_streamed_answer_plain_html_is_escaped() -> None:
    rendered = streamlit_app._plain_answer_html("<unsafe> & answer")

    assert "&lt;unsafe&gt; &amp; answer" in rendered
    assert "da-answer-streaming" in rendered


def test_scroll_to_selected_evidence_retries_until_anchor_exists() -> None:
    script = streamlit_app._scroll_to_selected_evidence_script()

    assert "selected-evidence-anchor" in script
    assert "setTimeout" in script
    assert 'scrollIntoView({ behavior: "auto", block: "start" })' in script


def test_auto_scroll_run_progress_tracks_running_status_widget() -> None:
    script = streamlit_app._auto_scroll_run_progress_script()

    assert 'data-testid="stStatusWidget"' in script
    assert 'scrollIntoView({ behavior: "smooth", block: "end" })' in script
    assert "documentElement.scrollHeight" in script


def test_live_mode_missing_key_has_specific_error() -> None:
    message = streamlit_app._friendly_run_error(RuntimeError("COHERE_API_KEY is missing for live mode."))

    assert message == "COHERE_API_KEY is missing for live mode."
    assert "ValueError" not in message


@pytest.mark.parametrize(
    ("raw_error", "expected"),
    [
        ("Too many requests: 429", "Cohere rate limit hit"),
        ("Unauthorized: invalid API key", "Cohere authentication failed"),
        ("RemoteProtocolError: peer closed connection", "Cohere transport error"),
        ("chromadb.errors.NotFoundError: no such collection", "Local retrieval index failed"),
    ],
)
def test_backend_live_errors_are_specific_and_do_not_switch_to_replay(raw_error: str, expected: str) -> None:
    message = streamlit_app._friendly_run_error(streamlit_app.backend_bridge.BackendRunError(raw_error))

    assert expected in message
    assert "No transcript fallback was used" in message
    assert raw_error not in message
