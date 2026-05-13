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


def test_streamlit_demo_replay_renders_single_column_result_without_backend_call() -> None:
    with patch("defence_agent.ui.backend_bridge.run_turn_in_subprocess") as backend:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=20)
        app.main.button[0].click()
        app.run(timeout=20)

    assert not app.exception
    backend.assert_not_called()
    rendered_markdown = "\n".join(item.value for item in app.markdown)
    assert "AI will be foundational to Defence modernization" in rendered_markdown
    assert "da-inline-cite" in rendered_markdown
    assert 'href="#selected-evidence-anchor"' in rendered_markdown
    assert 'target="_self"' in rendered_markdown
    assert 'data-citation-id="citation_1"' in rendered_markdown
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
    assert app.sidebar.button[0].label == "Load query"
    assert app.sidebar.radio[1].label == "Run mode"


def test_streamlit_renders_four_tabs() -> None:
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
    assert any(metric.label == "Visible docs" and metric.value == "7" for metric in app.metric)
    assert any(metric.label == "Visible pages" and metric.value == "201" for metric in app.metric)
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

    assert 'class="da-inline-cite"' in rendered
    assert 'href="#selected-evidence-anchor"' in rendered
    assert 'target="_self"' in rendered
    assert 'data-citation-id="citation_1"' in rendered
    assert 'data-citation-marker="[1]"' in rendered
    assert "data-tooltip=" in rendered
    assert "NATO 2022 Strategic Concept" in rendered
    assert "p.4" in rendered
    assert "[1]" in rendered
    assert "[C1" not in rendered


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


def test_demo_replay_sets_clear_trace_status_for_saved_runs() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    result = streamlit_app._load_demo_result(
        selected_example="Concise cited NATO answer",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Concise cited NATO answer"],
        target_answer_language="en",
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")

    assert result.retrieval_status == "option ready"
    assert view_model.retrieval_status == "option ready"
    assert "unknown" not in {str(row["value"]).lower() for row in streamlit_app._request_rows(view_model)}


def test_sidebar_contains_demo_controls_without_moving_answer_workflow() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.run(timeout=20)

    assert not app.exception
    assert app.sidebar.radio[0].label == "Persona"
    assert app.sidebar.selectbox[0].label == "Demo query"
    assert app.sidebar.button[0].label == "Load query"
    assert app.sidebar.radio[1].label == "Run mode"
    assert app.sidebar.checkbox[0].label == "Stream answer display"
    assert app.sidebar.checkbox[0].value is True
    sidebar_markdown = "\n".join(item.value for item in app.sidebar.markdown)
    assert "Demo status" not in sidebar_markdown
    assert "Demo trajectory" not in sidebar_markdown
    assert "Cited answer spans" not in sidebar_markdown
    assert "Selected evidence" not in sidebar_markdown
    assert app.main.text_area[0].label == "Question"
    assert app.main.button[0].label == "Run query"


def test_demo_query_selection_loads_prompt_text_without_button() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.run(timeout=20)
    app.sidebar.selectbox[0].set_value("Concise cited NATO answer")
    app.run(timeout=20)

    assert not app.exception
    assert app.main.text_area[0].value == streamlit_app.EXAMPLE_PROMPTS["Concise cited NATO answer"]


def test_demo_replay_registry_covers_visible_persona_examples() -> None:
    for persona_id in streamlit_app.UI_PERSONAS:
        for example, query in streamlit_app.DEMO_QUERIES.items():
            if query.live_only:
                assert (persona_id, example) not in streamlit_app.DEMO_REPLAYS
            else:
                assert (persona_id, example) in streamlit_app.DEMO_REPLAYS


def test_planning_option_uses_two_search_trace_candidate() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    replay = streamlit_app.DEMO_REPLAYS[("persona_a", "Planning brief comparison")]
    result = streamlit_app._load_demo_result(
        selected_example="Planning brief comparison",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Planning brief comparison"],
        target_answer_language="en",
    )
    view_model = build_view_model(result, ui_persona_id="persona_a")
    queries = [call.query for call in view_model.tool_calls]

    assert replay.transcript_dir == streamlit_app.PLANNING_TRANSCRIPT_DIR
    assert replay.transcript_name == "flagship_planning_brief_modernization.json"
    assert result.answer_audit["retrieval"]["search_count"] == 2
    assert result.tool_calls == ["search_documents", "search_documents"]
    assert queries == [
        "Canada's defence policy AI-enabled modernization",
        "DND/CAF AI Strategy AI-enabled modernization",
    ]
    assert len(set(queries)) == 2
    assert view_model.citations


def test_refusal_trace_details_distinguish_access_boundary_from_evidence_gap() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    access_result = streamlit_app._load_demo_result(
        selected_example="Access boundary",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Access boundary"],
        target_answer_language="en",
    )
    access_view = build_view_model(access_result, ui_persona_id="persona_a")

    gap_result = streamlit_app._load_demo_result(
        selected_example="Evidence gap: Arctic basing 2031",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Evidence gap: Arctic basing 2031"],
        target_answer_language="en",
    )
    gap_view = build_view_model(gap_result, ui_persona_id="persona_a")

    assert "denied source group" in streamlit_app._answerability_detail(access_view)
    gap_detail = streamlit_app._answerability_detail(gap_view)
    assert "2031" in gap_detail
    assert "basing" in gap_detail
    assert gap_view.documents_sent_to_model == 0


def test_bilingual_example_prompt_matches_saved_replay_topic() -> None:
    prompt = streamlit_app.EXAMPLE_PROMPTS["French NATO doctrine answer"].lower()
    query = streamlit_app.DEMO_QUERIES["French NATO doctrine answer"]

    assert "otan" in prompt
    assert "taches fondamentales" in prompt
    assert query.target_answer_language == "fr"
    assert query.live_only is True
    assert ("persona_a", "French NATO doctrine answer") not in streamlit_app.DEMO_REPLAYS
    assert streamlit_app._target_answer_language(streamlit_app.EXAMPLE_PROMPTS["French NATO doctrine answer"], "French NATO doctrine answer") == "fr"
    assert streamlit_app._can_use_guided_replay("demo", "French NATO doctrine answer", streamlit_app.EXAMPLE_PROMPTS["French NATO doctrine answer"]) is False


def test_persona_b_demo_replays_build_without_generic_value_error() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_b")
    for example in ("Concise cited NATO answer", "Access boundary"):
        result = streamlit_app._load_demo_result(
            selected_example=example,
            persona=persona,
            query=streamlit_app.EXAMPLE_PROMPTS[example],
            target_answer_language="en",
        )
        view_model = build_view_model(result, ui_persona_id="persona_b")
        assert view_model.backend_persona_id == "clearance_top_secret"
        assert view_model.citations
    assert set(view_model.allowed_access) == {"unclassified", "secret", "top_secret"}


def test_citation_trace_card_can_show_one_span_linked_to_multiple_pages() -> None:
    persona = streamlit_app.persona_for_ui_id("persona_a")
    result = streamlit_app._load_demo_result(
        selected_example="Concise cited NATO answer",
        persona=persona,
        query=streamlit_app.EXAMPLE_PROMPTS["Concise cited NATO answer"],
        target_answer_language="en",
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
    assert 'scrollIntoView({ behavior: "smooth", block: "start" })' in script
    assert "setTimeout" in script


def test_streamed_answer_plain_html_is_escaped() -> None:
    rendered = streamlit_app._plain_answer_html("<unsafe> & answer")

    assert "&lt;unsafe&gt; &amp; answer" in rendered
    assert "da-answer-streaming" in rendered


def test_scroll_to_selected_evidence_retries_until_anchor_exists() -> None:
    script = streamlit_app._scroll_to_selected_evidence_script()

    assert "selected-evidence-anchor" in script
    assert "setTimeout" in script
    assert 'scrollIntoView({ behavior: "smooth", block: "start" })' in script


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
