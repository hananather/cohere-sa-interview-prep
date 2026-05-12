from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from defence_agent.session import SessionPersonaMismatchError
from defence_agent.session import AgentTurnResult
from defence_agent.session import create_session_service
from defence_agent.session import ensure_session
from defence_agent.ui.backend_bridge import _result_from_stdout, _safe_error_detail, agent_turn_result_from_dict
from defence_agent.ui.backend_bridge import run_turn_in_subprocess
from defence_agent.ui.eval_results import (
    INSUFFICIENT_EVIDENCE_READINESS_RUN,
    PREFERRED_READINESS_RUN,
    filter_eval_rows,
    load_eval_rows,
    preferred_transcript_dir,
    transcript_run_options,
    transcript_run_label,
)
from defence_agent.ui.source_preview import resolve_source_preview
from defence_agent.ui.view_model import SourceView, build_view_model, citation_chip_label, persona_for_ui_id


ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = ROOT / "defence_agent" / "data" / "transcripts" / "live_readiness_final_20260511_013937"


def _result_from_transcript(path: Path) -> AgentTurnResult:
    data = json.loads(path.read_text(encoding="utf-8"))
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


def test_persona_b_maps_to_top_secret_backend_clearance() -> None:
    persona = persona_for_ui_id("persona_b")

    assert persona.backend_persona_id == "clearance_top_secret"
    assert persona.allowed_access == ("unclassified", "secret", "top_secret")
    assert "restricted corpus" in persona.visible_access_label


def test_view_model_uses_answer_audit_citations() -> None:
    result = replace(
        _result_from_transcript(TRANSCRIPTS / "acl_secret_sensor_fusion_release_rule.json"),
        persona_id="clearance_top_secret",
        user_id="clearance_top_secret",
    )
    view_model = build_view_model(result, ui_persona_id="persona_b")

    assert view_model.persona_label == "Persona B: Cleared User"
    assert view_model.allowed_access == ("unclassified", "secret", "top_secret")
    assert view_model.citations
    assert view_model.citations[0].answer_text
    assert view_model.citations[0].source_ids

    source = view_model.sources[view_model.citations[0].source_ids[0]]
    assert source.used_in_answer is True
    assert source.sent_to_model is True
    assert source.access_level == "secret"
    assert citation_chip_label(view_model.citations[0], view_model.sources).startswith("[1]")
    assert "C1" not in citation_chip_label(view_model.citations[0], view_model.sources)


def test_view_model_builds_display_citations_and_deduped_source_pages() -> None:
    audit = {
        "query": "demo",
        "session_id": "s_demo",
        "user_id": "clearance_unclassified",
        "persona_id": "clearance_unclassified",
        "retrieval": {
            "allowed_access": ["unclassified"],
            "authorized_sources": [
                {
                    "source_id": "NATO-STRAT-CONCEPT-2022-EN_page_004",
                    "chunk_id": "NATO-STRAT-CONCEPT-2022-EN_page_004",
                    "doc_id": "NATO-STRAT-CONCEPT-2022-EN",
                    "title": "NATO 2022 Strategic Concept",
                    "page": "4",
                    "access_level": "unclassified",
                    "language": "en",
                }
            ],
            "sources_sent_to_answer": [
                {
                    "source_id": "NATO-STRAT-CONCEPT-2022-EN_page_004",
                    "chunk_id": "NATO-STRAT-CONCEPT-2022-EN_page_004",
                    "doc_id": "NATO-STRAT-CONCEPT-2022-EN",
                    "title": "NATO 2022 Strategic Concept",
                    "page": "4",
                    "access_level": "unclassified",
                    "language": "en",
                }
            ],
        },
        "generation": {"document_count": 1, "citation_mode": "cohere_native_accurate_default"},
        "citations": [
            {
                "type": "cohere_native",
                "start": 0,
                "end": 23,
                "text": "Deterrence and defence",
                "sources": [
                    {
                        "source_id": "NATO-STRAT-CONCEPT-2022-EN_page_004",
                        "doc_id": "NATO-STRAT-CONCEPT-2022-EN",
                        "title": "NATO 2022 Strategic Concept",
                        "page": "4",
                        "access_level": "unclassified",
                        "language": "en",
                    }
                ],
            },
            {
                "type": "cohere_native",
                "start": 25,
                "end": 58,
                "text": "crisis prevention and management",
                "sources": [
                    {
                        "source_id": "NATO-STRAT-CONCEPT-2022-EN_page_004",
                        "doc_id": "NATO-STRAT-CONCEPT-2022-EN",
                        "title": "NATO 2022 Strategic Concept",
                        "page": "4",
                        "access_level": "unclassified",
                        "language": "en",
                    }
                ],
            },
        ],
    }
    result = AgentTurnResult(
        session_id="s_demo",
        user_id="clearance_unclassified",
        persona_id="clearance_unclassified",
        answer="Deterrence and defence [C1] and crisis prevention and management [C1].",
        raw_answer="Deterrence and defence [C1] and crisis prevention and management [C1].",
        events_seen=0,
        citations=audit["citations"],
        citation_mode="cohere_native_accurate_default",
        documents_sent_to_model=1,
        answer_audit=audit,
    )

    view_model = build_view_model(result, ui_persona_id="persona_a")

    assert view_model.display_answer == "Deterrence and defence and crisis prevention and management."
    assert len(view_model.citations) == 2
    assert citation_chip_label(view_model.citations[0], view_model.sources) == "[1] Deterrence and defence"
    assert "C1" not in citation_chip_label(view_model.citations[1], view_model.sources)
    assert len(view_model.evidence_pages) == 1
    assert view_model.evidence_pages[0].label == "NATO 2022 Strategic Concept · p.4 · unclassified"


def test_view_model_sanitizes_excluded_source_metadata() -> None:
    result = _result_from_transcript(TRANSCRIPTS / "acl_unclassified_sensor_fusion_release_rule.json")
    view_model = build_view_model(result, ui_persona_id="persona_a")

    assert view_model.answerability == "REFUSED"
    assert view_model.excluded_source_summary
    assert view_model.excluded_source_summary[0]["access_level"] == "secret"
    assert "title" not in view_model.excluded_source_summary[0]
    assert "doc_id" not in view_model.excluded_source_summary[0]
    sanitized = json.dumps(view_model.sanitized_answer_audit)
    assert "SYN-FUSION-S-RELEASE-001" not in sanitized
    assert "Fusion Model Release Control Procedure" not in sanitized
    assert "source_pdf_path" not in sanitized


def test_view_model_rejects_backend_ui_persona_mismatch() -> None:
    result = replace(
        _result_from_transcript(TRANSCRIPTS / "acl_secret_sensor_fusion_release_rule.json"),
        persona_id="clearance_top_secret",
        user_id="clearance_top_secret",
    )

    with pytest.raises(ValueError, match="Backend persona"):
        build_view_model(result, ui_persona_id="persona_a")


def test_source_preview_rechecks_persona_acl_against_manifest() -> None:
    source = SourceView(
        source_id="SYN-FUSION-S-RELEASE-001_page_001",
        doc_id="SYN-FUSION-S-RELEASE-001",
        title="Fusion Model Release Control Procedure",
        page="1",
        access_level="secret",
        sent_to_model=True,
        used_in_answer=True,
    )

    denied = resolve_source_preview(source, backend_persona_id="clearance_unclassified")
    allowed = resolve_source_preview(source, backend_persona_id="clearance_top_secret")

    assert denied.authorized is False
    assert denied.pdf_path is None
    assert allowed.authorized is True
    assert allowed.available is True
    assert allowed.pdf_path is not None
    assert allowed.pdf_path.name == "fusion_model_release_control.pdf"
    assert allowed.page_text


def test_source_preview_exposes_public_lineage_and_page_image() -> None:
    source = SourceView(
        source_id="NATO-STRAT-CONCEPT-2022-EN_page_004",
        doc_id="NATO-STRAT-CONCEPT-2022-EN",
        title="NATO 2022 Strategic Concept",
        page="4",
        access_level="unclassified",
        sent_to_model=True,
        used_in_answer=True,
    )

    preview = resolve_source_preview(source, backend_persona_id="clearance_unclassified")

    assert preview.authorized is True
    assert preview.available is True
    assert preview.official_pdf_page_url.endswith("#page=4")
    assert "nato.int" in preview.publisher_url
    assert preview.page_image
    assert preview.page_text


def test_source_preview_exposes_docx_origin_lineage_without_native_docx_claim() -> None:
    source = SourceView(
        source_id="UK-MOD-ASOEM-2023-EN_page_001",
        doc_id="UK-MOD-ASOEM-2023-EN",
        title="Aviation Safe Operating Environment Manual",
        page="1",
        access_level="unclassified",
        sent_to_model=True,
        used_in_answer=True,
    )

    preview = resolve_source_preview(source, backend_persona_id="clearance_unclassified")

    assert preview.authorized is True
    assert preview.source_format == "docx"
    assert preview.normalized_format == "pdf"
    assert preview.original_docx_url.endswith(".docx")
    assert preview.official_pdf_page_url.endswith("#page=1")
    assert "normalized page-evidence pipeline" in preview.provenance_note


@pytest.mark.slow_offline
def test_source_preview_renders_every_manifest_document_when_authorized() -> None:
    corpus_dir = ROOT / "defence_agent" / "data" / "corpus"
    manifest = yaml.safe_load((corpus_dir / "manifest.yaml").read_text(encoding="utf-8"))

    for entry in manifest["documents"]:
        access = entry.get("access_level", "")
        persona_id = "clearance_top_secret" if access in {"secret", "top_secret"} else "clearance_unclassified"
        source = SourceView(
            source_id=f"{entry['doc_id']}_page_001",
            doc_id=entry["doc_id"],
            title=entry.get("title", ""),
            page="1",
            access_level=access,
            sent_to_model=True,
            used_in_answer=True,
        )

        preview = resolve_source_preview(source, backend_persona_id=persona_id, corpus_dir=corpus_dir)

        assert preview.authorized is True, entry["doc_id"]
        assert preview.available is True, entry["doc_id"]
        assert preview.page_image, entry["doc_id"]
        assert preview.page_text, entry["doc_id"]


def test_source_preview_blocks_restricted_images_and_text_for_persona_a() -> None:
    corpus_dir = ROOT / "defence_agent" / "data" / "corpus"
    for doc_id, access_level in (
        ("SYN-FUSION-S-RELEASE-001", "secret"),
        ("SYN-FUSION-TS-ANNEX-002", "top_secret"),
    ):
        source = SourceView(
            source_id=f"{doc_id}_page_001",
            doc_id=doc_id,
            title=doc_id,
            page="1",
            access_level=access_level,
            sent_to_model=True,
            used_in_answer=True,
        )

        preview = resolve_source_preview(source, backend_persona_id="clearance_unclassified", corpus_dir=corpus_dir)

        assert preview.authorized is False
        assert preview.available is False
        assert preview.page_image is None
        assert preview.page_text == ""


def test_source_preview_denies_unused_source_even_when_persona_is_authorized() -> None:
    source = SourceView(
        source_id="SYN-FUSION-S-RELEASE-001_page_001",
        doc_id="SYN-FUSION-S-RELEASE-001",
        title="Fusion Model Release Control Procedure",
        page="1",
        access_level="secret",
    )

    preview = resolve_source_preview(source, backend_persona_id="clearance_top_secret")

    assert preview.authorized is False
    assert preview.reason == "source_not_used_in_answer"


def test_session_persona_mismatch_fails_closed() -> None:
    async def run() -> None:
        service = create_session_service(persistent=False)
        session_id = await ensure_session(
            service,
            user_id="demo_user",
            persona_id="clearance_unclassified",
            session_id="s_persona_guardrail",
        )
        with pytest.raises(SessionPersonaMismatchError):
            await ensure_session(
                service,
                user_id="demo_user",
                persona_id="clearance_top_secret",
                session_id=session_id,
            )

    asyncio.run(run())


def test_eval_rows_load_and_filter_saved_transcripts() -> None:
    rows = load_eval_rows(TRANSCRIPTS)
    acl_rows = filter_eval_rows(rows, "ACL")
    unclassified_acl = next(row for row in rows if row["case_id"] == "acl_unclassified_sensor_fusion_release_rule")
    cited_lookup = next(row for row in rows if row["case_id"] == "natural_multilingual_nato_core_tasks")

    assert rows
    assert any(row["case_id"] == "flagship_planning_brief_modernization" for row in rows)
    assert acl_rows
    assert all("case_id" in row for row in acl_rows)
    assert not any("SYN-FUSION" in str(row.get("excluded_documents", "")) for row in rows)
    assert unclassified_acl["final_answerability"] == "refused"
    assert unclassified_acl["answerability_reason"] == "denied_source_matches_query"
    assert unclassified_acl["documents_sent_to_model"] == 0
    assert unclassified_acl["zero_doc_refusal"] is True
    assert cited_lookup["final_answerability"] == "answered"
    assert cited_lookup["documents_sent_to_model"] > 0
    assert cited_lookup["zero_doc_refusal"] is False


def test_eval_default_prefers_curated_readiness_run(tmp_path: Path) -> None:
    preferred = tmp_path / PREFERRED_READINESS_RUN
    preferred.mkdir()
    newer = tmp_path / "insufficient_evidence_strategy_realignment_20260511_131708"
    newer.mkdir()
    os.utime(preferred, (100, 100))
    os.utime(newer, (200, 200))

    assert preferred_transcript_dir(tmp_path) == preferred


def test_eval_default_falls_back_to_latest_when_curated_run_missing(tmp_path: Path) -> None:
    older = tmp_path / "live_readiness_retry_20260511_012843"
    older.mkdir()
    newer = tmp_path / "insufficient_evidence_strategy_realignment_20260511_131708"
    newer.mkdir()
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    assert preferred_transcript_dir(tmp_path) == newer


def test_transcript_run_label_shows_case_count_and_pass_count(tmp_path: Path) -> None:
    run = tmp_path / "readiness_run"
    run.mkdir()
    (run / "passed.json").write_text(json.dumps({"case_id": "passed", "passed": True}), encoding="utf-8")
    (run / "failed.json").write_text(json.dumps({"case_id": "failed", "passed": False}), encoding="utf-8")

    assert transcript_run_label(run) == "readiness_run · 1/2 passed"


def test_eval_options_default_to_presentation_readiness_bundle(tmp_path: Path) -> None:
    preferred = tmp_path / PREFERRED_READINESS_RUN
    preferred.mkdir()
    refusal = tmp_path / INSUFFICIENT_EVIDENCE_READINESS_RUN
    refusal.mkdir()
    (preferred / "cited.json").write_text(
        json.dumps({"case_id": "cited", "passed": True}),
        encoding="utf-8",
    )
    (refusal / "insufficient.json").write_text(
        json.dumps({"case_id": "insufficient_evidence_planning_topic", "passed": True}),
        encoding="utf-8",
    )

    options = transcript_run_options(tmp_path)
    bundle = options[0]
    rows = load_eval_rows(bundle)

    assert bundle.is_bundle is True
    assert bundle.label == "Presentation readiness bundle"
    assert {row["case_id"] for row in rows} == {"cited", "insufficient_evidence_planning_topic"}
    assert {row["transcript_run"] for row in rows} == {
        PREFERRED_READINESS_RUN,
        INSUFFICIENT_EVIDENCE_READINESS_RUN,
    }
    assert transcript_run_label(bundle) == "Presentation readiness bundle · 2/2 passed"


def test_backend_bridge_rehydrates_agent_turn_result() -> None:
    result = _result_from_transcript(TRANSCRIPTS / "natural_multilingual_nato_core_tasks.json")
    rehydrated = agent_turn_result_from_dict(result.__dict__)

    assert rehydrated.session_id == result.session_id
    assert rehydrated.persona_id == result.persona_id
    assert rehydrated.answer_audit["query"]
    assert rehydrated.citation_mode == result.citation_mode


def test_backend_bridge_rehydrates_strict_worker_envelope() -> None:
    result = _result_from_transcript(TRANSCRIPTS / "natural_multilingual_nato_core_tasks.json")
    completed = _result_from_stdout(json.dumps({"ok": True, "result": result.__dict__}))

    assert completed is not None
    assert completed.answer == result.answer
    assert completed.answer_audit["query"] == result.answer_audit["query"]


def test_backend_bridge_error_detail_does_not_expose_stdout_payload() -> None:
    detail = _safe_error_detail("TimeoutError: upstream timed out\n" + ("x" * 1000))

    assert detail == "TimeoutError: upstream timed out"
    assert len(detail) < 240


def test_backend_bridge_handles_large_worker_json_without_polling_deadlock(monkeypatch) -> None:
    result = _result_from_transcript(TRANSCRIPTS / "natural_multilingual_nato_core_tasks.json")
    payload = result.__dict__.copy()
    payload["answer"] = "large answer " + ("x" * 300_000)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({"ok": True, "result": payload}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    completed = run_turn_in_subprocess(
        query="test",
        persona_id="clearance_unclassified",
        user_id="clearance_unclassified",
        session_id=None,
        timeout_seconds=1,
    )

    assert completed.answer.startswith("large answer")


def test_backend_bridge_timeout_ignores_partial_stdout(monkeypatch) -> None:
    result = _result_from_transcript(TRANSCRIPTS / "natural_multilingual_nato_core_tasks.json")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=1,
            output=json.dumps({"ok": True, "result": result.__dict__}),
            stderr="TimeoutError: backend still running",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        run_turn_in_subprocess(
            query="test",
            persona_id="clearance_unclassified",
            user_id="clearance_unclassified",
            session_id=None,
            timeout_seconds=1,
        )
    except TimeoutError as exc:
        assert "exceeded 1s" in str(exc)
    else:
        raise AssertionError("timeout should not return a partial worker result")
