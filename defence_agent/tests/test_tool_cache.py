from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest


class FakeActions:
    skip_summarization = False


class FakeToolContext:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.actions = FakeActions()


@pytest.fixture()
def cache_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "manifest.yaml").write_text("documents: []\n", encoding="utf-8")
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("DEFENCE_AGENT_CORPUS_DIR", str(corpus_dir))
    monkeypatch.setenv("DEFENCE_AGENT_DATA_DIR", str(tmp_path / "data"))

    from defence_agent.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_tool_cache_replays_successful_search_result_after_failed_turn(cache_env: None) -> None:
    from defence_agent.tool_cache import SESSION_CACHE_STATE_KEY
    from defence_agent.tool_cache import after_tool_cache_store
    from defence_agent.tool_cache import before_tool_cache_lookup

    tool = SimpleNamespace(name="search_documents")
    args = {
        "query": "What rule applies before sharing a candidate observation?",
        "top_k": 8,
        "status_filter": "approved",
        "language": "any",
    }
    first_context = FakeToolContext({"persona_id": "clearance_secret"})

    stored_response = after_tool_cache_store(tool, args, first_context, _search_result())

    assert stored_response is not None
    assert stored_response["tool_cache"]["stored"] is True
    assert "text" not in stored_response["excluded_sources"][0]
    assert SESSION_CACHE_STATE_KEY in first_context.state

    retry_state = {
        "persona_id": "clearance_secret",
        SESSION_CACHE_STATE_KEY: deepcopy(first_context.state[SESSION_CACHE_STATE_KEY]),
    }
    retry_context = FakeToolContext(retry_state)
    cached_response = before_tool_cache_lookup(tool, args, retry_context)

    assert cached_response is not None
    assert cached_response["tool_cache"]["hit"] is True
    assert cached_response["authorized_sources"][0]["doc_id"] == "SYN-FUSION-S-RELEASE-001"
    assert retry_context.state["last_search_sources"][0]["text"] == "Release requires review before sharing."
    assert retry_context.state["last_search_audit"]["cache"]["hit"] is True
    assert retry_context.state["search_history_audits"][0]["cache"]["hit"] is True
    assert "text" not in retry_context.state["last_search_audit"]["excluded_sources"][0]
    assert retry_context.actions.skip_summarization is True


def test_tool_cache_key_is_persona_isolated(cache_env: None) -> None:
    from defence_agent.tool_cache import SESSION_CACHE_STATE_KEY
    from defence_agent.tool_cache import after_tool_cache_store
    from defence_agent.tool_cache import before_tool_cache_lookup

    tool = SimpleNamespace(name="search_documents")
    args = {"query": "What routing rule applies?", "top_k": 8}
    secret_context = FakeToolContext({"persona_id": "clearance_secret"})
    after_tool_cache_store(tool, args, secret_context, _search_result())

    unclassified_context = FakeToolContext(
        {
            "persona_id": "clearance_unclassified",
            SESSION_CACHE_STATE_KEY: deepcopy(secret_context.state[SESSION_CACHE_STATE_KEY]),
        }
    )

    assert before_tool_cache_lookup(tool, args, unclassified_context) is None


def test_tool_cache_key_changes_when_search_controls_change(cache_env: None) -> None:
    from defence_agent.tool_cache import cache_key_for_tool

    tool_context = FakeToolContext({"persona_id": "clearance_secret"})
    base_args = {"query": "What routing rule applies?", "top_k": 8, "status_filter": "approved", "language": "any"}

    base_key, _ = cache_key_for_tool("search_documents", base_args, tool_context)
    language_key, _ = cache_key_for_tool("search_documents", {**base_args, "language": "fr"}, tool_context)
    top_k_key, _ = cache_key_for_tool("search_documents", {**base_args, "top_k": 16}, tool_context)
    status_key, _ = cache_key_for_tool("search_documents", {**base_args, "status_filter": "any"}, tool_context)

    assert language_key != base_key
    assert top_k_key != base_key
    assert status_key != base_key


def test_tool_cache_key_normalizes_tool_defaults_and_bounds(cache_env: None) -> None:
    from defence_agent.tool_cache import cache_key_payload

    tool_context = FakeToolContext({"persona_id": "clearance_secret"})
    payload = cache_key_payload(
        "search_documents",
        {"query": "  What routing rule applies?  ", "top_k": 999, "status_filter": "current", "language": "english"},
        tool_context,
    )

    assert payload["args"] == {
        "query": "What routing rule applies?",
        "top_k": 24,
        "status_filter": "approved",
        "language": "en",
    }
    assert payload["persona_id"] == "clearance_secret"
    assert payload["allowed_access"] == ["unclassified", "secret"]


def _search_result() -> dict[str, Any]:
    return {
        "index": "chroma",
        "collection": "defence_agent_pdf_pages_1536",
        "embedding_backend": "embed-v4.0",
        "rerank_backend": "rerank-v4.0-pro",
        "persona_id": "clearance_secret",
        "allowed_access": ["unclassified", "secret"],
        "filters_applied": {
            "access_level": ["unclassified", "secret"],
            "status": "approved",
            "language": "any",
        },
        "policy_decision": "allow",
        "answerability": {"answerable": True, "reason": "authorized_sources_available"},
        "authorized_sources": [
            {
                "citation_id": "C1",
                "citation": "[C1]",
                "chunk_id": "SYN-FUSION-S-RELEASE-001_page_002",
                "doc_id": "SYN-FUSION-S-RELEASE-001",
                "title": "Fusion Model Release Control",
                "section": "Release threshold",
                "page": 2,
                "status": "approved",
                "version": "1.0",
                "effective_date": "2026-01-01",
                "access_level": "secret",
                "language": "en",
                "source_type": "synthetic_pdf",
                "source_pdf_path": "/tmp/fusion_model_release_control.pdf",
                "manifest_path": "/tmp/manifest.yaml",
                "page_image_sha256": "abc123",
                "vector_score": 0.8,
                "rerank_score": 0.9,
                "text": "Release requires review before sharing.",
            }
        ],
        "citation_guide": ["[C1] = SYN-FUSION-S-RELEASE-001 page 2 (Fusion Model Release Control)"],
        "excluded_sources": [
            {
                "doc_id": "SYN-FUSION-TS-ANNEX-002",
                "title": "Fusion Model Restricted Routing Annex",
                "access_level": "top_secret",
                "reason": "access_denied",
                "text": "This excluded text must not enter audit state.",
            }
        ],
        "answering_rule": "Answer only from authorized_sources.",
        "tool_name": "search_documents",
        "tool_policy": {
            "persona_id": "clearance_secret",
            "decision": "allow",
            "reason": "persona is allowed to call read-only document search",
        },
    }
