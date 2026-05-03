from __future__ import annotations

import os
import json

os.environ.setdefault("USE_MOCK_COHERE", "true")
os.environ.setdefault("COHERE_CHAT_MODEL", "command-a-03-2025")
os.environ.setdefault("COHERE_EMBED_MODEL", "embed-v4.0")
os.environ.setdefault("COHERE_RERANK_MODEL", "rerank-v3.5")

from fastapi.testclient import TestClient

from defence_agent.api import app
from defence_agent.evals.runner import eval_runner
from defence_agent.sandbox.python_sandbox import validate_code, SandboxValidationError


def test_direct_rag_returns_trace_and_citations() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_analyst"},
            json={"query": "What review steps should planning staff complete before approving a cross-unit planning request?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["trace_id"].startswith("tr_")
    assert data["route"] == "direct_rag"
    assert data["citations"]
    assert "[C" in data["answer"]


def test_restricted_query_differs_by_persona() -> None:
    with TestClient(app) as client:
        analyst = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_analyst"},
            json={"query": "What does Restricted Annex B say about exception handling?"},
        )
        lead = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_lead"},
            json={"query": "What does Restricted Annex B say about exception handling?"},
        )
    assert analyst.status_code == 200
    assert lead.status_code == 200
    analyst_data = analyst.json()
    lead_data = lead.json()
    assert analyst_data["route"] == "restricted_access"
    assert "cannot access" in analyst_data["answer"].lower()
    assert not analyst_data["sources"]
    assert any(source["title"] == "Restricted Annex B" for source in lead_data["sources"])
    assert "[C" in lead_data["answer"]


def test_table_analysis_uses_sandbox_code() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_analyst"},
            json={"query": "Using the readiness review table, which units fall below the 80% readiness threshold?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "table_analysis"
    assert "Bravo" in data["answer"]
    assert "Delta" in data["answer"]
    assert "Echo" in data["answer"]
    assert any(call["tool"] == "analyze_table_with_python" for call in data["tool_calls"])


def test_sandbox_blocks_dangerous_imports() -> None:
    try:
        validate_code("import os\nresult = {'ok': True}")
    except SandboxValidationError:
        return
    raise AssertionError("Sandbox accepted a blocked import")


def test_stream_endpoint_returns_sse_final_event() -> None:
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/agent/stream",
            headers={"X-Demo-User": "planning_analyst"},
            json={"query": "What review steps should planning staff complete before approving a cross-unit planning request?"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
    assert "event: final" in body
    final_payload = None
    for block in body.split("\n\n"):
        if block.startswith("event: final"):
            data_line = next(line for line in block.splitlines() if line.startswith("data:"))
            final_payload = json.loads(data_line.replace("data:", "", 1).strip())
    assert final_payload
    assert final_payload["trace_id"].startswith("tr_")
    assert final_payload["citations"]


def test_document_source_endpoints_enforce_acl() -> None:
    with TestClient(app) as client:
        lead_sources = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_lead"},
            json={"query": "What does Restricted Annex B say about exception handling?"},
        ).json()["sources"]
        restricted = next(source for source in lead_sources if source["title"] == "Restricted Annex B")

        allowed = client.get(
            f"/v1/documents/{restricted['document_id']}/chunks/{restricted['chunk_id']}",
            headers={"X-Demo-User": "planning_lead"},
        )
        blocked = client.get(
            f"/v1/documents/{restricted['document_id']}/chunks/{restricted['chunk_id']}",
            headers={"X-Demo-User": "planning_analyst"},
        )

    assert allowed.status_code == 200
    assert allowed.json()["chunk"]["chunk_id"] == restricted["chunk_id"]
    assert blocked.status_code == 403


def test_eval_dataset_has_24_cases() -> None:
    cases = eval_runner.load_cases()
    assert len(cases) >= 24
    assert {case["slice"] for case in cases} >= {
        "direct_answerable",
        "ambiguous",
        "unanswerable",
        "version_comparison",
        "table_analysis",
        "permission_sensitive",
        "prompt_injection",
        "stale_conflicting_source",
        "human_review",
    }
