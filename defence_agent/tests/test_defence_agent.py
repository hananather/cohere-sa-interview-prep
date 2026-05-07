from __future__ import annotations

import os
import json

os.environ.setdefault("USE_MOCK_COHERE", "true")
os.environ.setdefault("COHERE_CHAT_MODEL", "command-a-03-2025")
os.environ.setdefault("COHERE_EMBED_MODEL", "embed-v4.0")
os.environ.setdefault("COHERE_RERANK_MODEL", "rerank-v4.0-pro")

from fastapi.testclient import TestClient

from defence_agent.api import app
from defence_agent.evals.advanced_runner import advanced_eval_runner
from defence_agent.evals.load_cases import load_suite
from defence_agent.evals.runner import eval_runner
from defence_agent.sandbox.python_sandbox import validate_code, SandboxValidationError


def test_evidence_lookup_returns_trace_and_citations() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_analyst"},
            json={"query": "What review steps are required before a planning brief is approved?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["trace_id"].startswith("tr_")
    assert data["route"] == "evidence_lookup"
    assert data["citations"]
    assert "[C" in data["answer"]
    assert {source["document_id"] for source in data["sources"]} == {"PB-SOP-2025"}


def test_permission_sensitive_query_differs_by_persona() -> None:
    with TestClient(app) as client:
        analyst = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_analyst"},
            json={"query": "What restricted annex handling steps apply before external distribution?"},
        )
        lead = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_lead"},
            json={"query": "What restricted annex handling steps apply before external distribution?"},
        )
    assert analyst.status_code == 200
    assert lead.status_code == 200
    analyst_data = analyst.json()
    lead_data = lead.json()
    assert analyst_data["route"] == "permission_sensitive_retrieval"
    assert "restricted source" in analyst_data["answer"].lower()
    assert not analyst_data["sources"]
    assert any(source["document_id"] == "ANNEX-HANDLING-2025" for source in lead_data["sources"])
    assert "[C" in lead_data["answer"]


def test_structured_table_analysis_uses_sandbox_code() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_analyst"},
            json={"query": "Which planning procedures are overdue for review? Group them by owner and show how many days overdue."},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "structured_table_analysis"
    assert "PB-SOP-2025" in data["answer"]
    assert "52 days overdue" in data["answer"]
    assert "LOG-RET-2025" in data["answer"]
    assert "96 days overdue" in data["answer"]
    assert any(call["tool"] == "run_table_analysis" for call in data["tool_calls"])


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
            json={"query": "What review steps are required before a planning brief is approved?"},
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
            json={"query": "What restricted annex handling steps apply before external distribution?"},
        ).json()["sources"]
        restricted = next(source for source in lead_sources if source["document_id"] == "ANNEX-HANDLING-2025")

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
        "evidence_lookup",
        "ambiguous",
        "unanswerable",
        "version_comparison",
        "structured_analysis",
        "permission_sensitive",
        "metadata_sensitive",
        "multi_hop",
        "bilingual",
        "claim_verification",
        "human_review",
    }


def test_golden_eval_suite_passes_fixture_mode() -> None:
    summary = eval_runner.run()
    assert summary.metrics["pass_rate"] == 1.0
    assert summary.metrics["route_accuracy"] == 1.0
    assert summary.metrics["permission_correctness"] == 1.0
    assert summary.metrics["structured_exact"] == 1.0


def test_advanced_eval_dataset_meets_distribution_targets() -> None:
    validation = advanced_eval_runner.validate()
    assert validation["ok"] is True
    assert validation["suite_counts"]["generated"] >= 120
    generated = load_suite("generated").cases
    task_counts: dict[str, int] = {}
    complexity_counts: dict[str, int] = {}
    for case in generated:
        task_counts[case.task_type] = task_counts.get(case.task_type, 0) + 1
        complexity_counts[case.complexity_level] = complexity_counts.get(case.complexity_level, 0) + 1
    assert task_counts["structured_analysis"] >= 10
    assert task_counts["bilingual"] >= 10
    assert task_counts["permission_sensitive"] >= 10
    assert task_counts["adversarial"] >= 8
    assert complexity_counts.get("L4", 0) + complexity_counts.get("L5", 0) >= 30


def test_advanced_canonical_fixture_eval_passes() -> None:
    report = advanced_eval_runner.run_suite("canonical", mode="fixture", variant="agentic_rag_tools")
    assert report.metrics["case_count"] == 24
    assert report.metrics["pass_rate"] == 1.0
    assert report.metrics["route_accuracy"] == 1.0
    assert report.metrics["citation_validation_pass_rate"] == 1.0
    assert report.metrics["structured_analysis_exact_correctness"] == 1.0
