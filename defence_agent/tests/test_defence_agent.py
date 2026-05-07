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
from defence_agent.tools.registry import tool_registry


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


def test_persona_compare_shows_partial_vs_restricted_answer() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/persona/compare",
            headers={"X-Demo-User": "planning_analyst"},
            json={
                "query": "What are the approval steps for a planning brief that includes restricted annexes?",
                "left_persona": "planning_analyst",
                "right_persona": "planning_lead",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "PB-SOP-2025" in {source["document_id"] for source in data["left"]["sources"]}
    assert "ANNEX-HANDLING-2025" not in {source["document_id"] for source in data["left"]["sources"]}
    assert "ANNEX-HANDLING-2025" in {source["document_id"] for source in data["right"]["sources"]}
    assert "ANNEX-HANDLING-2025" in data["diff"]["documents_only_right"]


def test_structured_trace_redacts_restricted_answer_for_auditor() -> None:
    with TestClient(app) as client:
        lead = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_lead"},
            json={"query": "What restricted annex handling steps apply before external distribution?"},
        )
        trace_id = lead.json()["trace_id"]
        auditor_trace = client.get(f"/v1/traces/{trace_id}/structured", headers={"X-Demo-User": "auditor"})
        lead_trace = client.get(f"/v1/traces/{trace_id}/structured", headers={"X-Demo-User": "planning_lead"})
    assert auditor_trace.status_code == 200
    assert lead_trace.status_code == 200
    assert auditor_trace.json()["generation"]["answer"].startswith("[answer redacted")
    assert "restricted" in lead_trace.json()["generation"]["answer"].lower()


def test_governance_endpoints_show_tool_registry_and_personas() -> None:
    with TestClient(app) as client:
        personas = client.get("/v1/personas", headers={"X-Demo-User": "auditor"})
        registry = client.get("/v1/governance/tool-registry", headers={"X-Demo-User": "auditor"})
        steps = client.get("/v1/demo/steps", headers={"X-Demo-User": "planning_analyst"})
    assert personas.status_code == 200
    assert registry.status_code == 200
    assert steps.status_code == 200
    assert any(item["display_name"] == "Alex Chen" for item in personas.json()["personas"])
    assert any(item["tool_name"] == "run_table_analysis" for item in registry.json()["tools"])
    assert len(steps.json()["steps"]) >= 7


def test_tool_policy_blocks_unauthorized_admin_action_and_traces_decision() -> None:
    from defence_agent.auth.context import DEMO_USERS
    from defence_agent.observability.tracing import trace_manager

    trace_id = trace_manager.new_trace_id()
    trace_manager.start_trace(trace_id, "planning_analyst", {"test": "blocked_tool"})
    try:
        tool_registry.call("admin_reindex", {}, DEMO_USERS["planning_analyst"], trace_id)
    except Exception:
        pass
    trace_manager.finish_trace(trace_id, "ok", {"blocked": True}, route="test", duration_ms=0)
    trace = trace_manager.get_trace(trace_id)
    assert trace
    assert any(span["name"] == "before_tool_policy" and span["status"] == "blocked" for span in trace["spans"])


def test_feedback_event_links_to_trace_id() -> None:
    with TestClient(app) as client:
        answer = client.post(
            "/v1/agent/query",
            headers={"X-Demo-User": "planning_analyst"},
            json={"query": "What review steps are required before a planning brief is approved?"},
        ).json()
        feedback = client.post(
            "/v1/feedback",
            headers={"X-Demo-User": "planning_analyst"},
            json={"trace_id": answer["trace_id"], "rating": "somewhat", "reason": "Missing source"},
        )
    assert feedback.status_code == 200
    assert feedback.json()["data"]["stored"] is True


def test_persona_security_graders_are_part_of_eval_breakdown() -> None:
    outcome = advanced_eval_runner.run_case("CAN_Q_PERM_095", suite="canonical", mode="fixture", variant="agentic_rag_tools")
    assert outcome.grades.persona_policy
    assert outcome.grades.excluded_sources
    assert outcome.grades.trace_completeness["passed"] is True
    assert outcome.grades.redaction["passed"] is True
