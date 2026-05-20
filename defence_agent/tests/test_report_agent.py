from __future__ import annotations

import asyncio
import json
from pathlib import Path

from defence_agent.grounding import GroundedAnswer
from defence_agent.reports import orchestrator
from defence_agent.reports.eval_harness import evaluate_report_payload
from defence_agent.reports.templates import get_report_task


def _source(doc_id: str, index: int) -> dict[str, object]:
    return {
        "chunk_id": f"{doc_id}_page_{index:03d}",
        "doc_id": doc_id,
        "title": f"{doc_id} title",
        "page": index,
        "status": "approved",
        "access_level": "unclassified",
        "language": "en",
        "text": f"{doc_id} page {index} supports north NATO AI readiness planning.",
        "rerank_score": 0.9,
    }


def test_report_agent_runs_parallel_researchers_and_writes_artifacts(monkeypatch, tmp_path: Path) -> None:
    task = get_report_task("weekly_northern_readiness_brief")

    def fake_search_pages(*, query: str, persona_id: str, top_k: int, status_filter: str, language: str) -> dict[str, object]:
        del query, persona_id, top_k, status_filter, language
        sources = [_source(doc_id, index + 1) for index, doc_id in enumerate(task.expected_doc_ids)]
        return {
            "authorized_sources": sources,
            "excluded_sources": [],
            "answerability": {"answerable": True, "reason": "authorized_sources_available"},
            "policy_decision": "answer_from_authorized_sources",
            "index": "test",
            "collection": "test",
            "embedding_backend": "test",
            "retrieval_mode": "hybrid",
            "chunk_strategy": "page",
            "rerank_backend": "rerank-v4.0-pro",
            "allowed_access": ["unclassified"],
            "filters_applied": {"access_level": ["unclassified"]},
            "retrieval_metrics": {},
        }

    def fake_finalize_answer(**kwargs: object) -> GroundedAnswer:
        sources = list(kwargs["sources"])  # type: ignore[index]
        return GroundedAnswer(
            answer="The researcher finding covers north, NATO, AI, and readiness [C1].",
            raw_answer="The researcher finding covers north, NATO, AI, and readiness.",
            citations=[],
            citation_validation={"passed": True, "errors": [], "warnings": []},
            documents_sent=len(sources),
            document_ids=[str(source["chunk_id"]) for source in sources],  # type: ignore[index]
            model="command-a-03-2025",
        )

    def fake_synthesize_report(task_arg, findings, persona_id):  # type: ignore[no-untyped-def]
        del persona_id
        sources = [_source(doc_id, index + 1) for index, doc_id in enumerate(task_arg.expected_doc_ids)]
        markdown = (
            f"# {task_arg.title}\n\n"
            "## Executive Summary\n"
            "The weekly brief connects north posture, NATO alignment, AI governance, and readiness [C1].\n\n"
            "## Key Findings By Section\n"
            "- Northern posture is ready for staff review [C1].\n"
            "- NATO alignment remains a planning factor [C2].\n"
            "- AI adoption needs governed data and oversight [C3].\n\n"
            "## Evidence Gaps And Follow-Up Questions\n"
            "The report should be refreshed after new approved sources are added [C1].\n\n"
            "## Planning Implications\n"
            "Central planning staff can use this as a recurring evidence-grounded brief [C2].\n"
        )
        return orchestrator.ReportSynthesis(
            markdown=markdown,
            raw_markdown=markdown,
            citations=[
                {
                    "text": "north posture, NATO alignment, AI governance, and readiness",
                    "sources": [{"source_id": "CA-DEF-POL-2024-EN_page_001", "doc_id": "CA-DEF-POL-2024-EN"}],
                }
            ],
            citation_validation={"passed": True, "errors": [], "warnings": []},
            sources=sources,
            model="command-a-03-2025",
        )

    async def fake_review_answer(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "status": "approved",
            "release_gate": "release",
            "credibility_score": 1.0,
            "summary": "Report citations are supported.",
        }

    monkeypatch.setattr(orchestrator, "search_pages", fake_search_pages)
    monkeypatch.setattr(orchestrator, "finalize_answer", fake_finalize_answer)
    monkeypatch.setattr(orchestrator, "_synthesize_report", fake_synthesize_report)
    monkeypatch.setattr(orchestrator, "review_answer", fake_review_answer)

    result = asyncio.run(
        orchestrator.run_report_task(
            "weekly_northern_readiness_brief",
            out_root=tmp_path,
            job_id="test_weekly_report",
        )
    )

    assert result.job_id == "test_weekly_report"
    assert len(result.findings) == len(task.sections)
    assert result.run_trace["architecture"]["parallel_execution"] is True
    assert result.eval_summary["checks"]["parallel_research"]["status"] == "pass"
    assert (tmp_path / "test_weekly_report" / "report.md").exists()
    audit = json.loads((tmp_path / "test_weekly_report" / "report_audit.json").read_text(encoding="utf-8"))
    assert audit["live_model_calls_used_to_generate"] is True
    assert audit["model_calls_required_for_ui"] is False


def test_report_eval_harness_combines_code_and_llm_judge_slots() -> None:
    task = get_report_task("weekly_northern_readiness_brief")
    sources = [_source(doc_id, index + 1) for index, doc_id in enumerate(task.expected_doc_ids)]
    markdown = (
        f"# {task.title}\n\n"
        "## Executive Summary\n"
        "North, NATO, AI, and readiness are covered in this scheduled report [C1].\n\n"
        "## Key Findings By Section\n"
        "Northern posture, Allied and NATO alignment, and AI-enabled readiness are each covered [C2].\n\n"
        "## Evidence Gaps And Follow-Up Questions\n"
        "The approved sources should be refreshed monthly [C3].\n\n"
        "## Planning Implications\n"
        + " ".join(["This gives staff a cited planning baseline."] * 45)
        + " [C1]."
    )

    report = evaluate_report_payload(
        task=task,
        markdown=markdown,
        sources=sources,
        findings=[{"status": "completed"}, {"status": "completed"}],
        citations=[{"text": "North, NATO, AI, and readiness", "sources": [{"source_id": "C1"}]}],
        critic_report={"status": "approved", "release_gate": "release"},
    )

    assert report["evaluator_types"] == ["code", "llm_judge"]
    assert report["checks"]["source_coverage"]["status"] == "pass"
    assert report["checks"]["parallel_research"]["status"] == "pass"
    assert report["checks"]["llm_judge"]["status"] == "skipped"
