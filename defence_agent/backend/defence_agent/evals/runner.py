from __future__ import annotations

import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sqlmodel import Session, desc, select

from defence_agent.agent.service import agent_service
from defence_agent.auth.context import DEMO_USERS
from defence_agent.db import engine, init_db
from defence_agent.ingestion.indexer import corpus_has_chunks, reindex_corpus
from defence_agent.models import AskRequest, EvalCaseResult, EvalRun
from defence_agent.observability.tracing import trace_manager


class EvalSummary(BaseModel):
    run_id: str
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    cases: list[dict[str, Any]] = Field(default_factory=list)


class EvalRunner:
    def __init__(self) -> None:
        self.dataset_path = Path(__file__).resolve().parents[3] / "evals" / "golden_dataset.yaml"

    def load_cases(self) -> list[dict[str, Any]]:
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data["cases"]

    def run(self) -> EvalSummary:
        init_db()
        if not corpus_has_chunks():
            reindex_corpus(force_generate=False)
        run_id = f"eval_{uuid.uuid4().hex[:12]}"
        cases = self.load_cases()
        started = time.perf_counter()
        with Session(engine) as session:
            session.add(EvalRun(id=run_id, status="started"))
            session.commit()

        results: list[dict[str, Any]] = []
        for case in cases:
            result = self._run_case(run_id, case)
            results.append(result)

        metrics = self._aggregate(results, elapsed_ms=(time.perf_counter() - started) * 1000)
        with Session(engine) as session:
            eval_run = session.get(EvalRun, run_id)
            if eval_run:
                eval_run.status = "completed"
                eval_run.metrics_json = json.dumps(metrics)
                session.add(eval_run)
                session.commit()
        return EvalSummary(run_id=run_id, status="completed", metrics=metrics, cases=results)

    def latest_results(self) -> dict[str, Any]:
        init_db()
        with Session(engine) as session:
            latest = session.exec(select(EvalRun).order_by(desc(EvalRun.started_at))).first()
            if not latest:
                return {"runs": [], "latest": None}
            cases = session.exec(select(EvalCaseResult).where(EvalCaseResult.run_id == latest.id)).all()
        return {
            "latest": {
                "run_id": latest.id,
                "status": latest.status,
                "metrics": json.loads(latest.metrics_json or "{}"),
                "started_at": latest.started_at.isoformat(),
                "completed_at": latest.completed_at.isoformat() if latest.completed_at else None,
            },
            "cases": [
                {
                    "case_id": case.case_id,
                    "slice": case.slice,
                    "passed": case.passed,
                    "metrics": json.loads(case.metrics_json or "{}"),
                    "trace_id": case.trace_id,
                    "error": case.error,
                }
                for case in cases
            ],
        }

    def _run_case(self, run_id: str, case: dict[str, Any]) -> dict[str, Any]:
        trace_id = trace_manager.new_trace_id()
        auth = DEMO_USERS[case["user"]]
        trace_manager.start_trace(trace_id, auth.user_id, {"eval_case": case["id"], "query": case["query"]})
        started = time.perf_counter()
        error = None
        try:
            response = agent_service.handle(
                AskRequest(query=case["query"], route_override=case.get("route_override")),
                auth,
                trace_id,
            )
            payload = response.model_dump()
            trace_manager.finish_trace(trace_id, "ok", payload, route=response.route, duration_ms=response.latency_ms)
            metrics = self._score_case(case, payload, elapsed_ms=(time.perf_counter() - started) * 1000)
        except Exception as exc:
            error = str(exc)
            payload = {}
            metrics = {"passed": False, "error": True}
            trace_manager.finish_trace(trace_id, "error", {"error": error}, route=None, duration_ms=(time.perf_counter() - started) * 1000)

        result = {
            "case_id": case["id"],
            "slice": case["slice"],
            "passed": bool(metrics.get("passed")),
            "metrics": metrics,
            "trace_id": trace_id,
            "error": error,
        }
        with Session(engine) as session:
            session.add(
                EvalCaseResult(
                    run_id=run_id,
                    case_id=case["id"],
                    slice=case["slice"],
                    passed=bool(metrics.get("passed")),
                    metrics_json=json.dumps(metrics),
                    trace_id=trace_id,
                    error=error,
                )
            )
            session.commit()
        return result

    def _score_case(self, case: dict[str, Any], response: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
        source_titles = {source["title"] for source in response.get("sources", [])}
        source_ids = {source.get("document_id") for source in response.get("sources", [])}
        expected_docs = set(case.get("expected_docs", []))
        expected_source_ids = set(case.get("expected_sources", []))
        tool_names = {call.get("tool") for call in response.get("tool_calls", []) if isinstance(call, dict)}
        expected_tools = set(case.get("expected_tools", []))
        answer = response.get("answer", "").lower()
        citation_presence = bool(response.get("citations")) and "[c" in answer
        expected_route = case.get("expected_route")
        expected_denied = bool(case.get("expect_permission_denied", False))
        permission_correct = True
        if expected_denied:
            permission_correct = "restricted annex b" not in " ".join(source_titles).lower() and (
                "cannot access" in answer or "human review" in answer or not response.get("sources")
            )
        retrieval_recall = 1.0
        context_precision = 1.0
        if expected_source_ids:
            retrieval_recall = len(expected_source_ids.intersection(source_ids)) / len(expected_source_ids)
            context_precision = len(expected_source_ids.intersection(source_ids)) / max(len(source_ids), 1)
        elif expected_docs:
            retrieval_recall = len(expected_docs.intersection(source_titles)) / len(expected_docs)
            context_precision = len(expected_docs.intersection(source_titles)) / max(len(source_titles), 1)
        elif response.get("sources") and case.get("expect_abstain"):
            retrieval_recall = 0.0
            context_precision = 0.0

        abstained = any(
            answer.startswith(prefix)
            for prefix in [
                "i cannot access",
                "i cannot process",
                "i could not complete",
                "i do not have enough",
                "i need more detail",
                "this request needs human review",
                "this answer requires",
                "the approved documents do not provide enough support",
                "approved documents do not provide enough support",
            ]
        )
        metrics = {
            "route_accuracy": response.get("route") == expected_route,
            "tool_accuracy": expected_tools.issubset(tool_names),
            "retrieval_recall_at_k": retrieval_recall,
            "context_precision": context_precision,
            "citation_presence": citation_presence == bool(case.get("expect_citations")),
            "citation_validation": response.get("safety", {}).get("citation_validation", {}).get("ok", False),
            "permission_correctness": permission_correct,
            "abstention_correctness": abstained == bool(case.get("expect_abstain", False)),
            "safety_pass": not response.get("safety", {}).get("output", {}).get("blocked", False),
            "latency_ms": elapsed_ms,
            "error": False,
        }
        contains = [str(item).lower() for item in case.get("expected_answer_contains", [])]
        metrics["answer_contains"] = all(item in answer for item in contains)
        forbidden_sources = set(case.get("forbidden_sources", []))
        metrics["forbidden_source_absent"] = not forbidden_sources.intersection(source_ids)
        if case.get("structured_expected"):
            structured = case["structured_expected"]
            metrics["structured_exact"] = all(str(value).lower() in answer for value in structured)
        else:
            metrics["structured_exact"] = True
        metrics["passed"] = (
            metrics["route_accuracy"]
            and metrics["tool_accuracy"]
            and metrics["permission_correctness"]
            and metrics["abstention_correctness"]
            and metrics["safety_pass"]
            and (retrieval_recall > 0 if expected_docs or expected_source_ids else True)
            and metrics["citation_presence"]
            and metrics["answer_contains"]
            and metrics["forbidden_source_absent"]
            and metrics["structured_exact"]
        )
        return metrics

    def _aggregate(self, results: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
        case_metrics = [result["metrics"] for result in results]
        slices: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            slices.setdefault(result["slice"], []).append(result["metrics"])

        def rate(name: str, items: list[dict[str, Any]] = case_metrics) -> float:
            if not items:
                return 0.0
            return sum(1 for item in items if item.get(name)) / len(items)

        return {
            "case_count": len(results),
            "pass_rate": rate("passed"),
            "route_accuracy": rate("route_accuracy"),
            "tool_accuracy": rate("tool_accuracy"),
            "retrieval_recall_at_k": statistics.mean(item.get("retrieval_recall_at_k", 0.0) for item in case_metrics),
            "context_precision": statistics.mean(item.get("context_precision", 0.0) for item in case_metrics),
            "citation_presence": rate("citation_presence"),
            "citation_validation": rate("citation_validation"),
            "permission_correctness": rate("permission_correctness"),
            "abstention_correctness": rate("abstention_correctness"),
            "safety_pass_rate": rate("safety_pass"),
            "answer_contains": rate("answer_contains"),
            "forbidden_source_absent": rate("forbidden_source_absent"),
            "structured_exact": rate("structured_exact"),
            "latency_p50_ms": statistics.median(item.get("latency_ms", 0.0) for item in case_metrics),
            "error_rate": rate("error"),
            "elapsed_ms": elapsed_ms,
            "by_slice": {
                slice_name: {
                    "count": len(items),
                    "pass_rate": rate("passed", items),
                    "route_accuracy": rate("route_accuracy", items),
                    "permission_correctness": rate("permission_correctness", items),
                    "safety_pass_rate": rate("safety_pass", items),
                }
                for slice_name, items in sorted(slices.items())
            },
        }


eval_runner = EvalRunner()
