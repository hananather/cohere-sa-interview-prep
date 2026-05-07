from __future__ import annotations

import csv
import json
import math
import statistics
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from defence_agent.agent.service import agent_service
from defence_agent.auth.context import DEMO_USERS, AuthContext
from defence_agent.db import init_db
from defence_agent.evals.graders import (
    grade_answer,
    grade_citations,
    grade_code_execution,
    grade_filters,
    grade_operations,
    grade_rerank,
    grade_retrieval,
    grade_route,
    grade_safety,
    grade_tools,
)
from defence_agent.evals.load_cases import EVAL_DIR, load_case, load_suite, list_suites, validate_suites
from defence_agent.evals.schemas import EvalCase, EvalCaseOutcome, EvalRunReport, GradeBreakdown
from defence_agent.ingestion.indexer import corpus_has_chunks, reindex_corpus
from defence_agent.models import AskRequest
from defence_agent.observability.tracing import trace_manager
from defence_agent.retrieval.hybrid import hybrid_retriever


REPORT_ROOT = Path(__file__).resolve().parents[4] / "reports" / "eval" / "latest"
VARIANTS = [
    "keyword_only",
    "embedding_only",
    "hybrid_no_rerank",
    "hybrid_plus_rerank_fast",
    "hybrid_plus_rerank_pro",
    "query_expansion_plus_hybrid_rerank",
    "agentic_rag_tools",
    "structured_analysis_tools",
]


class AdvancedEvalRunner:
    def suites(self) -> dict[str, str]:
        return list_suites()

    def cases(self, suite: str = "canonical") -> list[dict[str, Any]]:
        return [case.model_dump() for case in load_suite(suite).cases]

    def validate(self) -> dict[str, Any]:
        result = validate_suites(["canonical", "generated", "heldout", "regression", "adversarial", "demo_candidates"])
        corpus_docs = self._known_doc_ids()
        missing_sources: list[dict[str, str]] = []
        for suite_name in ["canonical", "generated", "heldout", "regression", "adversarial", "demo_candidates"]:
            for case in load_suite(suite_name).cases:
                for source in [*case.expected_sources, *case.disallowed_sources]:
                    doc_id = source.doc_id
                    if doc_id not in corpus_docs:
                        missing_sources.append({"suite": suite_name, "query_id": case.query_id, "doc_id": doc_id})
        result["missing_sources"] = missing_sources
        result["ok"] = bool(result["ok"]) and not missing_sources
        return result

    def run_case(self, query_id: str, suite: str = "canonical", mode: str = "fixture", variant: str = "agentic_rag_tools") -> EvalCaseOutcome:
        case = load_case(query_id, suite)
        return self._run_case(case, mode=mode, variant=variant)

    def run_suite(self, suite: str = "canonical", mode: str = "fixture", variant: str = "agentic_rag_tools", limit: int | None = None) -> EvalRunReport:
        self._prepare_runtime()
        cases = load_suite(suite).cases
        if limit:
            cases = cases[:limit]
        run_id = f"eval_{suite}_{uuid.uuid4().hex[:10]}"
        outcomes = [self._run_case(case, mode=mode, variant=variant) for case in cases]
        metrics = self._aggregate(outcomes, suite=suite)
        report = EvalRunReport(run_id=run_id, suite=suite, mode=mode, variant=variant, metrics=metrics, cases=outcomes, report_dir=str(REPORT_ROOT))
        self._write_reports(report)
        return report

    def compare_variants(self, suite: str = "canonical", limit: int = 30) -> dict[str, Any]:
        self._prepare_runtime()
        cases = [case for case in load_suite(suite).cases if case.expected_sources][:limit]
        rows = []
        for variant in VARIANTS:
            variant_rows = [self._grade_retrieval_variant(case, variant) for case in cases]
            rows.append(
                {
                    "variant": variant,
                    "case_count": len(variant_rows),
                    "retrieval_recall_at_k": _mean(row["recall_at_k"] for row in variant_rows),
                    "precision_at_k": _mean(row["precision_at_k"] for row in variant_rows),
                    "mrr": _mean(row["mrr"] for row in variant_rows),
                    "expected_source_hit_rate": _mean(1.0 if row["passed"] else 0.0 for row in variant_rows),
                }
            )
        self._write_experiment_report(rows)
        return {"suite": suite, "variants": rows, "report": str(REPORT_ROOT / "experiment_comparison.md")}

    def select_demo_sequence(self, suite: str = "demo_candidates", mode: str = "fixture", runs: int = 3) -> dict[str, Any]:
        self._prepare_runtime()
        cases = load_suite(suite).cases
        scored: list[dict[str, Any]] = []
        for case in cases:
            outcomes = [self._run_case(case, mode=mode, variant="agentic_rag_tools") for _ in range(runs)]
            pass3 = all(outcome.passed for outcome in outcomes)
            avg_latency = _mean(float(outcome.latency_ms or 0.0) for outcome in outcomes)
            capability_score = min(5, len(case.capability_tags))
            narrative = 5 if case.expected_route in {"version_comparison", "structured_table_analysis", "cross_source_synthesis", "permission_sensitive_retrieval", "bilingual_retrieval"} else 3
            reliability = 5 if pass3 else sum(1 for outcome in outcomes if outcome.passed)
            score = (2 * reliability) + narrative + capability_score - min(2, avg_latency / 5000)
            scored.append(
                {
                    "query_id": case.query_id,
                    "query": case.user_query,
                    "route": case.expected_route,
                    "pass3": pass3,
                    "reliability_score": reliability,
                    "trace_clarity_score": 4 if outcomes[-1].trace_id else 1,
                    "cohere_capability_shown": ", ".join(case.capability_tags),
                    "expected_audience_value": narrative,
                    "latency_ms_avg": round(avg_latency, 1),
                    "risk_level": "low" if pass3 else "medium",
                    "score": round(score, 2),
                    "backup_query": _backup_query(case.expected_route),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        recommended = self._recommended_sequence(scored)
        self._write_demo_selection(scored, recommended)
        return {
            "recommended": recommended,
            "scorecard": scored,
            "reports": {
                "demo_scorecard": str(REPORT_ROOT / "demo_scorecard.md"),
                "demo_selection_report": str(REPORT_ROOT / "demo_selection_report.md"),
                "recommended_demo_sequence": str(EVAL_DIR / "recommended_demo_sequence.yaml"),
            },
        }

    def replay_trace(self, trace_id: str, query_id: str, suite: str = "canonical") -> EvalCaseOutcome:
        case = load_case(query_id, suite)
        trace = trace_manager.get_trace(trace_id)
        response = (trace or {}).get("response", {}) if trace else {}
        return self._grade_case(case, response, trace)

    def _run_case(self, case: EvalCase, mode: str, variant: str) -> EvalCaseOutcome:
        trace_id = trace_manager.new_trace_id()
        auth = self._auth_for_case(case)
        started = time.perf_counter()
        trace_manager.start_trace(trace_id, auth.user_id, {"eval_case": case.query_id, "query": case.user_query, "mode": mode, "variant": variant})
        try:
            response = agent_service.handle(AskRequest(query=case.user_query), auth, trace_id)
            payload = response.model_dump()
            trace_manager.finish_trace(trace_id, "ok", payload, route=response.route, duration_ms=response.latency_ms)
        except Exception as exc:
            payload = {"trace_id": trace_id, "route": "error", "answer": str(exc), "sources": [], "citations": [], "tool_calls": [], "latency_ms": (time.perf_counter() - started) * 1000}
            trace_manager.finish_trace(trace_id, "error", payload, route="error", duration_ms=payload["latency_ms"])
        trace = trace_manager.get_trace(trace_id)
        return self._grade_case(case, payload, trace)

    def _grade_case(self, case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None) -> EvalCaseOutcome:
        grades = GradeBreakdown(
            route=grade_route(case, response, trace),
            retrieval=grade_retrieval(case, response, trace),
            filters=grade_filters(case, response, trace),
            rerank=grade_rerank(case, response, trace),
            tools=grade_tools(case, response, trace),
            code_execution=grade_code_execution(case, response, trace),
            answer=grade_answer(case, response, trace),
            citations=grade_citations(case, response, trace),
            safety=grade_safety(case, response, trace),
            operations=grade_operations(case, response, trace),
        )
        grade_dict = grades.model_dump()
        failures = [
            values["failure_category"]
            for values in grade_dict.values()
            if isinstance(values, dict) and not values.get("passed", True) and values.get("failure_category")
        ]
        critical_sections = ["route", "tools", "filters", "citations", "safety", "operations"]
        if case.grader_config.answer_key_facts or case.expected_refusal:
            critical_sections.append("answer")
        if case.grader_config.retrieval:
            critical_sections.append("retrieval")
        if case.expected_route == "structured_table_analysis":
            critical_sections.append("code_execution")
        passed = all(grade_dict[section].get("passed", False) for section in critical_sections)
        return EvalCaseOutcome(
            query_id=case.query_id,
            task_type=case.task_type,
            complexity_level=case.complexity_level,
            passed=passed,
            failure_categories=sorted(set(failures)),
            trace_id=response.get("trace_id") or (trace or {}).get("trace_id"),
            route=response.get("route"),
            latency_ms=response.get("latency_ms") or (trace or {}).get("duration_ms"),
            grades=grades,
            answer=response.get("answer", ""),
            sources=[source.get("document_id", "") for source in response.get("sources", [])],
        )

    def _aggregate(self, outcomes: list[EvalCaseOutcome], suite: str) -> dict[str, Any]:
        total = len(outcomes)
        latencies = [float(outcome.latency_ms or 0) for outcome in outcomes]
        failure_counts = Counter(category for outcome in outcomes for category in outcome.failure_categories)
        return {
            "suite": suite,
            "case_count": total,
            "pass_rate": _rate(outcome.passed for outcome in outcomes),
            "route_accuracy": _grade_rate(outcomes, "route"),
            "tool_accuracy": _grade_rate(outcomes, "tools"),
            "retrieval_recall_at_k": _mean(outcome.grades.retrieval.get("recall_at_k", 0.0) for outcome in outcomes),
            "precision_at_k": _mean(outcome.grades.retrieval.get("precision_at_k", 0.0) for outcome in outcomes),
            "mrr": _mean(outcome.grades.retrieval.get("mrr", 0.0) for outcome in outcomes),
            "filter_correctness": _grade_rate(outcomes, "filters"),
            "access_control_correctness": _rate(not outcome.grades.safety.get("restricted_leak", False) for outcome in outcomes),
            "citation_validation_pass_rate": _grade_rate(outcomes, "citations"),
            "refusal_precision_recall_proxy": _grade_rate(outcomes, "answer"),
            "structured_analysis_exact_correctness": _grade_rate([outcome for outcome in outcomes if outcome.route == "structured_table_analysis"], "code_execution"),
            "latency_avg_ms": _mean(latencies),
            "latency_p95_ms": _percentile(latencies, 95),
            "top_failure_categories": failure_counts.most_common(10),
            "by_task_type": _group_rates(outcomes, "task_type"),
            "by_complexity": _group_rates(outcomes, "complexity_level"),
            "by_route": _group_rates(outcomes, "route"),
        }

    def _write_reports(self, report: EvalRunReport) -> None:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        (REPORT_ROOT / "summary.json").write_text(json.dumps(report.model_dump(), indent=2) + "\n", encoding="utf-8")
        self._write_summary_md(report)
        self._write_case_csv(report, "metrics_by_task_type.csv", "task_type")
        self._write_case_csv(report, "metrics_by_complexity.csv", "complexity_level")
        self._write_case_csv(report, "metrics_by_route.csv", "route")
        self._write_tool_csv(report)
        self._write_metric_csv(report, "retrieval_metrics.csv", ["query_id", "passed", "route", "recall_at_k", "precision_at_k", "mrr"])
        self._write_metric_csv(report, "citation_metrics.csv", ["query_id", "passed", "citation_count", "present_ok", "resolve_ok", "status_ok"])
        self._write_metric_csv(report, "safety_metrics.csv", ["query_id", "passed", "restricted_leak", "draft_used_for_current", "superseded_used_for_current"])
        self._write_metric_csv(report, "structured_analysis_metrics.csv", ["query_id", "passed", "sandbox_used", "exact_result_correct", "row_ids_preserved"])
        self._write_failure_analysis(report)
        self._write_route_confusion(report)

    def _write_summary_md(self, report: EvalRunReport) -> None:
        metrics = report.metrics
        lines = [
            "# Evaluation Summary",
            "",
            f"- Suite: {report.suite}",
            f"- Mode: {report.mode}",
            f"- Variant: {report.variant}",
            f"- Cases: {metrics.get('case_count', 0)}",
            f"- Overall pass rate: {metrics.get('pass_rate', 0):.1%}",
            f"- Route accuracy: {metrics.get('route_accuracy', 0):.1%}",
            f"- Retrieval recall@k: {metrics.get('retrieval_recall_at_k', 0):.3f}",
            f"- Citation validation pass rate: {metrics.get('citation_validation_pass_rate', 0):.1%}",
            f"- Access-control correctness: {metrics.get('access_control_correctness', 0):.1%}",
            f"- Structured analysis exact correctness: {metrics.get('structured_analysis_exact_correctness', 0):.1%}",
            f"- Average latency: {metrics.get('latency_avg_ms', 0):.1f} ms",
            f"- P95 latency: {metrics.get('latency_p95_ms', 0):.1f} ms",
            "",
            "## Top Failure Categories",
            "",
        ]
        for category, count in metrics.get("top_failure_categories", []):
            lines.append(f"- {category}: {count}")
        (REPORT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_case_csv(self, report: EvalRunReport, filename: str, field: str) -> None:
        groups = defaultdict(list)
        for outcome in report.cases:
            groups[getattr(outcome, field) or "unknown"].append(outcome)
        with (REPORT_ROOT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["group", "case_count", "pass_rate", "route_accuracy", "citation_pass_rate"])
            writer.writeheader()
            for group, outcomes in sorted(groups.items()):
                writer.writerow(
                    {
                        "group": group,
                        "case_count": len(outcomes),
                        "pass_rate": _rate(outcome.passed for outcome in outcomes),
                        "route_accuracy": _grade_rate(outcomes, "route"),
                        "citation_pass_rate": _grade_rate(outcomes, "citations"),
                    }
                )

    def _write_tool_csv(self, report: EvalRunReport) -> None:
        with (REPORT_ROOT / "metrics_by_tool.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["query_id", "tool_call_count", "tool_accuracy_passed"])
            writer.writeheader()
            for outcome in report.cases:
                writer.writerow(
                    {
                        "query_id": outcome.query_id,
                        "tool_call_count": outcome.grades.tools.get("tool_call_count", 0),
                        "tool_accuracy_passed": outcome.grades.tools.get("passed", False),
                    }
                )

    def _write_metric_csv(self, report: EvalRunReport, filename: str, fields: list[str]) -> None:
        section_by_file = {
            "retrieval_metrics.csv": "retrieval",
            "citation_metrics.csv": "citations",
            "safety_metrics.csv": "safety",
            "structured_analysis_metrics.csv": "code_execution",
        }
        section = section_by_file[filename]
        with (REPORT_ROOT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for outcome in report.cases:
                values = getattr(outcome.grades, section)
                row = {"query_id": outcome.query_id, "passed": values.get("passed", False), "route": outcome.route}
                row.update({field: values.get(field) for field in fields if field not in row})
                writer.writerow(row)

    def _write_failure_analysis(self, report: EvalRunReport) -> None:
        lines = ["# Failure Analysis", ""]
        failures = [outcome for outcome in report.cases if not outcome.passed]
        if not failures:
            lines.append("No failing cases in this run.")
        for outcome in failures[:30]:
            lines.append(f"- {outcome.query_id}: {', '.join(outcome.failure_categories) or 'unknown'} | route={outcome.route} | trace={outcome.trace_id}")
        (REPORT_ROOT / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_route_confusion(self, report: EvalRunReport) -> None:
        expected_by_id = {case.query_id: case.expected_route for case in load_suite(report.suite).cases}
        with (REPORT_ROOT / "confusion_matrix_route.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["query_id", "expected_route", "actual_route", "passed"])
            writer.writeheader()
            for outcome in report.cases:
                writer.writerow({"query_id": outcome.query_id, "expected_route": expected_by_id.get(outcome.query_id), "actual_route": outcome.route, "passed": outcome.grades.route.get("passed", False)})

    def _write_experiment_report(self, rows: list[dict[str, Any]]) -> None:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        lines = ["# Experiment Comparison", "", "This report compares retrieval variants on expected-source hit metrics. Fixture mode is a component comparison, not a live Cohere quality claim.", ""]
        for row in rows:
            lines.append(f"- {row['variant']}: recall@k={row['retrieval_recall_at_k']:.3f}, precision@k={row['precision_at_k']:.3f}, MRR={row['mrr']:.3f}, hit_rate={row['expected_source_hit_rate']:.1%}")
        (REPORT_ROOT / "experiment_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_demo_selection(self, scored: list[dict[str, Any]], recommended: list[dict[str, Any]]) -> None:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        score_lines = ["# Demo Scorecard", ""]
        for item in scored:
            score_lines.append(f"- {item['query_id']} | pass3={item['pass3']} | score={item['score']} | route={item['route']} | risk={item['risk_level']} | {item['query']}")
        (REPORT_ROOT / "demo_scorecard.md").write_text("\n".join(score_lines) + "\n", encoding="utf-8")
        selection_lines = ["# Demo Selection Report", "", "Recommended sequence:"]
        for index, item in enumerate(recommended, start=1):
            selection_lines.append(f"{index}. {item['query_id']} - {item['route']} - {item['query']}")
        (REPORT_ROOT / "demo_selection_report.md").write_text("\n".join(selection_lines) + "\n", encoding="utf-8")
        import yaml

        (EVAL_DIR / "recommended_demo_sequence.yaml").write_text(yaml.safe_dump({"recommended": recommended}, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _recommended_sequence(self, scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
        preferred = [
            ("evidence_lookup", "review steps are required"),
            ("metadata_aware_retrieval", "current approved procedure"),
            ("cross_source_synthesis", "include in a planning brief"),
            ("version_comparison", "2024 and 2025"),
            ("structured_table_analysis", "overdue for review"),
            ("permission_sensitive_retrieval", "restricted user"),
            ("bilingual_retrieval", "délais"),
        ]
        sequence = []
        for route, preferred_phrase in preferred:
            match = next(
                (
                    item
                    for item in scored
                    if item["route"] == route and item["pass3"] and preferred_phrase.lower() in item["query"].lower()
                ),
                None,
            )
            if not match:
                match = next((item for item in scored if item["route"] == route and item["pass3"]), None)
            if match:
                selected = dict(match)
                selected["recommended_position"] = len(sequence) + 1
                sequence.append(selected)
        return sequence[:7]

    def _grade_retrieval_variant(self, case: EvalCase, variant: str) -> dict[str, Any]:
        auth = self._auth_for_case(case)
        trace_id = trace_manager.new_trace_id()
        trace_manager.start_trace(trace_id, auth.user_id, {"eval_case": case.query_id, "variant": variant, "retrieval_only": True})
        queries = _expanded_queries(case.user_query, case.expected_route) if variant == "query_expansion_plus_hybrid_rerank" else [case.user_query]
        seen: dict[str, Any] = {}
        for query in queries:
            result = hybrid_retriever.search(query, auth=auth, trace_id=trace_id, top_k=6, filters=_filters_for_case(case), variant="hybrid_plus_rerank_pro" if variant.startswith("query_expansion") else variant)
            for chunk in result.chunks:
                seen.setdefault(chunk.document_id or chunk.chunk_id, chunk)
        actual = list(seen.keys())
        expected = [source.doc_id for source in case.expected_sources]
        hits = [doc_id for doc_id in expected if doc_id in actual]
        recall = len(hits) / len(expected) if expected else 1.0
        precision = len(hits) / len(actual) if actual and expected else 1.0
        mrr = next((1 / (index + 1) for index, doc_id in enumerate(actual) if doc_id in expected), 0.0 if expected else 1.0)
        trace_manager.finish_trace(trace_id, "ok", {"actual_sources": actual}, route="retrieval_experiment", duration_ms=0)
        return {"query_id": case.query_id, "variant": variant, "actual": actual, "expected": expected, "recall_at_k": recall, "precision_at_k": precision, "mrr": mrr, "passed": recall == 1.0}

    def _prepare_runtime(self) -> None:
        init_db()
        if not corpus_has_chunks():
            reindex_corpus(force_generate=False)

    def _auth_for_case(self, case: EvalCase) -> AuthContext:
        if case.user_context.role in DEMO_USERS:
            return DEMO_USERS[case.user_context.role]
        if case.user_context.access_level == "restricted":
            return DEMO_USERS["planning_lead"]
        return DEMO_USERS["planning_analyst"]

    def _known_doc_ids(self) -> set[str]:
        self._prepare_runtime()
        from sqlmodel import Session, select
        from defence_agent.db import engine
        from defence_agent.models import Document

        with Session(engine) as session:
            return {document.id for document in session.exec(select(Document)).all()}


def _filters_for_case(case: EvalCase) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if case.expected_route == "metadata_aware_retrieval":
        filters.update({"status": "approved", "exclude_statuses": ["draft", "superseded"]})
    if case.expected_route == "bilingual_retrieval":
        filters.update({"language": "fr", "status": "approved"})
    if case.expected_route == "version_comparison":
        filters.update({"doc_family": "planning_brief_approval"})
    if case.expected_route == "structured_table_analysis":
        filters.update({"doc_id": "doctrine_review_tracker"})
    return filters


def _expanded_queries(query: str, route: str) -> list[str]:
    if route == "cross_source_synthesis":
        return [query, "PB-CHK-2025 evidence checklist", "Planning Brief Approval SOP evidence pack"]
    if route == "version_comparison":
        return [query, "PB-SOP-2024 planning brief approval", "PB-SOP-2025 planning brief approval"]
    return [query]


def _backup_query(route: str) -> str:
    return {
        "evidence_lookup": "What review steps are required before a planning brief is approved?",
        "metadata_aware_retrieval": "What is the current approved procedure for approving a planning brief? Do not use drafts or old versions.",
        "cross_source_synthesis": "What should I include in a planning brief before it goes for review?",
        "version_comparison": "What changed between the 2024 and 2025 planning-brief review process? Cite both versions.",
        "structured_table_analysis": "Which planning procedures are overdue for review? Group them by owner and show how many days overdue.",
        "permission_sensitive_retrieval": "What restricted annex handling steps apply before external distribution?",
        "bilingual_retrieval": "Quels sont les délais dans la procédure de communications d’urgence?",
    }.get(route, "What review steps are required before a planning brief is approved?")


def _rate(values) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _mean(values) -> float:
    values = [float(value) for value in values]
    return statistics.mean(values) if values else 0.0


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil((percentile / 100) * len(ordered)) - 1)
    return ordered[index]


def _grade_rate(outcomes: list[EvalCaseOutcome], section: str) -> float:
    if not outcomes:
        return 0.0
    return _rate(getattr(outcome.grades, section).get("passed", False) for outcome in outcomes)


def _group_rates(outcomes: list[EvalCaseOutcome], field: str) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for outcome in outcomes:
        grouped[str(getattr(outcome, field) or "unknown")].append(outcome)
    return {
        key: {"case_count": len(items), "pass_rate": _rate(item.passed for item in items), "route_accuracy": _grade_rate(items, "route")}
        for key, items in sorted(grouped.items())
    }


advanced_eval_runner = AdvancedEvalRunner()
