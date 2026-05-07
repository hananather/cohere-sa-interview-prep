from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import disallowed_doc_ids, expected_doc_ids, pass_fail, source_doc_ids
from defence_agent.evals.schemas import EvalCase


def grade_retrieval(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = expected_doc_ids(case)
    actual = source_doc_ids(response)
    disallowed = disallowed_doc_ids(case)
    expected_hits = [doc_id for doc_id in expected if doc_id in actual]
    disallowed_hits = [doc_id for doc_id in disallowed if doc_id in actual]
    recall = len(expected_hits) / len(expected) if expected else 1.0
    precision = len(expected_hits) / len(actual) if actual and expected else (1.0 if not actual else 0.0)
    mrr = _mrr(expected, actual)
    sufficiency = recall == 1.0 if expected else case.expected_refusal or not case.grader_config.retrieval
    passed = sufficiency and not disallowed_hits
    return pass_fail(
        passed,
        "RETRIEVAL_MISS" if not sufficiency else "FILTER_MISSING",
        expected_sources=expected,
        actual_sources=actual,
        disallowed_sources=disallowed,
        disallowed_hits=disallowed_hits,
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        context_sufficient=sufficiency,
        candidate_count=_candidate_count(trace),
    )


def _mrr(expected: list[str], actual: list[str]) -> float:
    for index, doc_id in enumerate(actual, start=1):
        if doc_id in expected:
            return 1.0 / index
    return 0.0 if expected else 1.0


def _candidate_count(trace: dict[str, Any] | None) -> int | None:
    if not trace:
        return None
    for span in trace.get("spans", []):
        if span.get("name") == "context_assembled":
            attrs = span.get("attributes", {}) or span.get("attributes_json", {})
            if isinstance(attrs, dict):
                return attrs.get("source_count")
    return None
