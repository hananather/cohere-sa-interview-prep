from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import pass_fail, source_doc_ids
from defence_agent.evals.schemas import EvalCase


def grade_excluded_sources(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    actual_sources = set(source_doc_ids(response))
    expected_visible = set(case.expected_visible_sources)
    expected_hidden = set(case.expected_hidden_sources or case.expected_excluded_sources)
    disallowed = {item.doc_id for item in case.disallowed_sources}
    visible_missing = sorted(expected_visible - actual_sources)
    hidden_leaks = sorted((expected_hidden | disallowed).intersection(actual_sources))
    passed = not visible_missing and not hidden_leaks
    return pass_fail(
        passed,
        "ACCESS_CONTROL_FAILURE" if hidden_leaks else "RETRIEVAL_MISS",
        expected_visible_sources=sorted(expected_visible),
        expected_hidden_sources=sorted(expected_hidden),
        visible_missing=visible_missing,
        hidden_leaks=hidden_leaks,
    )
