from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import pass_fail
from defence_agent.evals.schemas import EvalCase


def grade_rerank(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    if case.expected_route == "structured_table_analysis":
        return pass_fail(True, "RERANK_MISORDERED", rerank_score_present=False, skipped_for_table_analysis=True)
    scores = [source.get("rerank_score") for source in response.get("sources", [])]
    expected_retrieval = bool(case.expected_sources) and not case.expected_refusal
    score_present = any(score is not None for score in scores) if expected_retrieval else True
    return pass_fail(
        score_present,
        "RERANK_MISORDERED",
        rerank_score_present=score_present,
        scores=[score for score in scores if score is not None],
    )
