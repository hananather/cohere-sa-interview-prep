from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import pass_fail, source_doc_ids
from defence_agent.evals.schemas import EvalCase


def grade_filters(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    sources = response.get("sources", [])
    actual_ids = source_doc_ids(response)
    status_ok = True
    language_ok = True
    access_ok = True
    for source in sources:
        if "metadata_filter_required" in case.complexity_dimensions and case.expected_route in {"metadata_aware_retrieval", "evidence_lookup"}:
            status_ok = status_ok and source.get("status") == "approved"
        if "bilingual_or_crosslingual" in case.complexity_dimensions:
            language_ok = language_ok and source.get("language") == "fr"
        if case.user_context.access_level == "public_internal":
            access_ok = access_ok and source.get("classification") != "restricted"
    disallowed_hits = [item.doc_id for item in case.disallowed_sources if item.doc_id in actual_ids]
    passed = status_ok and language_ok and access_ok and not disallowed_hits
    return pass_fail(
        passed,
        "ACCESS_CONTROL_FAILURE" if not access_ok else "FILTER_MISSING",
        status_filter_correct=status_ok,
        language_filter_correct=language_ok,
        access_filter_correct=access_ok,
        disallowed_hits=disallowed_hits,
    )
