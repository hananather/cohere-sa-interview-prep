from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import citation_doc_ids, pass_fail, source_doc_ids
from defence_agent.evals.schemas import EvalCase


def grade_citations(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    citations = response.get("citations", [])
    cited_docs = citation_doc_ids(response)
    source_docs = set(source_doc_ids(response))
    present_ok = bool(citations) if case.citation_required else not bool(citations) or case.expected_route == "permission_sensitive_retrieval"
    resolve_ok = all(doc_id in source_docs for doc_id in cited_docs)
    unauthorized = [
        citation.get("document_id")
        for citation in citations
        if case.user_context.access_level == "public_internal" and citation.get("classification") == "restricted"
    ]
    status_ok = True
    if case.expected_route == "metadata_aware_retrieval":
        status_ok = all(citation.get("status") == "approved" for citation in citations)
    validation = response.get("safety", {}).get("citation_validation", {}).get("data", {})
    validation_ok = bool(validation.get("valid", True)) if case.citation_required else True
    passed = present_ok and resolve_ok and not unauthorized and status_ok and validation_ok
    return pass_fail(
        passed,
        "CITATION_MISSING" if not present_ok else "CITATION_UNSUPPORTED",
        citation_required=case.citation_required,
        citation_count=len(citations),
        present_ok=present_ok,
        resolve_ok=resolve_ok,
        unauthorized_citations=unauthorized,
        status_ok=status_ok,
        citation_validation_ok=validation_ok,
    )
