from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from defence_agent.auth.context import AuthContext
from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.config import get_settings


ROUTES = {
    "evidence_lookup",
    "grounded_summary",
    "metadata_aware_retrieval",
    "cross_source_synthesis",
    "direct_rag",
    "version_comparison",
    "structured_table_analysis",
    "table_analysis",
    "claim_verification",
    "permission_sensitive_retrieval",
    "bilingual_retrieval",
    "refuse_or_clarify",
    "multi_source_synthesis",
    "ambiguous_query",
    "restricted_access",
    "security_test",
    "human_review",
}


class RouteDecision(BaseModel):
    route: str = Field(pattern="^(evidence_lookup|grounded_summary|metadata_aware_retrieval|cross_source_synthesis|direct_rag|version_comparison|structured_table_analysis|table_analysis|claim_verification|permission_sensitive_retrieval|bilingual_retrieval|refuse_or_clarify|multi_source_synthesis|ambiguous_query|restricted_access|security_test|human_review)$")
    confidence: float = Field(ge=0, le=1)
    reason: str
    needs_tools: list[str] = Field(default_factory=list)
    max_steps: int


class Router:
    def __init__(self) -> None:
        self.settings = get_settings()

    def route(self, query: str, auth: AuthContext, route_override: str | None = None) -> RouteDecision:
        if route_override and route_override != "auto":
            if route_override not in ROUTES:
                return self._decision("human_review", 0.4, f"Unknown route override: {route_override}", ["request_human_review"])
            return self._decision(route_override, 1.0, "Route override selected in demo UI", self._tools_for_route(route_override))

        deterministic = self._deterministic(query, auth)
        if deterministic:
            return deterministic

        optional = self._optional_cohere_route(query)
        if optional:
            return optional

        return self._decision("evidence_lookup", 0.64, "Default route for answerable doctrine question", ["search_documents", "validate_answer_citations"])

    def _deterministic(self, query: str, auth: AuthContext) -> RouteDecision | None:
        lowered = query.lower()
        if _looks_french(lowered):
            return self._decision("bilingual_retrieval", 0.95, "French language query detected", ["search_documents", "validate_answer_citations"])
        if re.search(r"\b(overdue|group by|days overdue|how many|count)\b", lowered):
            return self._decision("structured_table_analysis", 0.97, "Query asks for deterministic table grouping or date math", ["get_table", "run_table_analysis", "validate_answer_citations"])
        if re.search(r"\b(pending approvals|approvals pending|approval register|corrective actions?|annex inventory|due on or before)\b", lowered):
            return self._decision("structured_table_analysis", 0.96, "Query asks for deterministic table lookup or grouping", ["get_table", "run_table_analysis", "validate_answer_citations"])
        if re.search(r"\b(readiness|threshold|below threshold|below the readiness)\b", lowered) and re.search(r"\b(which|group|show|count|table|units?)\b", lowered):
            return self._decision("structured_table_analysis", 0.96, "Query asks for deterministic readiness table analysis", ["get_table", "run_table_analysis", "validate_answer_citations"])
        if (
            "current approved" in lowered
            or "do not use drafts" in lowered
            or "do not cite draft" in lowered
            or "only approved" in lowered
            or "ignore superseded" in lowered
            or "old versions" in lowered
            or ("ignore metadata" in lowered and "draft" in lowered)
            or ("newest" in lowered and "draft" in lowered)
            or ("2026" in lowered and "draft" in lowered and "current" in lowered)
        ):
            return self._decision("metadata_aware_retrieval", 0.96, "Query asks for current approved guidance and excludes drafts or old versions", ["search_documents", "validate_answer_citations"])
        if re.search(r"\b(compare|changed|change|difference|versions?|2024)\b", lowered):
            return self._decision("version_comparison", 0.94, "Version comparison terms matched", ["compare_versions", "validate_citations"])
        if re.search(r"\b(supported|is this statement|verify|claim)\b", lowered):
            return self._decision("claim_verification", 0.94, "Claim verification terms matched", ["search_documents", "validate_answer_citations"])
        if (
            "include in a planning brief" in lowered
            or "before it goes for review" in lowered
            or "pb-chk" in lowered
            or "checklist require" in lowered
            or ("evidence checklist" in lowered and ("support" in lowered or "sop" in lowered))
            or ("interagency emergency" in lowered and "public release" in lowered)
            or ("public release" in lowered and "disclosure checks" in lowered)
            or ("classification" in lowered and ("differ" in lowered or "conflict" in lowered or "errata" in lowered))
        ):
            return self._decision("cross_source_synthesis", 0.92, "Query requires SOP plus referenced checklist evidence", ["search_documents", "follow_references", "validate_answer_citations"])
        if (
            "restricted annex handling" in lowered
            or ("restricted" in lowered and "annex" in lowered)
            or "external distribution" in lowered
            or "data sharing annex" in lowered
            or "ic-annex" in lowered
            or "class-annex" in lowered
            or "restricted data" in lowered
        ):
            return self._decision("permission_sensitive_retrieval", 0.96, "Restricted-source terms matched", ["search_documents", "validate_answer_citations"])
        if "summarize" in lowered or "summary" in lowered:
            return self._decision("grounded_summary", 0.91, "Summarization terms matched", ["search_documents", "validate_answer_citations"])
        if (
            "not covered by any approved document" in lowered
            or "not in any approved document" in lowered
            or "private meeting yesterday" in lowered
            or "private call" in lowered
            or "invent the missing" in lowered
            or "real-world" in lowered
            or "right now" in lowered
        ):
            return self._decision("refuse_or_clarify", 0.91, "Query asks beyond approved evidence", ["request_human_review"])
        if re.search(r"\b(readiness|table|threshold|below|percent|%)\b", lowered):
            return self._decision("table_analysis", 0.94, "Readiness table analysis terms matched", ["search_doctrine", "analyze_table_with_python", "validate_citations"])
        if "scanned manual" in lowered or "field-manual" in lowered or "ocr" in lowered or "2008 scanned" in lowered:
            return self._decision("metadata_aware_retrieval", 0.9, "Legacy scanned source requires currentness metadata handling", ["search_documents", "validate_answer_citations"])
        if re.search(r"\b(restricted|annex\s*b)\b", lowered):
            return self._decision("restricted_access", 0.96, "Restricted annex terms matched", ["search_doctrine", "validate_citations"])
        if re.search(r"\b(poisoned|ignore instructions|ignore all previous|test document|injection)\b", lowered):
            return self._decision("security_test", 0.95, "Security test terms matched", ["search_doctrine", "validate_citations"])
        if "approval process" in lowered and "cross-unit" not in lowered and "planning" not in lowered:
            return self._decision("ambiguous_query", 0.86, "Approval process is underspecified", ["request_human_review"])
        if len(lowered.split()) <= 3:
            return self._decision("refuse_or_clarify", 0.7, "Query is too short to safely retrieve", ["request_human_review"])
        if "conflict" in lowered or "sources" in lowered:
            return self._decision("cross_source_synthesis", 0.75, "Multi-source synthesis terms matched", ["search_documents", "validate_answer_citations"])
        if "review steps" in lowered or "planning brief" in lowered or "approval" in lowered:
            return self._decision("evidence_lookup", 0.88, "Default evidence lookup for planning procedure question", ["search_documents", "validate_answer_citations"])
        return None

    def _optional_cohere_route(self, query: str) -> RouteDecision | None:
        if self.settings.use_mock_cohere:
            return None
        try:
            prompt = (
                "Classify this query into one route: evidence_lookup, grounded_summary, metadata_aware_retrieval, "
                "cross_source_synthesis, version_comparison, structured_table_analysis, claim_verification, "
                "permission_sensitive_retrieval, bilingual_retrieval, refuse_or_clarify. "
                "Return JSON with route, confidence, reason.\nQuery: "
                + query
            )
            answer = cohere_gateway.generate_answer(prompt, [], "router")
            match = re.search(r"\{.*\}", answer, re.DOTALL)
            if not match:
                return None
            payload = json.loads(match.group(0))
            route = payload.get("route", "direct_rag")
            return self._decision(route, float(payload.get("confidence", 0.55)), str(payload.get("reason", "Cohere router")), self._tools_for_route(route))
        except Exception:
            return None

    def _decision(self, route: str, confidence: float, reason: str, needs_tools: list[str] | None = None) -> RouteDecision:
        return RouteDecision(
            route=route if route in ROUTES else "human_review",
            confidence=confidence,
            reason=reason,
            needs_tools=needs_tools or self._tools_for_route(route),
            max_steps=self.settings.max_agent_steps,
        )

    def _tools_for_route(self, route: str) -> list[str]:
        return {
            "evidence_lookup": ["search_documents", "validate_answer_citations"],
            "grounded_summary": ["search_documents", "validate_answer_citations"],
            "metadata_aware_retrieval": ["search_documents", "validate_answer_citations"],
            "cross_source_synthesis": ["search_documents", "follow_references", "validate_answer_citations"],
            "direct_rag": ["search_doctrine", "validate_citations"],
            "version_comparison": ["compare_versions", "validate_citations"],
            "structured_table_analysis": ["get_table", "run_table_analysis", "validate_answer_citations"],
            "table_analysis": ["search_doctrine", "analyze_table_with_python", "validate_citations"],
            "claim_verification": ["search_documents", "validate_answer_citations"],
            "permission_sensitive_retrieval": ["search_documents", "validate_answer_citations"],
            "bilingual_retrieval": ["search_documents", "validate_answer_citations"],
            "refuse_or_clarify": ["request_human_review"],
            "multi_source_synthesis": ["search_doctrine", "validate_citations"],
            "ambiguous_query": ["request_human_review"],
            "restricted_access": ["search_doctrine", "validate_citations"],
            "security_test": ["search_doctrine", "validate_citations"],
            "human_review": ["request_human_review"],
        }.get(route, ["request_human_review"])


router = Router()


def _looks_french(lowered: str) -> bool:
    return bool(
        re.search(r"\b(french|francais|francaise|quels|quelles|delais|delai|communications d'urgence)\b", lowered)
        or any(character in lowered for character in "àâçéèêëîïôûùüÿñæœ")
    )
