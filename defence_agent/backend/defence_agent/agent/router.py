from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from defence_agent.auth.context import AuthContext
from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.config import get_settings


ROUTES = {
    "direct_rag",
    "version_comparison",
    "table_analysis",
    "multi_source_synthesis",
    "ambiguous_query",
    "restricted_access",
    "security_test",
    "human_review",
}


class RouteDecision(BaseModel):
    route: str = Field(pattern="^(direct_rag|version_comparison|table_analysis|multi_source_synthesis|ambiguous_query|restricted_access|security_test|human_review)$")
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

        return self._decision("direct_rag", 0.64, "Default route for answerable doctrine question", ["search_doctrine", "validate_citations"])

    def _deterministic(self, query: str, auth: AuthContext) -> RouteDecision | None:
        lowered = query.lower()
        if re.search(r"\b(compare|changed|change|version|2024|2025)\b", lowered):
            return self._decision("version_comparison", 0.94, "Version comparison terms matched", ["compare_versions", "validate_citations"])
        if re.search(r"\b(readiness|table|threshold|below|percent|%)\b", lowered):
            return self._decision("table_analysis", 0.94, "Readiness table analysis terms matched", ["search_doctrine", "analyze_table_with_python", "validate_citations"])
        if re.search(r"\b(restricted|annex\s*b)\b", lowered):
            return self._decision("restricted_access", 0.96, "Restricted annex terms matched", ["search_doctrine", "validate_citations"])
        if re.search(r"\b(poisoned|ignore instructions|ignore all previous|test document|injection)\b", lowered):
            return self._decision("security_test", 0.95, "Security test terms matched", ["search_doctrine", "validate_citations"])
        if "approval process" in lowered and "cross-unit" not in lowered and "planning" not in lowered:
            return self._decision("ambiguous_query", 0.86, "Approval process is underspecified", ["request_human_review"])
        if len(lowered.split()) <= 3:
            return self._decision("ambiguous_query", 0.7, "Query is too short to safely retrieve", ["request_human_review"])
        if "conflict" in lowered or "sources" in lowered:
            return self._decision("multi_source_synthesis", 0.75, "Multi-source synthesis terms matched", ["search_doctrine", "validate_citations"])
        return None

    def _optional_cohere_route(self, query: str) -> RouteDecision | None:
        if self.settings.use_mock_cohere:
            return None
        try:
            prompt = (
                "Classify this query into one route: direct_rag, version_comparison, table_analysis, "
                "multi_source_synthesis, ambiguous_query, restricted_access, security_test, human_review. "
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
            "direct_rag": ["search_doctrine", "validate_citations"],
            "version_comparison": ["compare_versions", "validate_citations"],
            "table_analysis": ["search_doctrine", "analyze_table_with_python", "validate_citations"],
            "multi_source_synthesis": ["search_doctrine", "validate_citations"],
            "ambiguous_query": ["request_human_review"],
            "restricted_access": ["search_doctrine", "validate_citations"],
            "security_test": ["search_doctrine", "validate_citations"],
            "human_review": ["request_human_review"],
        }.get(route, ["request_human_review"])


router = Router()
