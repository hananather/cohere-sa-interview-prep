from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException, status

from defence_agent.auth.context import AclFilter, AuthContext


CLASSIFICATION_ORDER = {
    "public": 0,
    "official": 1,
    "public_internal": 2,
    "protected": 2,
    "restricted": 3,
    "secret_placeholder": 4,
}


TOOL_ALLOWLIST: dict[str, set[str]] = {
    "search_doctrine": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "search_documents": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "get_document_sections": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "follow_references": {"planning_analyst", "planning_lead", "admin"},
    "compare_versions": {"planning_analyst", "planning_lead", "admin"},
    "compare_document_versions": {"planning_analyst", "planning_lead", "admin"},
    "analyze_table_with_python": {"planning_analyst", "planning_lead", "admin"},
    "get_table": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "run_table_analysis": {"planning_analyst", "planning_lead", "admin"},
    "get_document_registry_status": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "validate_citations": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "validate_answer_citations": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "request_human_review": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "log_feedback": {"planning_analyst", "planning_lead", "auditor", "admin"},
    "admin_reindex": {"admin"},
}


class PolicyEngine:
    def acl_filter(self, auth: AuthContext) -> AclFilter:
        max_level = CLASSIFICATION_ORDER.get(auth.clearance, -1)
        allowed = tuple(name for name, level in CLASSIFICATION_ORDER.items() if level <= max_level)
        return AclFilter(
            tenant_id=auth.tenant_id,
            role=auth.role,
            groups=tuple(auth.groups),
            clearance=auth.clearance,
            allowed_classifications=allowed,
        )

    def can_access_chunk(self, auth: AuthContext, chunk: Any) -> bool:
        tenant_id = getattr(chunk, "tenant_id", None)
        if tenant_id and tenant_id != auth.tenant_id:
            return False

        classification = str(getattr(chunk, "classification", "restricted")).lower()
        if CLASSIFICATION_ORDER.get(classification, 99) > CLASSIFICATION_ORDER.get(auth.clearance, -1):
            return False

        allowed_roles = self._roles_from_chunk(chunk)
        if "all" in allowed_roles or auth.role in allowed_roles:
            return True

        allowed_groups = {role.replace("group:", "") for role in allowed_roles if role.startswith("group:")}
        if allowed_groups and allowed_groups.intersection(auth.groups):
            return True

        return False

    def can_call_tool(self, auth: AuthContext, tool_name: str) -> bool:
        return auth.role in TOOL_ALLOWLIST.get(tool_name, set())

    def enforce_tool_call(self, auth: AuthContext, tool_name: str) -> None:
        if not self.can_call_tool(auth, tool_name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{auth.role} cannot call tool {tool_name}",
            )

    def tool_allowlist_for(self, auth: AuthContext) -> list[str]:
        return sorted(tool for tool, roles in TOOL_ALLOWLIST.items() if auth.role in roles)

    def _roles_from_chunk(self, chunk: Any) -> set[str]:
        raw = getattr(chunk, "allowed_roles_json", "[]")
        if isinstance(raw, list):
            return set(raw)
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return set()
        return {str(item) for item in parsed}


policy_engine = PolicyEngine()
