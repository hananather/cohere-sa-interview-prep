from __future__ import annotations

import json
from typing import Any

from defence_agent.auth.context import AclFilter, AuthContext


CLASSIFICATION_ORDER = {
    "unclassified": 0,
    "secret": 1,
    "top_secret": 2,
}


TOOL_ALLOWLIST: dict[str, set[str]] = {
    "search_documents": {"clearance_unclassified", "clearance_secret", "clearance_top_secret"},
}


class PolicyDeniedError(PermissionError):
    """Raised when a demo persona is not allowed to call a tool."""


class PolicyEngine:
    def acl_filter(self, auth: AuthContext) -> AclFilter:
        """Return the document tiers this persona may retrieve."""

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
        """Fail closed unless tenant, tier, and optional role checks pass."""

        tenant_id = getattr(chunk, "tenant_id", None)
        if tenant_id and tenant_id != auth.tenant_id:
            return False

        classification = str(getattr(chunk, "classification", "top_secret")).lower()
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
            raise PolicyDeniedError(f"{auth.role} cannot call tool {tool_name}")

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
