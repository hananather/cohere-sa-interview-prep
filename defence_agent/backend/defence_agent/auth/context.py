from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException, status
from pydantic import BaseModel, Field


class AuthContext(BaseModel):
    user_id: str
    role: str
    groups: list[str] = Field(default_factory=list)
    clearance: str
    tenant_id: str = "deftech"


DEMO_USERS: dict[str, AuthContext] = {
    "planning_analyst": AuthContext(
        user_id="planning_analyst",
        role="planning_analyst",
        groups=["central_planning", "readiness_review"],
        clearance="public_internal",
        tenant_id="deftech",
    ),
    "planning_lead": AuthContext(
        user_id="planning_lead",
        role="planning_lead",
        groups=["central_planning", "readiness_review", "restricted_annex"],
        clearance="restricted",
        tenant_id="deftech",
    ),
    "auditor": AuthContext(
        user_id="auditor",
        role="auditor",
        groups=["audit"],
        clearance="public_internal",
        tenant_id="deftech",
    ),
    "admin": AuthContext(
        user_id="admin",
        role="admin",
        groups=["admin"],
        clearance="public_internal",
        tenant_id="deftech",
    ),
}


DEMO_PERSONA_PROFILES: dict[str, dict[str, Any]] = {
    "planning_analyst": {
        "persona_id": "planning_analyst",
        "display_name": "Alex Chen",
        "role_label": "Planning Analyst",
        "access_level": "public_internal",
        "allowed_doc_access": ["public_internal", "protected"],
        "can_view_audit_metadata": False,
        "can_view_restricted_content": False,
        "notes": "Uses approved public-internal doctrine and sees restricted gaps as refusals or partial answers.",
    },
    "planning_lead": {
        "persona_id": "planning_lead",
        "display_name": "Morgan Singh",
        "role_label": "Doctrine Steward",
        "access_level": "restricted",
        "allowed_doc_access": ["public_internal", "protected", "restricted"],
        "can_view_audit_metadata": False,
        "can_view_restricted_content": True,
        "notes": "Can inspect restricted doctrine content when the document ACL grants access.",
    },
    "auditor": {
        "persona_id": "auditor",
        "display_name": "Priya Rao",
        "role_label": "Security Auditor",
        "access_level": "security_admin",
        "allowed_doc_access": ["metadata_only_for_restricted", "public_internal", "protected"],
        "can_view_audit_metadata": True,
        "can_view_restricted_content": False,
        "notes": "Can inspect policy decisions and audit metadata without automatically seeing restricted doctrine text.",
    },
    "admin": {
        "persona_id": "admin",
        "display_name": "Sam Rivera",
        "role_label": "Platform Admin",
        "access_level": "admin",
        "allowed_doc_access": ["public_internal", "protected"],
        "can_view_audit_metadata": True,
        "can_view_restricted_content": False,
        "notes": "Administers the platform. System administration privileges do not imply doctrine content access.",
    },
}


def _decode_demo_jwt(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None


async def get_auth_context(
    x_demo_user: str | None = Header(default=None, alias="X-Demo-User"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AuthContext:
    user_key = x_demo_user
    if not user_key and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.startswith("demo:"):
            user_key = token.replace("demo:", "", 1)
        elif scheme.lower() == "bearer":
            claims = _decode_demo_jwt(token)
            if claims:
                user_key = str(claims.get("sub") or claims.get("user_id") or "")

    if not user_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No demo user provided")
    if user_key not in DEMO_USERS:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Unknown demo user: {user_key}")
    return DEMO_USERS[user_key]


@dataclass(frozen=True)
class AclFilter:
    tenant_id: str
    role: str
    groups: tuple[str, ...]
    clearance: str
    allowed_classifications: tuple[str, ...]
