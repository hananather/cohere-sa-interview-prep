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
        clearance="protected",
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
        clearance="protected",
        tenant_id="deftech",
    ),
    "admin": AuthContext(
        user_id="admin",
        role="admin",
        groups=["central_planning", "readiness_review", "restricted_annex", "admin"],
        clearance="restricted",
        tenant_id="deftech",
    ),
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
