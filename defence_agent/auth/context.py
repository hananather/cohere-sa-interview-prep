from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


DEFAULT_PERSONA_ID = "clearance_unclassified"


class AuthContext(BaseModel):
    user_id: str
    role: str
    groups: list[str] = Field(default_factory=list)
    clearance: str
    tenant_id: str = "deftech"


DEMO_USERS: dict[str, AuthContext] = {
    "clearance_unclassified": AuthContext(
        user_id="clearance_unclassified",
        role="clearance_unclassified",
        groups=["demo_users"],
        clearance="unclassified",
        tenant_id="deftech",
    ),
    "clearance_secret": AuthContext(
        user_id="clearance_secret",
        role="clearance_secret",
        groups=["demo_users"],
        clearance="secret",
        tenant_id="deftech",
    ),
    "clearance_top_secret": AuthContext(
        user_id="clearance_top_secret",
        role="clearance_top_secret",
        groups=["demo_users"],
        clearance="top_secret",
        tenant_id="deftech",
    ),
}


@dataclass(frozen=True)
class AclFilter:
    tenant_id: str
    role: str
    groups: tuple[str, ...]
    clearance: str
    allowed_classifications: tuple[str, ...]
