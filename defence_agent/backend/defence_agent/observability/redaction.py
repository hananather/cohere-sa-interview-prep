from __future__ import annotations

import hashlib
import os
import re
from typing import Any

from defence_agent.auth.context import AuthContext
from defence_agent.auth.policy import CLASSIFICATION_ORDER


SECRET_PATTERNS = [
    re.compile(r"(COHERE_API_KEY\s*=\s*)[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\b0[A-Za-z0-9]{32,}\b"),
]
RESTRICTED_REDACTION = "[restricted content redacted]"
SECRET_REDACTION = "[secret redacted]"
MAX_SUMMARY_CHARS = 900


def debug_full_trace_enabled() -> bool:
    return os.getenv("DEFTECH_DEBUG_FULL_TRACE", "").lower() in {"1", "true", "yes"}


def can_view_classification(auth: AuthContext, classification: str | None) -> bool:
    if not classification:
        return True
    level = CLASSIFICATION_ORDER.get(str(classification).lower(), 99)
    clearance = CLASSIFICATION_ORDER.get(auth.clearance, -1)
    return level <= clearance


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def redact_text(value: str, *, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: (match.group(1) if match.groups() else "") + SECRET_REDACTION, redacted)
    if not debug_full_trace_enabled() and len(redacted) > max_chars:
        redacted = redacted[:max_chars].rstrip() + "..."
    return redacted


def redact_value(value: Any, auth: AuthContext, *, classification: str | None = None) -> Any:
    if isinstance(value, str):
        if classification and not can_view_classification(auth, classification) and not debug_full_trace_enabled():
            return RESTRICTED_REDACTION
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item, auth, classification=classification) for item in value]
    if isinstance(value, dict):
        return redact_mapping(value, auth)
    return value


def redact_mapping(payload: dict[str, Any], auth: AuthContext) -> dict[str, Any]:
    classification = str(
        payload.get("classification")
        or payload.get("access_level")
        or payload.get("access")
        or payload.get("metadata", {}).get("classification")
        or ""
    )
    visible = can_view_classification(auth, classification) or debug_full_trace_enabled()
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        key_lower = str(key).lower()
        if key_lower in {"text", "summary", "snippet", "chunk_text", "prompt", "context", "source_text"} and not visible:
            if isinstance(value, str):
                redacted[f"{key}_hash"] = content_hash(value)
            redacted[key] = RESTRICTED_REDACTION
            continue
        if "key" in key_lower or "secret" in key_lower or "token" in key_lower:
            redacted[key] = SECRET_REDACTION
            continue
        redacted[key] = redact_value(value, auth, classification=classification if not visible else None)
    return redacted


def redact_trace(trace: dict[str, Any], auth: AuthContext) -> dict[str, Any]:
    return redact_mapping(trace, auth)
