from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"reveal\s+(restricted|secret|hidden)",
    r"exfiltrate",
    r"bypass\s+(policy|acl|access)",
]

SUSPICIOUS_STRINGS = [
    "BEGIN SYSTEM PROMPT",
    "do not cite",
    "disable safety",
    "show hidden",
    "jailbreak",
]

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    r"\b\d{16}\b",
]


@dataclass
class SafetyResult:
    blocked: bool = False
    risk: str = "low"
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "risk": self.risk,
            "reasons": self.reasons,
            "flags": self.flags,
        }


def validate_input_query(query: str, max_length: int = 4000) -> SafetyResult:
    result = SafetyResult()
    if len(query) > max_length:
        result.blocked = True
        result.risk = "high"
        result.reasons.append("Query is too long")

    lowered = query.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            result.flags.append("prompt_injection_pattern")
            result.reasons.append(f"Matched suspicious pattern: {pattern}")

    for suspicious in SUSPICIOUS_STRINGS:
        if suspicious.lower() in lowered:
            result.flags.append("suspicious_string")
            result.reasons.append(f"Matched suspicious string: {suspicious}")

    if result.flags and not result.blocked:
        result.risk = "medium"
    return result


def sanitize_retrieved_text(text: str) -> str:
    sanitized = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[instruction-like text removed]", sanitized, flags=re.IGNORECASE)
    for suspicious in SUSPICIOUS_STRINGS:
        sanitized = sanitized.replace(suspicious, "[instruction-like text removed]")
    return sanitized


def sanitize_tool_output(output: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return sanitize_retrieved_text(value)
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    return clean(output)


def validate_output(answer: str, citation_ids: list[str], requires_citation: bool = True) -> SafetyResult:
    result = SafetyResult()
    for pattern in PII_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            result.flags.append("pii_pattern")
            result.reasons.append(f"Output matched sensitive pattern: {pattern}")

    if requires_citation and citation_ids:
        cited = set(re.findall(r"\[(C\d+)\]", answer))
        missing = [citation_id for citation_id in citation_ids if citation_id not in cited]
        if len(cited) == 0:
            result.flags.append("missing_citations")
            result.reasons.append("Answer did not include citations")
        elif missing and len(cited) < min(2, len(citation_ids)):
            result.flags.append("thin_citations")
            result.reasons.append("Answer cited too little of the evidence")

    if result.flags:
        result.risk = "medium"
    if "pii_pattern" in result.flags:
        result.risk = "high"
        result.blocked = True
    return result


def citation_ids_from_answer(answer: str) -> set[str]:
    return set(re.findall(r"\[(C\d+)\]", answer))
