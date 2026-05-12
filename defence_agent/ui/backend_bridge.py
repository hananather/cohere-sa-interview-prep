"""Subprocess bridge between Streamlit and the live ADK backend.

The Cohere SDK calls used by the backend are synchronous. Running the backend
turn in a short-lived subprocess keeps the Streamlit process responsive and
lets the UI enforce a real timeout instead of spinning forever.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from defence_agent.session import AgentTurnResult


DEFAULT_TIMEOUT_SECONDS = int(os.getenv("DEFENCE_AGENT_UI_TIMEOUT_SECONDS", "120"))
ROOT = Path(__file__).resolve().parents[2]


class BackendRunError(RuntimeError):
    """Raised when the backend worker exits without a valid agent result."""


def run_turn_in_subprocess(
    *,
    query: str,
    persona_id: str,
    user_id: str,
    session_id: str | None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    progress: Callable[[str], None] | None = None,
) -> AgentTurnResult:
    """Run one live ``run_turn`` call in an isolated Python subprocess."""

    payload = {
        "query": query,
        "persona_id": persona_id,
        "user_id": user_id,
        "session_id": session_id,
    }
    if progress:
        progress("Live backend call started. Waiting for a structured worker result.")

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "defence_agent.ui.backend_worker"],
            cwd=str(ROOT),
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run has already killed the child and drained stdout/stderr.
        detail = _safe_error_detail(exc.stderr or "")
        message = f"Defence Agent backend exceeded {timeout_seconds}s."
        if detail:
            message += f" Last worker error: {detail}"
        raise TimeoutError(message) from exc

    response = _worker_response_from_stdout(completed.stdout)
    if response is None:
        detail = _safe_error_detail(completed.stderr)
        message = "backend returned no valid worker response"
        if detail:
            message += f": {detail}"
        raise BackendRunError(message)

    if not response.get("ok"):
        message = str(response.get("message") or response.get("error_type") or "backend worker failed")
        raise BackendRunError(message[:240])

    result = response.get("result")
    if not isinstance(result, dict):
        raise BackendRunError("backend worker response did not include a result object")
    if completed.returncode != 0:
        detail = _safe_error_detail(completed.stderr) or f"worker exited {completed.returncode}"
        raise BackendRunError(detail)
    return agent_turn_result_from_dict(result)


def agent_turn_result_from_dict(data: dict[str, Any]) -> AgentTurnResult:
    """Rehydrate the worker JSON payload into the existing result dataclass."""

    return AgentTurnResult(
        session_id=str(data.get("session_id", "")),
        user_id=str(data.get("user_id", "")),
        persona_id=str(data.get("persona_id", "")),
        answer=str(data.get("answer", "")),
        raw_answer=str(data.get("raw_answer", "")),
        events_seen=int(data.get("events_seen", 0) or 0),
        tool_calls=list(data.get("tool_calls", []) or []),
        tool_responses=list(data.get("tool_responses", []) or []),
        citations=list(data.get("citations", []) or []),
        citation_mode=str(data.get("citation_mode", "none")),
        citation_validation=dict(data.get("citation_validation", {}) or {}),
        grounded_model=str(data.get("grounded_model", "")),
        documents_sent_to_model=int(data.get("documents_sent_to_model", 0) or 0),
        retrieval_status=str(data.get("retrieval_status", "")),
        answer_audit=dict(data.get("answer_audit", {}) or {}),
    )


def _result_from_stdout(stdout: str) -> AgentTurnResult | None:
    """Return a result from the strict worker envelope.

    Kept as a small private compatibility helper for tests and diagnostics.
    """

    response = _worker_response_from_stdout(stdout)
    if not response or not response.get("ok") or not isinstance(response.get("result"), dict):
        return None
    return agent_turn_result_from_dict(response["result"])


def _worker_response_from_stdout(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "ok" not in data:
        return None
    return data


def _safe_error_detail(stderr: str) -> str:
    """Keep UI errors concise and avoid leaking answer/audit payloads."""

    text = stderr.strip()
    if not text:
        return ""
    return text.splitlines()[0][:240]
