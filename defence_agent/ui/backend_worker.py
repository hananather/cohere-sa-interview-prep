"""Worker process for one live Defence Agent turn."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from typing import Any

from defence_agent.session import run_turn


async def _run(payload: dict[str, Any]) -> dict[str, Any]:
    result = await run_turn(
        str(payload["query"]),
        persona_id=str(payload["persona_id"]),
        user_id=str(payload["user_id"]),
        session_id=payload.get("session_id") or None,
    )
    return asdict(result)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        result = asyncio.run(_run(payload))
        _write_response({"ok": True, "result": result})
        # The worker is an isolated demo subprocess. Exit without waiting for
        # SDK-owned background threads so Streamlit gets a deterministic finish.
        os._exit(0)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        _write_response({"ok": False, "error_type": type(exc).__name__, "message": message[:240]})
        sys.stderr.write(message[:240])
        sys.stderr.flush()
        os._exit(1)


def _write_response(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
