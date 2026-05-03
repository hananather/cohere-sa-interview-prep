from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing as mp
import time
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session

from defence_agent.db import engine
from defence_agent.models import SandboxRun


BLOCKED_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
}

BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "pathlib",
    "shutil",
    "importlib",
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


class SandboxValidationError(ValueError):
    pass


def validate_code(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxValidationError("Import statements are blocked. Use preloaded pd, np, math, and statistics.")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES.union(BLOCKED_MODULES):
            raise SandboxValidationError(f"Blocked name: {node.id}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                raise SandboxValidationError(f"Blocked call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
                raise SandboxValidationError("Dunder attribute calls are blocked")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in BLOCKED_MODULES:
                raise SandboxValidationError(f"Blocked module attribute: {node.value.id}.{node.attr}")
            if node.attr.startswith("__"):
                raise SandboxValidationError("Dunder attributes are blocked")


def run_sandboxed_python(
    code: str,
    input_data: dict[str, Any],
    trace_id: str,
    user_id: str,
    timeout_seconds: int = 5,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_hash = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode("utf-8")).hexdigest()
    output: dict[str, Any] | None = None
    error: str | None = None

    try:
        validate_code(code)
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        context = mp.get_context(start_method)
        queue: mp.Queue = context.Queue()
        process = context.Process(target=_worker, args=(code, input_data, queue))
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(1)
            raise TimeoutError(f"Sandbox timed out after {timeout_seconds} seconds")
        if queue.empty():
            raise RuntimeError("Sandbox returned no result")
        payload = queue.get()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        output = payload["output"]
        return {"ok": True, "output": output, "input_hash": input_hash}
    except Exception as exc:
        error = str(exc)
        return {"ok": False, "error": error, "input_hash": input_hash}
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        with Session(engine) as session:
            session.add(
                SandboxRun(
                    trace_id=trace_id,
                    user_id=user_id,
                    code=code,
                    input_hash=input_hash,
                    output_json=json.dumps(output) if output is not None else None,
                    error=error,
                    duration_ms=duration_ms,
                )
            )
            session.commit()


def _worker(code: str, input_data: dict[str, Any], queue: mp.Queue) -> None:
    import math
    import statistics

    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "input_data": input_data,
        "pd": pd,
        "np": np,
        "math": math,
        "statistics": statistics,
        "result": None,
    }
    try:
        exec(compile(code, "<defence-agent-sandbox>", "exec"), namespace, namespace)
        result = namespace.get("result")
        json.dumps(result)
        queue.put({"output": result})
    except Exception as exc:
        queue.put({"error": str(exc)})
