from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import HTTPException

from defence_agent.auth.context import AuthContext
from defence_agent.auth.policy import policy_engine
from defence_agent.observability.redaction import redact_mapping
from defence_agent.observability.tracing import trace_manager


class AgentLifecyclePlugin(Protocol):
    name: str

    def before_tool(self, trace_id: str, tool_name: str, payload: dict[str, Any], auth: AuthContext) -> None:
        ...

    def after_tool(self, trace_id: str, tool_name: str, output: dict[str, Any], auth: AuthContext) -> None:
        ...

    def on_tool_error(self, trace_id: str, tool_name: str, error: str, auth: AuthContext) -> None:
        ...


@dataclass(frozen=True)
class TracePlugin:
    name: str = "trace"

    def before_tool(self, trace_id: str, tool_name: str, payload: dict[str, Any], auth: AuthContext) -> None:
        trace_manager.add_span(
            trace_id,
            "before_tool",
            {"tool": tool_name, "payload": redact_mapping(payload, auth)},
        )

    def after_tool(self, trace_id: str, tool_name: str, output: dict[str, Any], auth: AuthContext) -> None:
        trace_manager.add_span(
            trace_id,
            "after_tool",
            {"tool": tool_name, "output_keys": sorted(output.keys())},
        )

    def on_tool_error(self, trace_id: str, tool_name: str, error: str, auth: AuthContext) -> None:
        trace_manager.add_span(trace_id, "on_tool_error", {"tool": tool_name}, status="error", error=error)


@dataclass(frozen=True)
class PolicyGuardPlugin:
    name: str = "policy_guard"

    def before_tool(self, trace_id: str, tool_name: str, payload: dict[str, Any], auth: AuthContext) -> None:
        allowed = policy_engine.can_call_tool(auth, tool_name)
        trace_manager.add_span(
            trace_id,
            "before_tool_policy",
            {
                "tool": tool_name,
                "persona_id": auth.user_id,
                "role": auth.role,
                "decision": "allow" if allowed else "block",
                "reason": None if allowed else f"{auth.role} is not allowed to call {tool_name}",
            },
            status="ok" if allowed else "blocked",
        )
        if not allowed:
            raise HTTPException(status_code=403, detail=f"{auth.role} cannot call tool {tool_name}")

    def after_tool(self, trace_id: str, tool_name: str, output: dict[str, Any], auth: AuthContext) -> None:
        return None

    def on_tool_error(self, trace_id: str, tool_name: str, error: str, auth: AuthContext) -> None:
        return None


@dataclass(frozen=True)
class MetricsPlugin:
    name: str = "metrics"

    def before_tool(self, trace_id: str, tool_name: str, payload: dict[str, Any], auth: AuthContext) -> None:
        return None

    def after_tool(self, trace_id: str, tool_name: str, output: dict[str, Any], auth: AuthContext) -> None:
        trace_manager.add_span(
            trace_id,
            "tool_metrics",
            {
                "tool": tool_name,
                "ok": True,
                "result_size": len(str(output)),
            },
        )

    def on_tool_error(self, trace_id: str, tool_name: str, error: str, auth: AuthContext) -> None:
        trace_manager.add_span(trace_id, "tool_metrics", {"tool": tool_name, "ok": False}, status="error", error=error)


@dataclass(frozen=True)
class RedactionPlugin:
    name: str = "redaction"

    def before_tool(self, trace_id: str, tool_name: str, payload: dict[str, Any], auth: AuthContext) -> None:
        return None

    def after_tool(self, trace_id: str, tool_name: str, output: dict[str, Any], auth: AuthContext) -> None:
        return None

    def on_tool_error(self, trace_id: str, tool_name: str, error: str, auth: AuthContext) -> None:
        return None


@dataclass(frozen=True)
class FeedbackPlugin:
    name: str = "feedback"

    def before_tool(self, trace_id: str, tool_name: str, payload: dict[str, Any], auth: AuthContext) -> None:
        return None

    def after_tool(self, trace_id: str, tool_name: str, output: dict[str, Any], auth: AuthContext) -> None:
        if tool_name == "log_feedback":
            trace_manager.add_span(trace_id, "feedback_captured", {"feedback_id": output.get("feedback_id")})

    def on_tool_error(self, trace_id: str, tool_name: str, error: str, auth: AuthContext) -> None:
        return None


class PluginManager:
    def __init__(self, plugins: list[AgentLifecyclePlugin] | None = None) -> None:
        self.plugins = plugins or [
            PolicyGuardPlugin(),
            TracePlugin(),
            MetricsPlugin(),
            RedactionPlugin(),
            FeedbackPlugin(),
        ]

    def before_tool(self, trace_id: str, tool_name: str, payload: dict[str, Any], auth: AuthContext) -> None:
        for plugin in self.plugins:
            plugin.before_tool(trace_id, tool_name, payload, auth)

    def after_tool(self, trace_id: str, tool_name: str, output: dict[str, Any], auth: AuthContext) -> None:
        for plugin in self.plugins:
            plugin.after_tool(trace_id, tool_name, output, auth)

    def on_tool_error(self, trace_id: str, tool_name: str, error: str, auth: AuthContext) -> None:
        for plugin in self.plugins:
            plugin.on_tool_error(trace_id, tool_name, error, auth)


plugin_manager = PluginManager()
