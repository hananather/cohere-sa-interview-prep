from __future__ import annotations

import time
from typing import Any

from defence_agent.agent.router import router
from defence_agent.agent.workflows import WORKFLOWS
from defence_agent.auth.context import AuthContext
from defence_agent.models import AskRequest, AskResponse, SourceChunk
from defence_agent.observability.metrics import ROUTE_COUNT, SAFETY_BLOCKS
from defence_agent.observability.tracing import trace_manager
from defence_agent.safety import validate_input_query, validate_output
from defence_agent.tools.registry import tool_registry


class AgentService:
    def handle(self, request: AskRequest, auth: AuthContext, trace_id: str) -> AskResponse:
        started = time.perf_counter()
        trace_manager.add_span(trace_id, "input_safety_checked", {"query_length": len(request.query)})
        input_safety = validate_input_query(request.query)
        if input_safety.blocked:
            SAFETY_BLOCKS.labels(stage="input").inc()
            return AskResponse(
                trace_id=trace_id,
                route="human_review",
                answer="I cannot process this query safely. Please shorten or rephrase it.",
                safety={"input": input_safety.as_dict()},
                needs_human_review=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        decision = router.route(request.query, auth, request.route_override)
        ROUTE_COUNT.labels(route=decision.route).inc()
        trace_manager.set_route(trace_id, decision.route)
        trace_manager.add_span(trace_id, "route_selected", decision.model_dump())
        trace_manager.add_span(
            trace_id,
            "plan_created",
            {"route": decision.route, "tools": decision.needs_tools, "max_steps": decision.max_steps},
        )

        workflow = WORKFLOWS.get(decision.route, WORKFLOWS["human_review"])
        try:
            result = workflow(request.query, auth, trace_id)
        except Exception as exc:
            result = {
                "answer": "The selected workflow failed. I am returning a safe fallback and asking for human review.",
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "degradations": [f"workflow_exception: {exc}"],
                "needs_human_review": True,
            }

        sources = [SourceChunk.model_validate(item) for item in result.get("sources", [])]
        citation_ids = [citation.id for citation in result.get("citations", [])]
        validation = tool_registry.call(
            "validate_citations",
            {"answer": result["answer"], "citation_ids": citation_ids},
            auth,
            trace_id,
        )
        trace_manager.add_span(
            trace_id,
            "citation_validation_completed",
            validation.model_dump() if hasattr(validation, "model_dump") else {"ok": False},
        )

        requires_citation = decision.route not in {"ambiguous_query", "human_review"} and bool(sources)
        output_safety = validate_output(result["answer"], citation_ids, requires_citation=requires_citation)
        if output_safety.blocked:
            SAFETY_BLOCKS.labels(stage="output").inc()
            result["answer"] = "I blocked the generated answer because it may contain sensitive content. A human review is required."
            result["needs_human_review"] = True
        trace_manager.add_span(trace_id, "output_safety_checked", output_safety.as_dict())

        latency_ms = (time.perf_counter() - started) * 1000
        token_cost_estimate = self._estimate_tokens_and_cost(request.query, result["answer"], sources)
        return AskResponse(
            trace_id=trace_id,
            route=decision.route,
            answer=result["answer"],
            citations=result.get("citations", []),
            sources=sources,
            plan=[decision.reason, *[f"Use tool: {tool}" for tool in decision.needs_tools]],
            tool_calls=[*result.get("tool_calls", []), validation.model_dump()],
            safety={
                "input": input_safety.as_dict(),
                "output": output_safety.as_dict(),
                "citation_validation": validation.model_dump(),
            },
            degradations=result.get("degradations", []),
            token_cost_estimate=token_cost_estimate,
            needs_human_review=result.get("needs_human_review", False) or output_safety.risk == "high",
            latency_ms=latency_ms,
        )

    def _estimate_tokens_and_cost(self, query: str, answer: str, sources: list[SourceChunk]) -> dict[str, Any]:
        context_text = " ".join(source.text for source in sources)
        input_tokens = max(1, int((len(query.split()) + len(context_text.split())) * 1.33))
        output_tokens = max(1, int(len(answer.split()) * 1.33))
        return {
            "input_tokens_estimate": input_tokens,
            "output_tokens_estimate": output_tokens,
            "total_tokens_estimate": input_tokens + output_tokens,
            "estimated_cost_usd": 0.0,
            "mode": "mock_or_local_estimate",
        }


agent_service = AgentService()
