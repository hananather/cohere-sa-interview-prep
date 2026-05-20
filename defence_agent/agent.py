"""Canonical Google ADK agent for the Defence Agent demo.

Run with:
    adk web .

Then choose the `defence_agent` agent in ADK Dev UI.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm

from .prompts import AGENT_INSTRUCTION
from .tool_cache import after_tool_cache_store, before_tool_cache_lookup
from .tools import search_documents


load_dotenv()

AGENT_ID = "defence_agent"
MODEL = os.getenv("DEFTECH_ADK_MODEL", "cohere/command-a-03-2025")
ADK_TIMEOUT_SECONDS = float(os.getenv("DEFTECH_ADK_TIMEOUT_SECONDS", "45"))


def _instruction(context: ReadonlyContext) -> str:
    """Inject compact session memory without adding a custom agent loop."""

    previous_answer = str(context.state.get("last_grounded_answer", "")).strip()
    if not previous_answer:
        return AGENT_INSTRUCTION
    return (
        AGENT_INSTRUCTION
        + "\n\nPrevious answer in this session:\n"
        + previous_answer[:1800]
        + "\n\nUse that previous answer for follow-up questions. Search again if the follow-up needs fresh evidence."
    )


def _lite_llm_kwargs(model: str) -> dict[str, object]:
    kwargs: dict[str, object] = {"timeout": ADK_TIMEOUT_SECONDS}
    if _is_command_a_plus(model):
        kwargs["custom_llm_provider"] = "cohere_chat"
        kwargs["allowed_openai_params"] = ["tools"]
    return kwargs


def _is_command_a_plus(model: str) -> bool:
    model_name = str(model).rsplit("/", maxsplit=1)[-1]
    return model_name.startswith("command-a-plus-")


root_agent = LlmAgent(
    name=AGENT_ID,
    model=LiteLlm(model=MODEL, **_lite_llm_kwargs(MODEL)),
    instruction=_instruction,
    tools=[search_documents],
    output_key="last_retrieval_status",
    before_tool_callback=before_tool_cache_lookup,
    after_tool_callback=after_tool_cache_store,
)
