"""Offline evaluation helpers for Defence Agent transcripts."""

from defence_agent.evals.harness import (
    DEFAULT_REGISTRY,
    DEFAULT_TRANSCRIPT_RUN_NAMES,
    EvalHarnessError,
    load_outcomes,
    load_registry,
    markdown_report,
    score_outcome,
    summarize_transcript_runs,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "DEFAULT_TRANSCRIPT_RUN_NAMES",
    "EvalHarnessError",
    "load_outcomes",
    "load_registry",
    "markdown_report",
    "score_outcome",
    "summarize_transcript_runs",
]
