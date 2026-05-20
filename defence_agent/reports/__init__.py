"""Background report-generation agents for the Defence Agent demo."""

from defence_agent.reports.orchestrator import run_report_task
from defence_agent.reports.templates import get_report_task, list_report_tasks

__all__ = ["get_report_task", "list_report_tasks", "run_report_task"]
