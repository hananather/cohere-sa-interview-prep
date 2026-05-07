from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskType = Literal[
    "find",
    "answer",
    "summarize",
    "synthesize",
    "compare",
    "verify",
    "refuse_or_clarify",
    "permission_sensitive",
    "bilingual",
    "structured_analysis",
    "scanned_ocr",
    "adversarial",
    "demo_candidate",
]


class EvalUserContext(BaseModel):
    user_id: str = "alex_analyst"
    role: str = "planning_analyst"
    access_level: str = "public_internal"
    language: str = "en"


class ExpectedSource(BaseModel):
    doc_id: str
    required: bool = True
    status: str | None = None
    access_level: str | None = None


class DisallowedSource(BaseModel):
    doc_id: str
    reason: str


class GraderConfig(BaseModel):
    route: bool = True
    retrieval: bool = True
    filter: bool = True
    answer_key_facts: bool = True
    citation_validation: bool = True
    llm_judge: bool = False


class EvalCase(BaseModel):
    query_id: str
    user_query: str
    task_type: TaskType
    topic: str
    capability_tags: list[str] = Field(default_factory=list)
    complexity_level: Literal["L1", "L2", "L3", "L4", "L5"]
    complexity_score: int = Field(ge=1, le=10)
    complexity_dimensions: list[str] = Field(default_factory=list)
    user_context: EvalUserContext = Field(default_factory=EvalUserContext)
    expected_route: str
    expected_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    disallowed_sources: list[DisallowedSource] = Field(default_factory=list)
    expected_key_facts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    expected_answer_behavior: str = "answer_with_citations"
    expected_refusal: bool = False
    citation_required: bool = True
    exact_expected_result: dict[str, Any] | None = None
    grader_config: GraderConfig = Field(default_factory=GraderConfig)
    demo_notes: str = ""


class EvalSuite(BaseModel):
    suite: str
    generated_by: str | None = None
    cases: list[EvalCase]


class GradeBreakdown(BaseModel):
    route: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    rerank: dict[str, Any] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)
    code_execution: dict[str, Any] = Field(default_factory=dict)
    answer: dict[str, Any] = Field(default_factory=dict)
    citations: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)
    operations: dict[str, Any] = Field(default_factory=dict)


class EvalCaseOutcome(BaseModel):
    query_id: str
    task_type: str
    complexity_level: str
    passed: bool
    failure_categories: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    route: str | None = None
    latency_ms: float | None = None
    grades: GradeBreakdown
    answer: str = ""
    sources: list[str] = Field(default_factory=list)


class EvalRunReport(BaseModel):
    run_id: str
    suite: str
    mode: Literal["fixture", "live"] = "fixture"
    variant: str = "agentic_rag_tools"
    metrics: dict[str, Any] = Field(default_factory=dict)
    cases: list[EvalCaseOutcome] = Field(default_factory=list)
    report_dir: str | None = None
