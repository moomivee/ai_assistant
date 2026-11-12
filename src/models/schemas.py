from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

Grade = Literal["Brent", "WTI", "Urals"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParameterEvidence(StrictModel):
    field: Literal["months", "discount", "cut", "sensitivity"]
    quote: str = Field(min_length=1, max_length=500)


class Plan(StrictModel):
    intent: Literal["research", "forecast", "outside"]
    rag_query: str = Field(max_length=2000)
    fresh: bool = Field(
        description=(
            "Needs verification as of today, including an unbounded latest event even without "
            "an explicit date. False for a fixed historical period, a report's contents, "
            "or a calculation from CSV history. Independent of report_only."
        )
    )
    report_only: bool
    wants_context: bool
    news: bool
    quotes: list[Grade] = Field(max_length=3)
    benchmark: Grade | None
    months: int | None
    discount: FiniteFloat | None
    cut: FiniteFloat | None
    sensitivity: FiniteFloat | None
    parameters: list[ParameterEvidence] = Field(max_length=4)
    clarify: str = Field(
        max_length=500,
        description="Уточнение только недостающих параметров forecast. Для research и outside всегда пустая строка.",
    )


class ForecastInputs(StrictModel):
    benchmark: Grade | None
    benchmark_quote: str = Field(max_length=500)
    months: int | None
    discount: FiniteFloat | None
    cut: FiniteFloat | None
    sensitivity: FiniteFloat | None
    parameters: list[ParameterEvidence] = Field(max_length=4)


class ForecastScope(StrictModel):
    context_request: str = Field(max_length=2000)
    fresh: bool
    news: bool
    quote_request: str = Field(max_length=2000)
    quotes: list[Grade] = Field(max_length=3)


class Claim(StrictModel):
    source: str = Field(min_length=1, max_length=20)
    quote: str = Field(min_length=1, max_length=1800)
    text: str = Field(min_length=1, max_length=1200)
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceAnswer(StrictModel):
    complete: bool
    claims: list[Claim] = Field(max_length=10)
    limitations: list[str] = Field(max_length=5)


class DraftAnswer(StrictModel):
    # Extract the facts before deciding whether they answer the whole question.
    claims: list[Claim] = Field(max_length=10)
    complete: bool
    limitations: list[str] = Field(max_length=0)


class EvidenceSpan(StrictModel):
    source: str = Field(min_length=1, max_length=20)
    quote: str = Field(min_length=6, max_length=1800)


class ClaimReview(StrictModel):
    index: int = Field(ge=0, le=9)
    verdict: Literal["supported", "irrelevant", "contradicted", "unsupported"]


class AnswerReview(StrictModel):
    checks: list[ClaimReview] = Field(max_length=10)
    complete: bool
    repair_evidence: list[EvidenceSpan] = Field(max_length=6)


class JudgeResult(StrictModel):
    groundedness: int = Field(ge=0, le=4)
    relevance: int = Field(ge=0, le=4)
    tone: int = Field(ge=0, le=4)
    completeness: int = Field(ge=0, le=4)
    explanation: str = Field(max_length=2000)


class Source(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    kind: Literal["report", "web", "model"]
    title: str
    date: str | None
    url: str
    text: str
    page: int | None = None
    chunk_id: str | None = None


class ToolEvent(StrictModel):
    tool: str
    engine: str
    status: Literal["ok", "error"]
    ms: int


class DiagnosticStep(StrictModel):
    step: int = Field(ge=1)
    tool: str
    purpose: str
    arguments: dict
    status: Literal["ok", "error", "rejected"]
    ms: int | None
    error: str | None = None
    result_count: int | None = None
    source_ids: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    result: dict | None = None
    duplicate_of: int | None = None
    reused_source_ids: list[str] = Field(default_factory=list)
    new_source_ids: list[str] = Field(default_factory=list)
    new_evidence: bool | None = None
    no_progress_steps: int = 0


class AgentDiagnostics(StrictModel):
    plan: Plan | None = None
    steps: list[DiagnosticStep] = Field(default_factory=list)
    retrieved_sources: list[Source] = Field(default_factory=list)
    answer_feedback: str = ""
    failure: str = ""
    no_progress_steps: int = 0
    no_progress_limit: int | None = None
    timeout_seconds: float | None = None
    answer_reserve_seconds: float | None = None
    answer_checks: list[dict] = Field(default_factory=list)


class AgentResponse(StrictModel):
    question: str
    answer: str
    status: Literal["ok", "partial", "clarification", "outside", "unavailable"]
    clarification: bool
    intent: Literal["research", "forecast", "outside", "unknown"]
    mode: str
    sources: list[Source]
    claims: list[Claim]
    trace: list[ToolEvent]
    trace_id: str
    forecast: dict | None
    decisions: list[dict] = Field(default_factory=list)
    stop_reason: str = "completed"
    agent_mode: str = "workflow"
    rejected_actions: int = 0
    diagnostics: AgentDiagnostics | None = None


class EvidenceCheck(StrictModel):
    fact: str = Field(min_length=1, max_length=180)
    source_ids: list[str] = Field(max_length=4)


class NextAction(StrictModel):
    assessment: Literal["need_evidence", "need_detail", "enough", "unavailable"] = "need_evidence"
    purpose: str = Field(max_length=250)
    action: Literal["report_search", "read_report", "web_search", "price_forecast", "finish"]
    source_ids: list[str] = Field(default_factory=list, max_length=4)
    query: str = Field(max_length=1500)
    report: str = Field(max_length=100)
    page: int | None
    missing_facts: list[str] = Field(default_factory=list, max_length=6)
    evidence_checks: list[EvidenceCheck] = Field(default_factory=list, max_length=8)


class CitedClaim(StrictModel):
    evidence_ids: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Идентификаторы блоков, подтверждающих это утверждение.",
    )
    text: str = Field(
        min_length=1,
        max_length=1200,
        description="Короткое утверждение по-русски: запрошенные характеристики объекта или вывод сравнения.",
    )


class CitedAnswer(StrictModel):
    claims: list[CitedClaim] = Field(max_length=8)
    complete: bool = Field(
        description="true, только если claims отвечают на ВСЕ части вопроса, включая вывод сравнения."
    )


class CitationReview(StrictModel):
    feedback: str = Field(max_length=600)
    supported: list[int] = Field(max_length=8)
    complete: bool


class Question(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    previous: str | None = Field(default=None, max_length=2000)
