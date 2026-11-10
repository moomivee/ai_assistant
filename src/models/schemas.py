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
    fresh: bool
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
    clarify: str = Field(max_length=500)


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


class NextAction(StrictModel):
    assessment: Literal["need_evidence", "need_detail", "enough", "unavailable"] = "need_evidence"
    purpose: str = Field(max_length=250)
    action: Literal["report_search", "read_report", "web_search", "price_forecast", "finish"]
    source_ids: list[str] = Field(default_factory=list, max_length=4)
    query: str = Field(max_length=1500)
    report: str = Field(max_length=100)
    page: int | None


class CitedClaim(StrictModel):
    evidence_ids: list[str] = Field(min_length=1, max_length=12)
    text: str = Field(min_length=1, max_length=1200)


class CitedAnswer(StrictModel):
    evidence_summary: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Краткие проверяемые наблюдения до формулировки ответа: значения и единицы из источника, направление сравнения. Объясни, как единицы связаны с физической величиной. Это не финальный ответ.",
    )
    claims: list[CitedClaim] = Field(max_length=8)
    complete: bool


class CitationReview(StrictModel):
    feedback: str = Field(max_length=600)
    supported: list[int] = Field(max_length=8)
    complete: bool


class Question(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    previous: str | None = Field(default=None, max_length=2000)
