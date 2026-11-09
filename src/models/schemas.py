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


class Claim(StrictModel):
    source: str = Field(min_length=1, max_length=20)
    quote: str = Field(min_length=6, max_length=1800)
    text: str = Field(min_length=1, max_length=1200)


class EvidenceAnswer(StrictModel):
    complete: bool
    claims: list[Claim] = Field(max_length=10)
    limitations: list[str] = Field(max_length=5)


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


class Question(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    previous: str | None = Field(default=None, max_length=2000)
