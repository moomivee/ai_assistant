from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Grade = Literal["Brent", "WTI", "Urals"]
FORECAST_TARGET = "Сорт, цену которого пользователь просит спрогнозировать. Упомянутая база сравнения или дисконта сама по себе не является целью прогноза."


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParameterEvidence(StrictModel):
    field: Literal["months", "discount", "cut", "sensitivity"]
    quote: str = Field(min_length=1, max_length=500)


LAST_EVENT_SCOPE = (
    "True if the research question asks which actual occurrence is the most recent "
    "as of the request time (and possibly what happened then). Distinguish a completed "
    "event from an upcoming event, current value, generic news, historical interval, "
    "or the last event mentioned in a specified document."
)


class LatestEventScope(StrictModel):
    latest_completed_event: bool = Field(description=LAST_EVENT_SCOPE)


class EventPublicationWindow(StrictModel):
    quote: str = Field(
        min_length=1,
        max_length=500,
        description="Verbatim request clause limiting publication dates of sources.",
    )
    start_date: str | None = Field(
        default=None, max_length=10, description="Inclusive YYYY-MM-DD, null if open."
    )
    end_date: str | None = Field(
        default=None, max_length=10, description="Inclusive YYYY-MM-DD, null if open."
    )

    @model_validator(mode="after")
    def validate_period(self) -> Self:
        bounds = [value for value in (self.start_date, self.end_date) if value is not None]
        if not bounds or any(date.fromisoformat(value).isoformat() != value for value in bounds):
            raise ValueError("A publication window requires at least one ISO calendar date.")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("The publication window ends before its start.")
        return self


class Plan(StrictModel):
    skill: Literal["report_analysis", "web_research", "price_forecast"] | None = None
    intent: Literal["research", "forecast", "capabilities", "outside"] = Field(
        description=(
            "research — анализ нефтегазового рынка и его факторов; forecast — наш расчёт цены. "
            "capabilities — пользователь просит описать возможности этого агента или привести "
            "примеры вопросов к нему. Просьба выполнить конкретный анализ или расчёт "
            "относится к research или forecast, даже если начинается с «можешь». "
            "outside — другая цель, включая личную карьеру, вакансии и зарплату профессии, "
            "даже если упомянут нефтегаз. Влияние расходов на персонал на добычу/рынок — research."
        )
    )
    rag_query: str = Field(max_length=2000)
    fresh: bool = Field(
        description=(
            "Needs verification as of today, including an unbounded latest event even without "
            "an explicit date. False for a fixed historical period, a report's contents, "
            "or a calculation from CSV history. Independent of report_only."
        )
    )
    latest_completed_event: bool = Field(default=False, description=LAST_EVENT_SCOPE)
    event_publication_window: EventPublicationWindow | None = Field(
        default=None,
        description="Only for latest-event research with a user-specified period for source publication. Event dates, request time and the word 'latest' do not specify a publication window.",
    )
    report_only: bool
    wants_context: bool
    news: bool
    quotes: list[Grade] = Field(max_length=3)
    quote_request: str = Field(
        default="",
        max_length=2000,
        description="For research with current quotes, copy the price clause verbatim, or the whole question if only quotes are requested.",
    )
    context_request: str = Field(
        default="",
        max_length=2000,
        description="For research with current quotes, copy the separate research clause verbatim. Empty for quotes only.",
    )
    context_fresh: bool | None = Field(
        default=None,
        description="Does the separate research clause need current verification independently of a spot quote? Null when inapplicable.",
    )
    benchmark: Grade | None = Field(description=FORECAST_TARGET)
    months: int | None
    discount: FiniteFloat | None
    cut: FiniteFloat | None
    sensitivity: FiniteFloat | None
    parameters: list[ParameterEvidence] = Field(max_length=4)
    clarify: str = Field(
        max_length=500,
        description=(
            "Один вопрос пользователю по-русски при недостающем параметре прогноза "
            "или существенной неоднозначности исследования, указанной в research_options. "
            "В остальных случаях пустая строка."
        ),
    )
    research_options: list[Annotated[str, Field(max_length=180)]] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Для clarify в исследовании: 2–3 конкретные трактовки запроса, выбор между "
            "которыми изменит искомый показатель или объект. Иначе пустой список."
        ),
    )


class SourceAccess(StrictModel):
    mode: Literal["reports_only", "web_permitted", "no_instruction"]
    quote: str = Field(
        min_length=1,
        max_length=2000,
        description="Exact excerpt of the current question supporting this interpretation, including any exception or negation.",
    )


class ResearchQuoteScope(StrictModel):
    quote_request: str = Field(min_length=1, max_length=2000)
    context_request: str = Field(max_length=2000)
    context_fresh: bool


class ForecastInputs(StrictModel):
    benchmark: Grade | None = Field(description=FORECAST_TARGET)
    benchmark_quote: str = Field(
        max_length=500,
        description="Дословный фрагмент о том, для какого сорта нужен прогноз, отдельно от условия дисконта.",
    )
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
    citations: list[dict] = Field(default_factory=list)


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
    argument_repair: dict | None = None
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
    time_context: dict = Field(default_factory=dict)
    semantic_review_enabled: bool | None = None
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
    intent: Literal["research", "forecast", "capabilities", "outside", "unknown"]
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
    dialogue_token: str | None = Field(default=None, repr=False)


class EvidenceCheck(StrictModel):
    fact: str = Field(min_length=1, max_length=180)
    source_ids: list[str] = Field(max_length=4)


class NextAction(StrictModel):
    assessment: Literal["need_evidence", "need_detail", "enough", "unavailable"] = "need_evidence"
    purpose: str = Field(
        max_length=250,
        description=(
            "Краткая цель по-русски: какой пробел закрыть и почему выбранный источник "
            "подходит этому вопросу. Без утверждения об успехе ещё не выполненного шага."
        ),
    )
    action: Literal[
        "report_search", "read_report", "web_search", "read_web", "price_forecast", "finish"
    ]
    source_ids: list[str] = Field(default_factory=list, max_length=4)
    query: str = Field(
        default="",
        max_length=1500,
        description="Поисковый запрос для report_search/web_search; для остальных действий можно не передавать.",
    )
    report: str = Field(
        default="",
        max_length=100,
        description="ID найденного документа только для read_report; для остальных действий можно не передавать.",
    )
    page: int | None = Field(
        default=None,
        description="Положительный номер страницы для read_report; вместо него для Word/Excel допустим block_id.",
    )
    block_id: str = Field(default="", max_length=160)
    version_id: str = Field(default="", max_length=100)
    url: str = Field(default="", max_length=2048)
    topic: Literal["general", "news"] = "general"
    days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="Окно последних N дней. При заданных start_date/end_date должно быть null.",
    )
    start_date: str | None = Field(
        default=None,
        max_length=10,
        description="Начало периода YYYY-MM-DD. При заданном days должно быть null.",
    )
    end_date: str | None = Field(
        default=None,
        max_length=10,
        description="Конец периода YYYY-MM-DD. При заданном days должно быть null.",
    )
    search_depth: Literal["basic", "advanced"] = "basic"
    missing_facts: list[str] = Field(default_factory=list, max_length=6)
    evidence_checks: list[EvidenceCheck] = Field(default_factory=list, max_length=8)


class WebTimeFilterChoice(StrictModel):
    mode: Literal["unbounded", "relative", "dates"] = Field(
        description="Ровно один вариант: без окна, предложенные days или предложенные даты."
    )


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


MAX_CITED_CLAIMS = 8


class CitedAnswer(StrictModel):
    claims: list[CitedClaim] = Field(max_length=MAX_CITED_CLAIMS)
    complete: bool = Field(
        description="true, только если claims отвечают на ВСЕ части вопроса, включая вывод сравнения."
    )


class CitationReview(StrictModel):
    feedback: str = Field(max_length=600)
    supported: list[int] = Field(max_length=MAX_CITED_CLAIMS)
    complete: bool
    coverage: Literal["sufficient", "partial", "unverified"] = Field(
        default="unverified",
        description=(
            "Whether the supported claims sufficiently answer the scope and relation requested "
            "in the question, including its time scope. A true individual fact does not establish "
            "complete coverage."
        ),
    )
    missing_aspects: list[Annotated[str, Field(min_length=1, max_length=250)]] = Field(
        default_factory=list,
        max_length=6,
        description="Specific requested aspects or evidence still missing; empty only if none are identified.",
    )
    research_options: list[str] = Field(
        default_factory=list,
        description=(
            "Только при allow_research_clarification=true и неоднозначной цели: "
            "2–3 коротких варианта смысла вопроса на русском для выбора пользователем. "
            "Если не хватает сведений об однозначной задаче, верни []."
        ),
    )


class Question(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    previous: str | None = Field(default=None, max_length=2000)
    remember_clarification: bool = False
    dialogue_token: str | None = Field(default=None, max_length=128, repr=False)
    document_ids: list[str] = Field(default_factory=list, max_length=30)
    report_only: bool = False
    note_id: str | None = Field(default=None, max_length=100)
