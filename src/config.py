import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_MIB = 100
MAX_UPLOAD_BYTES = MAX_UPLOAD_MIB * 1024 * 1024
DEFAULT_WEB_DOMAINS = (
    "eia.gov",
    "fred.stlouisfed.org",
    "opec.org",
    "iea.org",
    "cbr.ru",
    "economy.gov.ru",
    "minfin.gov.ru",
    "minenergo.gov.ru",
    "consilium.europa.eu",
    "ofac.treasury.gov",
    "reuters.com",
    "interfax.ru",
    "rbc.ru",
    "euronews.com",
    "ft.com",
    "theguardian.com",
    "ru.wikipedia.org",
    "en.wikipedia.org",
)
load_dotenv(ROOT / ".env")


class Settings(BaseModel):
    app_timezone: str = "Europe/Moscow"
    data: Path
    cache: Path
    top_k: int = Field(ge=1, le=15)
    llm_key: str
    llm_base: str
    llm_model: str
    llm_provider: Literal["openai", "gigachat"] = "openai"
    gigachat_credentials: str = Field(default="", repr=False)
    gigachat_scope: Literal["GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"] = (
        "GIGACHAT_API_PERS"
    )
    gigachat_base_url: str = "https://api.giga.chat/v1"
    gigachat_model: str = "GigaChat-2"
    gigachat_ca_bundle: str = ""
    web_key: str
    domains: tuple[str, ...]
    timeout: float = Field(gt=0, le=120)
    llm_timeout: float = Field(default=120, gt=0, le=600)
    news_days: int = Field(ge=1, le=365)
    forecast_windows: int = Field(ge=24, le=96)
    embedding: str = "minilm-multi"
    rag_search: str = "dense"
    llm_max_tokens: int = Field(default=2400, ge=256, le=8192)
    trace_file: Path | None = None
    trace_content: bool = False
    langfuse_enabled: bool = False
    prompt_source: str = "local"
    prompt_label: str = "production"
    agent_mode: str = "loop"
    agent_max_steps: int = Field(default=6, ge=2, le=12)
    agent_no_progress_limit: int = Field(default=2, ge=1, le=12)
    agent_timeout_seconds: float = Field(default=180, ge=5, le=900)
    agent_answer_reserve_seconds: float = Field(default=60, ge=0, le=300)
    answer_review: bool = True

    @field_validator("app_timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(
                "APP_TIMEZONE должен быть доступным часовым поясом IANA, например Europe/Moscow. "
                "Проверьте значение и наличие системного пакета tzdata."
            ) from None
        return value

    @property
    def active_model(self) -> str:
        return self.gigachat_model if self.llm_provider == "gigachat" else self.llm_model

    @property
    def llm_configured(self) -> bool:
        return bool(self.gigachat_credentials if self.llm_provider == "gigachat" else self.llm_key)


@lru_cache
def settings() -> Settings:
    s = Settings(
        app_timezone=os.getenv("APP_TIMEZONE", "Europe/Moscow").strip(),
        data=Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve(),
        cache=Path(os.getenv("CACHE_DIR", str(ROOT / ".cache"))).resolve(),
        top_k=int(os.getenv("RAG_TOP_K", "6")),
        llm_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        gigachat_credentials=os.getenv("GIGACHAT_CREDENTIALS", "").strip(),
        gigachat_scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
        gigachat_base_url=os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1").strip(),
        gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat-2").strip(),
        gigachat_ca_bundle=os.getenv("GIGACHAT_CA_BUNDLE", "").strip(),
        web_key=os.getenv("TAVILY_API_KEY", "").strip(),
        domains=tuple(
            d.strip().lower()
            for d in os.getenv("WEB_DOMAINS", ",".join(DEFAULT_WEB_DOMAINS)).split(",")
            if d.strip()
        ),
        timeout=float(os.getenv("HTTP_TIMEOUT", "25")),
        llm_timeout=float(os.getenv("LLM_HTTP_TIMEOUT", os.getenv("HTTP_TIMEOUT", "25"))),
        news_days=int(os.getenv("NEWS_MAX_DAYS", "30")),
        forecast_windows=int(os.getenv("BACKTEST_ORIGINS", "48")),
        embedding=os.getenv("RAG_EMBEDDING", "minilm-multi"),
        rag_search=os.getenv("RAG_SEARCH_MODE", "dense"),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2400")),
        trace_file=Path(os.environ["TRACE_FILE"]) if os.getenv("TRACE_FILE") else None,
        trace_content=os.getenv("TRACE_CONTENT", "false").lower() == "true",
        langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
        prompt_source=os.getenv("PROMPT_SOURCE", "local"),
        prompt_label=os.getenv("PROMPT_LABEL", "production"),
        agent_mode=os.getenv("AGENT_MODE", "loop"),
        agent_max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
        agent_no_progress_limit=int(os.getenv("AGENT_NO_PROGRESS_LIMIT", "2")),
        agent_timeout_seconds=float(os.getenv("AGENT_TIMEOUT_SECONDS", "180")),
        agent_answer_reserve_seconds=float(os.getenv("AGENT_ANSWER_RESERVE_SECONDS", "60")),
        answer_review=os.getenv("ANSWER_SELF_REVIEW", "true").lower() == "true",
    )
    os.environ.setdefault("OUROBOROS_DATA_DIR", str(s.cache / "ouroboros"))
    return s


def manifest() -> list[dict]:
    return json.loads((settings().data / "reports.json").read_text(encoding="utf-8"))
