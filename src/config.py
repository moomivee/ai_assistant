import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class Settings(BaseModel):
    data: Path
    cache: Path
    top_k: int = Field(ge=1, le=15)
    llm_key: str
    llm_base: str
    llm_model: str
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
    answer_review: bool = False


@lru_cache
def settings() -> Settings:
    s = Settings(
        data=Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve(),
        cache=Path(os.getenv("CACHE_DIR", str(ROOT / ".cache"))).resolve(),
        top_k=int(os.getenv("RAG_TOP_K", "6")),
        llm_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
        web_key=os.getenv("TAVILY_API_KEY", "").strip(),
        domains=tuple(
            d.strip().lower()
            for d in os.getenv(
                "WEB_DOMAINS",
                "eia.gov,fred.stlouisfed.org,opec.org,iea.org,cbr.ru,economy.gov.ru,minfin.gov.ru,minenergo.gov.ru,reuters.com,interfax.ru,consilium.europa.eu,ofac.treasury.gov",
            ).split(",")
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
        answer_review=os.getenv("ANSWER_SELF_REVIEW", "false").lower() == "true",
    )
    os.environ.setdefault("OUROBOROS_DATA_DIR", str(s.cache / "ouroboros"))
    return s


def manifest() -> list[dict]:
    return json.loads((settings().data / "reports.json").read_text(encoding="utf-8"))
