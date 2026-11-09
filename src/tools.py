import json
from time import monotonic

from ouroboros.tools.registry import ToolEntry, ToolRegistry
from ouroboros.tools.tool_catalog import ToolCatalog
from pydantic import BaseModel, Field

from .calculations.forecast import forecast
from .config import ROOT, settings
from .integrations.telemetry import span
from .integrations.web import web_lookup
from .logging import log
from .models.schemas import Grade
from .rag import index


class ReportArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2500)


class WebArgs(ReportArgs):
    fresh: bool
    quotes: list[Grade]
    search: bool


class ForecastArgs(BaseModel):
    benchmark: Grade
    months: int = Field(ge=1, le=12)
    discount: float | None = None
    cut: float | None = None
    sensitivity: float | None = None


def reports(query: str):
    return index().search(query, settings().top_k)


FUNCTIONS = {
    "report_search": (
        ReportArgs,
        reports,
        "Найти top-k фрагментов отраслевых PDF с датой и страницей.",
    ),
    "web_search": (
        WebArgs,
        web_lookup,
        "Найти публикации доверенных издателей или последний спот EIA/FRED.",
    ),
    "price_forecast": (
        ForecastArgs,
        forecast,
        "Рассчитать месячные цены, прогнозный интервал и сценарий.",
    ),
}


class AnalystRegistry(ToolRegistry):
    def _load_modules(self):
        # This embedded application publishes exactly its own read-only domain tools.
        return ToolCatalog(())


def make_registry() -> AnalystRegistry:
    cfg = settings()
    root = cfg.cache / "ouroboros"
    root.mkdir(parents=True, exist_ok=True)
    registry = AnalystRegistry(ROOT, root)
    for name, (schema_cls, fn, description) in FUNCTIONS.items():

        def handler(ctx, _fn=fn, _schema=schema_cls, **kwargs):
            try:
                args = _schema.model_validate(kwargs)
                data = _fn(**args.model_dump())
                return json.dumps({"data": data}, ensure_ascii=False)
            except ValueError as exc:
                return json.dumps({"error": str(exc)}, ensure_ascii=False)

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema_cls.model_json_schema(),
            },
        }
        registry.register(
            ToolEntry(name, schema, handler, safety_policy="skip"), origin="src.tools"
        )
    return registry


def _run_tool(registry: AnalystRegistry, name: str, **args):
    started = monotonic()
    try:
        result = registry.execute_result(name, args)
        raw = json.loads(result.text) if result.status == "ok" else {}
        error = raw.get("error") if result.status == "ok" else f"Ouroboros: {result.code}"
        data = raw.get("data")
    except Exception as exc:
        # Keep provider internals, filesystem layouts and secrets out of the web response.
        data, error = None, f"{name} недоступен ({type(exc).__name__})."
    event = {
        "tool": name,
        "engine": "ouroboros.ToolRegistry",
        "status": "error" if error else "ok",
        "ms": round((monotonic() - started) * 1000),
    }
    return data, error, event


def run_tool(registry: AnalystRegistry, name: str, **args):
    with span(
        name,
        "retriever" if name == "report_search" else "tool",
        input=args,
        metadata={"engine": "ouroboros.ToolRegistry"},
    ) as obs:
        data, error, event = _run_tool(registry, name, **args)
        obs.update(output={"data": data, "error": error}, status=event["status"])
        log("tool.completed", **event)
        return data, error, event
