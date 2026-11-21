import json
import logging
from time import monotonic

from ouroboros.tools.registry import ToolEntry, ToolRegistry
from ouroboros.tools.tool_catalog import ToolCatalog
from pydantic import BaseModel, Field, model_validator

from .budget import RequestBudgetExceeded
from .calculations.forecast import forecast
from .config import ROOT, settings
from .integrations.telemetry import span
from .integrations.web import read_web, web_lookup
from .logging import log
from .models.schemas import Grade
from .request_context import current_scope


class ReportArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2500)


class WebArgs(ReportArgs):
    query: str = Field(default="", max_length=2500)
    fresh: bool
    quotes: list[Grade]
    search: bool
    quote_fallback: bool = False
    topic: str = "general"
    days: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    search_depth: str = "basic"

    @model_validator(mode="after")
    def search_query_required(self):
        if self.search and not self.query.strip():
            raise ValueError("web_search requires a non-empty query when searching publications.")
        return self


class ReadWebArgs(BaseModel):
    url: str = Field(max_length=2048)
    query: str = Field(default="", max_length=1500)


class ForecastArgs(BaseModel):
    benchmark: Grade
    months: int = Field(ge=1, le=12)
    discount: float | None = None
    cut: float | None = None
    sensitivity: float | None = None


class ReadReportArgs(BaseModel):
    report: str = Field(min_length=1, max_length=100)
    page: int | None = Field(default=None, ge=1)
    block_id: str = Field(default="", max_length=160)
    version_id: str = Field(default="", max_length=100)

    @model_validator(mode="after")
    def location_required(self):
        if self.page is None and not self.block_id.strip():
            raise ValueError(
                "read_report requires a positive page or a block_id from retrieved sources."
            )
        return self


def read_report(report: str, page: int | None = None, block_id="", version_id=""):
    """Open a known immutable page or document block, including OCR and tables."""
    from .knowledge.store import get_store

    scope, store = current_scope(), get_store()
    if scope.document_ids and report not in scope.document_ids:
        raise ValueError("Document is outside the user-selected scope.")
    document = store.document(report)
    if not document or document["archived"]:
        raise ValueError("Unknown report ID. Use an ID returned by report_search.")
    version_id = version_id or document["active_version_id"]
    version = store.version(report, version_id) if version_id else None
    if not version or version["status"] not in {"ready", "partial"}:
        raise ValueError("This document version has no readable extracted content.")
    chunks = store.blocks(report, version_id)
    if block_id and not any(chunk["block_id"] == block_id for chunk in chunks):
        raise ValueError("The requested block is absent in this document version.")
    if page is not None:
        chunks = [chunk for chunk in chunks if chunk.get("page") == page]
    elif block_id:
        chunks = [chunk for chunk in chunks if chunk["block_id"] == block_id]
    else:
        raise ValueError("Provide page for PDF/image, or block_id for DOCX/Excel/CSV.")
    if not chunks:
        raise ValueError("The requested page or block is absent in this document version.")
    if block_id and page is not None and not any(chunk["block_id"] == block_id for chunk in chunks):
        raise ValueError("The requested block does not belong to this page.")
    text = "\n\n".join(chunk["text"] for chunk in chunks)
    methods = set(chunk.get("method", "native") for chunk in chunks)
    suffix = f"#page={page}" if page is not None else ""
    return [
        {
            "report": report,
            "document_id": report,
            "version_id": version_id,
            "block_id": chunks[0]["block_id"] if len(chunks) == 1 else None,
            "block_ids": [chunk["block_id"] for chunk in chunks],
            "page": page,
            "chunk_id": f"{report}:{version_id}:{page or block_id}:full",
            "kind": "report",
            "title": version["title"],
            "date": version["published_date"],
            "url": (version["source_url"] or version["local_url"]) + suffix,
            "local_url": version["local_url"] + suffix,
            "locator": {"page": page} if page is not None else chunks[0]["locator"],
            "method": next(iter(methods)) if len(methods) == 1 else "mixed",
            "text": text[:20000],
            "full_page": page is not None,
            "full_block": page is None,
            "truncated": len(text) > 20000,
        }
    ]


def reports(query: str):
    from .rag.managed import managed_search

    return managed_search(
        query, settings().top_k, document_ids=current_scope().document_ids or None
    )


FUNCTIONS = {
    "read_report": (
        ReadReportArgs,
        read_report,
        "Прочитать страницу PDF/скана или блок Word/Excel/CSV из найденной версии документа.",
    ),
    "report_search": (
        ReportArgs,
        reports,
        "Найти top-k фрагментов выбранной базы знаний с версией, страницей или диапазоном ячеек.",
    ),
    "web_search": (
        WebArgs,
        web_lookup,
        "Найти публикации доверенных издателей или последний спот EIA/FRED.",
    ),
    "read_web": (
        ReadWebArgs,
        read_web,
        "Извлечь текст уже найденной доверенной веб-страницы, сохранив ссылку и фрагменты.",
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

        def handler(ctx, _fn=fn, _schema=schema_cls, _name=name, **kwargs):
            try:
                if current_scope().report_only and _fn in (web_lookup, read_web):
                    raise ValueError("Web access is disabled by the user's report-only scope.")
                args = _schema.model_validate(kwargs)
                data = _fn(**args.model_dump())
                error = data.get("error") if isinstance(data, dict) else None
                return json.dumps({"data": data, "error": error}, ensure_ascii=False)
            except (ValueError, RequestBudgetExceeded) as exc:
                return json.dumps({"error": str(exc)}, ensure_ascii=False)
            except Exception as exc:
                # Ouroboros otherwise reduces handler exceptions to TOOL_ERROR. Keep
                # the error class without exposing URLs, credentials or raw responses.
                kind = type(exc).__name__
                log("tool.handler_failed", logging.ERROR, tool=_name, error_type=kind)
                return json.dumps(
                    {"error": f"Инструмент {_name} недоступен ({kind})."}, ensure_ascii=False
                )

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
