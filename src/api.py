import csv
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import ask
from .config import manifest, settings
from .integrations.telemetry import flush
from .logging import configure_logging, log, request_id
from .models.schemas import AgentResponse, Question

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app):
    configure_logging()
    log(
        "service.started",
        model=settings().llm_model,
        llm_configured=bool(settings().llm_key),
    )
    try:
        yield
    finally:
        flush()
        log("service.stopped")


app = FastAPI(title="AI-агент «Нефтегазовый аналитик»", version="2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def request_logging(request, call_next):
    rid, start, code = uuid4().hex, perf_counter(), 500
    token = request_id.set(rid)
    try:
        response = await call_next(request)
        code = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    except Exception as exc:
        log("http.failed", logging.ERROR, error_type=type(exc).__name__)
        raise
    finally:
        route = request.scope.get("route")
        log(
            "http.completed",
            method=request.method,
            route=route.path if route else "unmatched",
            status=code,
            ms=round((perf_counter() - start) * 1000, 2),
        )
        request_id.reset(token)


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    from .integrations.llm import llm
    from .rag import index

    try:
        index()
        if llm().client is None:
            raise ValueError("LLM not configured")
        llm().client.with_options(timeout=3).models.list()
        return {"status": "ready"}
    except Exception as exc:
        log("service.not_ready", logging.WARNING, error_type=type(exc).__name__)
        raise HTTPException(503, "Индекс или LLM пока недоступны.") from exc


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    cfg = settings()
    with (cfg.data / "prices/monthly.csv").open(encoding="utf-8") as f:
        history = list(csv.DictReader(f))[-24:]
    return {
        "model": cfg.llm_model if cfg.llm_key else None,
        "tavily": bool(cfg.web_key),
        "reports": [
            {"title": r["title"], "date": r["date"], "url": "/reports/" + r["file"]}
            for r in manifest()
        ],
        "history": [
            {"month": r["month"], "Brent": float(r["Brent"]), "WTI": float(r["WTI"])}
            for r in history
        ],
    }


@app.get("/reports/{filename}", include_in_schema=False)
def pdf(filename: str):
    if filename not in {row["file"] for row in manifest()}:
        raise HTTPException(404)
    return FileResponse(settings().data / "reports" / filename, media_type="application/pdf")


@app.post("/api/ask", response_model=AgentResponse)
def query(body: Question):
    try:
        return ask(body.question, body.previous)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
