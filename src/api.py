import asyncio
import csv
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .agent import ask
from .config import manifest, settings
from .forecast_memory import forecast_memory, next_snapshot, previous_text
from .integrations.dialogue_memory import (
    DialogueBusy,
    DialogueConflict,
    DialogueStorageError,
    open_dialogue,
    resume_dialogue,
)
from .integrations.llm import LLMUnavailable
from .integrations.telemetry import flush
from .knowledge.api import router as knowledge_router
from .knowledge.jobs import shutdown_jobs
from .knowledge.notes_api import router as notes_router
from .knowledge.updates_api import router as updates_router
from .logging import configure_logging, log, request_id
from .models.schemas import AgentResponse, Question
from .request_context import request_scope
from .request_progress import report_progress_to
from .upload_limits import UploadLimitMiddleware

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app):
    configure_logging()
    log(
        "service.started",
        model=settings().active_model,
        provider=settings().llm_provider,
        llm_configured=settings().llm_configured,
    )
    try:
        yield
    finally:
        shutdown_jobs(wait=False)
        flush()
        log("service.stopped")


app = FastAPI(title="AI-агент «Нефтегазовый аналитик»", version="3.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
@app.get("/static/index.html", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(knowledge_router)
app.include_router(notes_router)
app.include_router(updates_router)
app.add_middleware(UploadLimitMiddleware)


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
        llm().check_ready(timeout=3)
        return {"status": "ready"}
    except Exception as exc:
        log("service.not_ready", logging.WARNING, error_type=type(exc).__name__)
        raise HTTPException(503, "Индекс или LLM пока недоступны.") from exc


@app.get("/api/status")
def status():
    cfg = settings()
    with (cfg.data / "prices/monthly.csv").open(encoding="utf-8") as f:
        history = list(csv.DictReader(f))[-24:]
    return {
        "model": cfg.active_model if cfg.llm_configured else None,
        "provider": cfg.llm_provider,
        "llm_configured": cfg.llm_configured,
        "tavily": bool(cfg.web_key),
        "knowledge_base": True,
        "ocr": {"engine": "tesseract", "configured_languages": ["rus", "eng"]},
        "web_extract": bool(cfg.web_key),
        "research_notes": True,
        "agent_mode": cfg.agent_mode,
        "agent_max_steps": cfg.agent_max_steps,
        "agent_no_progress_limit": cfg.agent_no_progress_limit,
        "agent_timeout_seconds": cfg.agent_timeout_seconds,
        "agent_answer_reserve_seconds": min(
            cfg.agent_answer_reserve_seconds, cfg.agent_timeout_seconds / 2
        ),
        "reports": [
            {"title": r["title"], "date": r["date"], "url": "/reports/" + r["file"]}
            for r in manifest()
        ],
        "history": [
            {"month": r["month"], "Brent": float(r["Brent"]), "WTI": float(r["WTI"])}
            for r in history
        ],
    }


@app.get("/api/rag/status")
def rag_status():
    from .rag.managed import managed_status

    return {
        **managed_status(),
        "top_k": settings().top_k,
        "metrics_explanation": {
            "hit@1": "Доля размеченных вопросов, где первый фрагмент релевантен.",
            "hit@5": "Доля вопросов с хотя бы одним релевантным фрагментом в первых пяти.",
            "MRR": "Среднее обратное место первого релевантного фрагмента. Чем выше, тем лучше.",
            "limits": "Метрики зависят от размеченного корпуса и не измеряют правильность ответа LLM. Новые загрузки требуют отдельной разметки.",
        },
    }


@app.get("/api/skills")
def skills():
    from .integrations.agent_skills import get_skill_catalog

    return {"skills": get_skill_catalog()}


@app.get("/reports/{filename}", include_in_schema=False)
def pdf(filename: str):
    if filename not in {row["file"] for row in manifest()}:
        raise HTTPException(404)
    return FileResponse(settings().data / "reports" / filename, media_type="application/pdf")


def _answer(body: Question, previous: str | None):
    from .integrations.research_notes import load_note_context
    from .knowledge.store import get_store

    if body.document_ids:
        store = get_store()
        for identifier in body.document_ids:
            document = store.document(identifier)
            if not document or document["archived"] or not document["active_version_id"]:
                raise ValueError("Один из выбранных документов отсутствует или ещё не обработан.")
    note = load_note_context(body.note_id) if body.note_id else None
    with request_scope(body.document_ids, body.report_only, note):
        return ask(body.question, previous)


def _pending_forecast(response):
    if response["status"] == "clarification" and response["intent"] == "forecast":
        return (response.get("diagnostics") or {}).get("plan")
    return None


@app.post("/api/ask", response_model=AgentResponse)
def query(body: Question):
    scope = {
        "document_ids": sorted(set(body.document_ids)),
        "report_only": body.report_only,
        "note_id": body.note_id,
    }
    try:
        if body.dialogue_token is not None:
            if body.previous is not None:
                raise ValueError("При продолжении по токену передайте только новую реплику.")
            with resume_dialogue(body.dialogue_token, scope) as turn:
                with forecast_memory(turn.snapshot):
                    response = _answer(body, previous_text(turn.snapshot))
                    if plan := _pending_forecast(response):
                        response["dialogue_token"] = turn.advance(
                            next_snapshot(turn.snapshot, body.question, plan)
                        )
                    elif response["status"] == "unavailable":
                        response["dialogue_token"] = body.dialogue_token
                    else:
                        turn.close()
                return response
        response = _answer(body, body.previous)
        if (
            body.remember_clarification
            and body.previous is None
            and (plan := _pending_forecast(response))
        ):
            response["dialogue_token"] = open_dialogue(
                next_snapshot(None, body.question, plan), scope
            )
        return response
    except DialogueBusy:
        raise HTTPException(
            409, {"code": "dialogue_busy", "message": "Предыдущее уточнение ещё обрабатывается."}
        ) from None
    except DialogueConflict:
        raise HTTPException(
            409,
            {
                "code": "dialogue_conflict",
                "message": "Это уточнение больше недоступно. Начните новый прогноз.",
            },
        ) from None
    except DialogueStorageError:
        log("dialogue.storage_error", logging.ERROR)
        raise HTTPException(503, "Хранилище уточнений временно недоступно.") from None
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except FileNotFoundError:
        raise HTTPException(404, "Выбранная заметка не найдена.") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


async def _answer_events(body: Question):
    events = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def notify(stage):
        loop.call_soon_threadsafe(events.put_nowait, {"type": "progress", "stage": stage})

    async def run():
        try:
            with report_progress_to(notify):
                response = await run_in_threadpool(query, body)
            result = AgentResponse.model_validate(response).model_dump(mode="json")
            final = {"type": "result", "result": result}
        except HTTPException as exc:
            final = {"type": "error", "status": exc.status_code, "detail": exc.detail}
        except Exception as exc:
            log("ask.stream_failed", logging.ERROR, error_type=type(exc).__name__)
            final = {"type": "error", "status": 500, "detail": "Не удалось обработать запрос."}
        events.put_nowait(final)

    worker = asyncio.create_task(run())
    try:
        while True:
            event = await events.get()
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event["type"] != "progress":
                break
    finally:
        # A disconnected client must not release a synchronous agent's thread slot
        # or interrupt dialogue-state cleanup while its turn is still running.
        await asyncio.shield(worker)


@app.post("/api/ask/stream", response_class=StreamingResponse)
async def query_stream(body: Question):
    return StreamingResponse(
        _answer_events(body),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Accel-Buffering": "no",
        },
    )
