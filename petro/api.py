# принимаем вопрос из браузера, передаём агенту и возвращаем ответ интерфейсу

import csv
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .agent import ask
from .config import manifest, settings

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="AI_Oil_Gas_assistant", description="Нефтегазовый аналитик", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    previous: str | None = Field(default=None, max_length=2000)


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
            {"title": r["title"], "date": r["date"], "url": "/reports/" + r["file"]} for r in manifest()
        ],
        "history": [
            {"month": r["month"], "Brent": float(r["Brent"]), "WTI": float(r["WTI"])} for r in history
        ],
    }


@app.get("/reports/{filename}", include_in_schema=False)
def pdf(filename: str):
    if filename not in {row["file"] for row in manifest()}:
        raise HTTPException(404)
    return FileResponse(settings().data / "reports" / filename, media_type="application/pdf")


@app.post("/api/ask")
def query(body: Question):
    try:
        return ask(body.question, body.previous)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
