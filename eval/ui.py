"""Local evaluation launcher: uv run python -m eval.ui (127.0.0.1:8001)."""

import threading
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import ROOT
from src.logging import log

from .run import run

app = FastAPI(title="Оценка нефтегазового аналитика")
_jobs = {}
_lock = threading.Lock()


class Start(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["retrieval", "routing", "agent"]
    split: Literal["all", "dev", "test"] = "dev"
    judge: bool = False
    limit: int = Field(default=5, ge=1, le=100)


def worker(job_id, args):
    path = ROOT / ".cache/eval" / f"{job_id}.json"
    try:
        result = run(**args, output=path)
        with _lock:
            _jobs[job_id].update(status="done", summary=result["summary"])
    except Exception as exc:
        log("eval.failed", error_type=type(exc).__name__)
        with _lock:
            _jobs[job_id].update(
                status="error",
                error="Оценка не завершена. Подробности и частичные результаты сохранены локально.",
            )


@app.get("/")
def home():
    return FileResponse(Path(__file__).with_name("ui.html"))


@app.post("/runs", status_code=202)
def start(body: Start, tasks: BackgroundTasks):
    if body.mode == "routing" and body.judge:
        raise HTTPException(422, "Judge доступен только для полного agent eval.")
    with _lock:
        if any(j["status"] == "running" for j in _jobs.values()):
            raise HTTPException(409, "Оценка уже выполняется.")
        job = uuid4().hex
        _jobs[job] = {"id": job, "status": "running"}
    tasks.add_task(worker, job, body.model_dump())
    return {"id": job}


@app.get("/runs/{job_id}")
def progress(job_id: str):
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(404)
        result = dict(_jobs[job_id])
    path = ROOT / ".cache/eval" / f"{job_id}.jsonl"
    result["completed_rows"] = len(path.read_text().splitlines()) if path.exists() else 0
    return result


@app.get("/runs/{job_id}/result")
def download(job_id: str):
    with _lock:
        if job_id not in _jobs or _jobs[job_id]["status"] != "done":
            raise HTTPException(404)
    return FileResponse(ROOT / ".cache/eval" / f"{job_id}.json", filename=f"eval-{job_id}.json")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
