"""Shared trace ids for Langfuse, native LangGraph callbacks and local JSONL."""

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
from time import perf_counter
from uuid import uuid4

from filelock import FileLock, Timeout

from ..config import settings
from ..logging import log, request_id

_trace = ContextVar("analyst_trace", default=None)
_parent = ContextVar("analyst_span", default=None)


def mask(data, **_):
    return data if settings().trace_content else "[content disabled]"


@lru_cache
def langfuse_client():
    if not settings().langfuse_enabled:
        return None
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        log("langfuse.missing_credentials", logging.WARNING)
        return None
    from langfuse import Langfuse

    try:
        return Langfuse(mask=mask, timeout=5)
    except Exception as exc:
        log("langfuse.init_failed", logging.WARNING, error_type=type(exc).__name__)
        return None


def callbacks():
    if langfuse_client() is None:
        return []
    from langfuse.langchain import CallbackHandler

    try:
        return [CallbackHandler(public_key=os.environ["LANGFUSE_PUBLIC_KEY"])]
    except Exception as exc:
        log("langfuse.callback_failed", logging.WARNING, error_type=type(exc).__name__)
        return []


def write_event(event):
    path = settings().trace_file
    if path:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock", timeout=5):
                with path.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except (OSError, Timeout):
            log("trace.write_failed", logging.ERROR)


class Observation:
    def __init__(self):
        self.remote = None
        self.output = None
        self.metadata = {}
        self.status = "ok"

    def update(self, *, output=None, metadata=None, status="ok", **kwargs):
        self.output = output
        self.metadata.update(metadata or {})
        self.status = status
        if self.remote is not None:
            try:
                self.remote.update(
                    output=mask(output),
                    metadata=metadata,
                    level="ERROR" if status == "error" else "DEFAULT",
                    **kwargs,
                )
            except Exception as exc:
                log(
                    "langfuse.update_failed",
                    logging.WARNING,
                    error_type=type(exc).__name__,
                )


@contextmanager
def span(name, kind="span", input=None, metadata=None, **kwargs):
    start, trace_id = perf_counter(), _trace.get() or uuid4().hex
    span_id, parent = uuid4().hex[:16], _parent.get()
    token = _parent.set(span_id)
    obs, remote_cm = Observation(), None
    try:
        client = langfuse_client()
        if client:
            try:
                remote_cm = client.start_as_current_observation(
                    name=name,
                    as_type=kind,
                    input=mask(input),
                    metadata=metadata,
                    trace_context={"trace_id": trace_id} if parent is None else None,
                    **kwargs,
                )
                obs.remote = remote_cm.__enter__()
            except Exception as exc:
                remote_cm = None
                log(
                    "langfuse.span_failed",
                    logging.WARNING,
                    error_type=type(exc).__name__,
                )
        yield obs
    except Exception as exc:
        obs.update(status="error", metadata={"error_type": type(exc).__name__})
        raise
    finally:
        elapsed = round((perf_counter() - start) * 1000, 2)
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": parent,
            "name": name,
            "kind": kind,
            "status": obs.status,
            "ms": elapsed,
            "metadata": {**(metadata or {}), **obs.metadata},
        }
        if settings().trace_content:
            event.update(input=input, output=obs.output)
        write_event(event)
        log("span.finished", name=name, trace_id=trace_id, status=obs.status, ms=elapsed)
        try:
            if remote_cm:
                remote_cm.__exit__(None, None, None)
        except Exception as exc:
            log("langfuse.end_failed", logging.WARNING, error_type=type(exc).__name__)
        finally:
            _parent.reset(token)


@contextmanager
def request_trace(question, previous=None):
    trace_id, incoming = uuid4().hex, request_id.get()
    token, rid = _trace.set(trace_id), request_id.set(trace_id)
    try:
        with span(
            "agent.request",
            "agent",
            {"question": question, "previous": previous},
            metadata={"http_request_id": incoming if incoming != "-" else None},
        ) as obs:
            yield trace_id, obs
    finally:
        request_id.reset(rid)
        _trace.reset(token)


def flush():
    client = langfuse_client()
    if client:
        try:
            client.flush()
        except Exception as exc:
            log("langfuse.flush_failed", logging.WARNING, error_type=type(exc).__name__)
