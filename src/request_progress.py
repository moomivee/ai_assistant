"""Request-scoped progress for browser clients observing a running agent."""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

_listener: ContextVar[Callable[[str], None] | None] = ContextVar("progress_listener", default=None)


@contextmanager
def report_progress_to(listener: Callable[[str], None]):
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


def announce_quote_fallback():
    if listener := _listener.get():
        listener("quote_fallback")
