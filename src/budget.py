"""Cooperative request deadline shared by graph nodes and outbound clients."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

_deadline = ContextVar("agent_deadline", default=None)


class RequestBudgetExceeded(RuntimeError):
    def __init__(self):
        super().__init__("Достигнут общий лимит времени запроса.")


@contextmanager
def request_budget(seconds):
    token = _deadline.set(monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_seconds():
    deadline = _deadline.get()
    return None if deadline is None else max(0.0, deadline - monotonic())


def check_budget():
    if remaining_seconds() == 0:
        raise RequestBudgetExceeded()


def bounded_timeout(maximum):
    check_budget()
    remaining = remaining_seconds()
    return maximum if remaining is None else min(maximum, remaining)


class BudgetedClient:
    """Also check between multiple generations inside one planning/answer node."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def generate(self, *args, **kwargs):
        check_budget()
        return self.client.generate(*args, **kwargs)
