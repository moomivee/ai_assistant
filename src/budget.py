"""Cooperative request deadline shared by graph nodes and outbound clients."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

_deadline = ContextVar("agent_deadline", default=None)
_research_deadline = ContextVar("agent_research_deadline", default=None)
_research_phase = ContextVar("agent_research_phase", default=False)


class RequestBudgetExceeded(RuntimeError):
    def __init__(self):
        super().__init__("Достигнут общий лимит времени запроса.")


class ResearchBudgetExceeded(RequestBudgetExceeded):
    def __init__(self):
        RuntimeError.__init__(self, "Поиск остановлен, чтобы сохранить время на ответ.")


@contextmanager
def request_budget(seconds, answer_reserve=0):
    deadline = monotonic() + seconds
    token = _deadline.set(deadline)
    research = _research_deadline.set(deadline - min(answer_reserve, seconds / 2))
    try:
        yield
    finally:
        _research_deadline.reset(research)
        _deadline.reset(token)


@contextmanager
def research_phase():
    token = _research_phase.set(True)
    try:
        yield
    finally:
        _research_phase.reset(token)


def remaining_seconds():
    deadline = _research_deadline.get() if _research_phase.get() else _deadline.get()
    return None if deadline is None else max(0.0, deadline - monotonic())


def check_budget():
    if _deadline.get() is not None and monotonic() >= _deadline.get():
        raise RequestBudgetExceeded()
    if remaining_seconds() == 0:
        raise ResearchBudgetExceeded()


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
