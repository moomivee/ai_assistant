"""Trusted request time, independent of model knowledge and document dates."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from .config import settings

_request_time: ContextVar[datetime | None] = ContextVar("request_time", default=None)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _local_now() -> datetime:
    return _utc_now().astimezone(ZoneInfo(settings().app_timezone))


@contextmanager
def request_clock():
    """Freeze one timestamp for all stages; nested scopes retain their parent clock."""
    current = _request_time.get()
    if current is not None:
        yield current
        return
    current = _local_now()
    token = _request_time.set(current)
    try:
        yield current
    finally:
        _request_time.reset(token)


def clock_payload() -> dict:
    """A fresh standalone reading, or the stable clock of the active request."""
    current = _request_time.get() or _local_now()
    return {
        "today": current.date().isoformat(),
        "current_time": current.isoformat(timespec="seconds"),
        "timezone": current.tzinfo.key,
        "current_year": current.year,
    }


def clock_today() -> date:
    """The same reference date used by prompts and time-sensitive tool filters."""
    return (_request_time.get() or _local_now()).date()


def clock_instruction(clock: dict) -> str:
    """System-level reference time; historical questions retain their explicit period."""
    return (
        "\n\nСлужебное время приложения, зафиксированное в начале запроса: "
        f"сегодня {clock['today']}, текущий год {clock['current_year']}, "
        f"время {clock['current_time']}, часовой пояс {clock['timezone']}. "
        "Это опорная дата для слов «сегодня», «сейчас», «последний» и «актуальный». "
        "Не определяй сегодняшнюю дату по знаниям модели, прежним ответам или датам источников. "
        "Явно указанный пользователем исторический период сохраняет своё значение. "
        "Различай дату события, публикации/обновления источника и его загрузки: "
        "свежая загрузка старой страницы не доказывает актуальность события. "
        "Старый отчёт сам по себе не подтверждает отсутствие более нового события."
    )
