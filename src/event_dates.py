"""Calendar expressions in Russian and English sources and search queries."""

import re
from datetime import date

# The model classifies events and their scope. Code checks that its ISO date
# is an actual full date in the cited passage, including localized month names.
_MONTHS = (
    "январь январе january jan января янв",
    "февраль феврале february feb февраля фев февр",
    "март марте march mar марта мар",
    "апрель апреле april apr апреля апр",
    "май мае may мая",
    "июнь июне june jun июня июн",
    "июль июле july jul июля июл",
    "август августе august aug августа авг",
    "сентябрь сентябре september sept sep сентября сен сент",
    "октябрь октябре october oct октября окт",
    "ноябрь ноябре november nov ноября ноя нояб",
    "декабрь декабре december dec декабря дек",
)
_MONTH_NUMBER = {word: i for i, names in enumerate(_MONTHS, 1) for word in names.split()}
_MONTH = "(?:" + "|".join(sorted(_MONTH_NUMBER, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+(\d{4}))?(?!\w)"
_DAY_FIRST = re.compile(rf"(?<!\w){_DAY}\s+(?:of\s+)?({_MONTH}){_YEAR}", re.I)
_MONTH_FIRST = re.compile(rf"(?<!\w)({_MONTH})\s+{_DAY}{_YEAR}", re.I)


def mentioned_dates(text: str, *, default_year: int | None = None) -> set[date]:
    """Find explicit dates; a caller can supply a year for day + named month in user-facing prose."""
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text, flags=re.I)
    parts = [
        (year, month, day)
        for year, month, day in re.findall(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", text)
    ]
    parts.extend(
        (year, month, day)
        for day, month, year in re.findall(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)", text)
    )
    parts.extend(
        (year or default_year, _MONTH_NUMBER[month.lower().rstrip(".")], day)
        for day, month, year in _DAY_FIRST.findall(text)
    )
    parts.extend(
        (year or default_year, _MONTH_NUMBER[month.lower().rstrip(".")], day)
        for month, day, year in _MONTH_FIRST.findall(text)
    )
    found = set()
    for year, month, day in parts:
        if not year:
            continue
        try:
            found.add(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return found


_SEARCH_PERIOD = re.compile(
    rf"(?<!\w)(?:(?:{_DAY}\s+(?:of\s+)?)?(?P<month>{_MONTH})\s+"
    rf"(?:{_DAY},?\s+)?(?P<year>\d{{4}})"
    r"|(?P<iso_year>\d{4})-(?P<iso_month>0[1-9]|1[0-2])(?:-\d{2})?"
    r"|\d{1,2}\.(?P<ru_month>\d{1,2})\.(?P<ru_year>\d{4}))(?!\w)",
    re.I,
)


def open_event_query(query, *, question, lead_date=""):
    """Drop unsupported month/year filters from an open discovery query.

    An explicit user period or a chosen dated lead may improve retrieval.
    The current month alone is not evidence of when the last event occurred.
    """

    def period(match):
        if match["month"]:
            return int(match["year"]), _MONTH_NUMBER[match["month"].lower().rstrip(".")]
        return int(match["iso_year"] or match["ru_year"]), int(
            match["iso_month"] or match["ru_month"]
        )

    allowed = {period(m) for m in _SEARCH_PERIOD.finditer(question)}
    if lead_date:
        day = date.fromisoformat(lead_date)
        allowed.add((day.year, day.month))
    return " ".join(
        _SEARCH_PERIOD.sub(lambda m: m[0] if period(m) in allowed else " ", query).split()
    )
