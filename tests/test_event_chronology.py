from datetime import UTC, date, datetime

import pytest

from src import clock
from src.event_chronology import build_event_chronology, mentioned_dates
from src.integrations.prompts import get_prompt


def event(date, quote, *, kind="held", match="in_scope"):
    return dict(date=date, quote=quote, kind=kind, match=match, scope="Комитет")


def block(key, *events, status="assessed"):
    return dict(evidence_id=key, status=status, events=list(events))


def chronology(monkeypatch, evidence, blocks):
    monkeypatch.setattr(clock, "_utc_now", lambda: datetime(2026, 11, 21, 8, tzinfo=UTC))

    class Reader:
        def generate(self, name, payload, schema):
            assert name == "event_chronology" and get_prompt(name).text
            assert payload["today"] == "2026-11-21"
            assert payload["previous"] == "Вопрос про комитет"
            assert {k: v["text"] for k, v in payload["evidence"].items()} == {
                k: v["text"] for k, v in evidence.items()
            }
            return schema.model_validate({"blocks": blocks})

    return build_event_chronology(
        Reader(), "Когда состоялось последнее заседание?", evidence, previous="Вопрос про комитет"
    )


def test_compares_all_sources_and_disregards_publication_dates_and_plans(monkeypatch):
    january = "The committee convened on January 4, 2026."
    november = "Члены комитета встретились 1 ноября 2026 года."
    planned = "Следующая встреча назначена на 29 ноября 2026 года."
    other = "Наблюдательный орган провёл заседание 15 ноября 2026 года."
    sources = {
        "W1:B1": {"text": january, "date": "2026-11-20"},
        "W2:B1": {"text": november + " " + planned, "date": None},
        "W3:B1": {"text": other, "date": "2026-11-18"},
        "W4:B1": {"text": "Решение вступает в силу 1 декабря 2026 года.", "date": None},
    }
    result = chronology(
        monkeypatch,
        sources,
        [
            block("W1:B1", event("2026-01-04", january)),
            block("W3:B1", event("2026-11-15", other, match="different_scope")),
            block(
                "W2:B1",
                event("2026-11-01", november),
                event("2026-11-29", planned, kind="scheduled"),
            ),
            block("W4:B1"),
        ],
    )
    assert result["latest_held_date"] == "2026-11-01"
    assert result["latest_evidence_ids"] == ["W2:B1"]
    assert [r["date"] for r in result["scheduled_events"]] == ["2026-11-29"]
    assert result["unresolved_evidence_ids"] == []


def test_uncertain_or_invalid_candidates_are_visible_to_answer_gate(monkeypatch):
    schedule = "Событие запланировано на 29 ноября 2026 года."
    prior = "Встреча состоялась 1 ноября 2026 года."
    sources = {
        f"W{i}:B1": {"text": prior + " " + schedule, "date": "2026-11-01"} for i in range(1, 8)
    }
    result = chronology(
        monkeypatch,
        sources,
        [
            block("W1:B1", event("2026-11-29", schedule)),  # held after the request date
            block("W2:B1", event("2026-11-01", "It met on 1 November 2026.")),  # wrong excerpt
            block(
                "W3:B1", event("2026-11-01", "Встреча состоялась")
            ),  # only metadata could date this
            block("W4:B1", event("2026-11-01", prior), status="uncertain"),
            block("W5:B1", event("2026-11-01", prior + " " + schedule)),  # ambiguous date binding
            block("W6:B1"),
            block("W6:B1"),  # duplicate review
            # No review at all for W7.
        ],
    )
    assert result["unresolved_evidence_ids"] == sorted(sources)
    assert result["latest_evidence_ids"] == []


def test_no_age_cutoff_or_assumption_that_a_past_scheduled_event_happened(monkeypatch):
    prior = "The commission last met on 14 April 1994."
    proposal = "A session was planned for 6 March 2001."
    result = chronology(
        monkeypatch,
        {"R1:B1": {"text": prior + " " + proposal}},
        [
            block(
                "R1:B1", event("1994-04-14", prior), event("2001-03-06", proposal, kind="scheduled")
            )
        ],
    )
    assert result["latest_held_date"] == "1994-04-14"
    assert result["unresolved_evidence_ids"] == []


@pytest.mark.parametrize(
    "text", ["21 Nov. 2026", "Nov 21st, 2026", "21 ноября 2026 г.", "2026-11-21", "21.11.2026"]
)
def test_dates_must_be_fully_spelled(text):
    assert mentioned_dates(text) == {date(2026, 11, 21)}
    assert mentioned_dates("30 февраля 2026 года, ноябрь 2026") == set()
    assert mentioned_dates("Результат опубликован на https://example.org/2026-11-01/press") == set()
    assert mentioned_dates("29 ноября") == set()
    assert mentioned_dates("29 ноября и 1 ноября 2025", default_year=2026) == {
        date(2026, 11, 29),
        date(2025, 11, 1),
    }
