from datetime import timedelta

import pytest

from src.event_archive import archive_event_gap, archive_verification_search


def archived():
    state = {
        "question": "Когда была последняя встреча ОПЕК?",
        "plan": {"latest_completed_event": True, "fresh": True},
        "observations": [],
        "evidence": {
            "W1:B1": {"kind": "web", "url": "https://www.interfax.ru/meeting", "date": "2009-05-28"}
        },
    }
    validation = {
        "event_chronology": {"latest_held_date": "2009-05-27", "latest_evidence_ids": ["W1:B1"]}
    }
    return state, validation


def test_archive_keeps_dated_evidence_and_checks_organizer_only_once(web_today):
    state, validation = archived()
    gap = archive_event_gap(state, validation)
    assert gap["dated_claim"]["evidence_ids"] == ["W1:B1"]
    assert "27.05.2009" in gap["dated_claim"]["text"]
    assert "Более поздняя встреча не подтверждена" in gap["notice"]
    action = archive_verification_search(state, gap)
    assert action["action"] == "web_search" and action["publisher_domain"] == "opec.org"
    assert "official" in action["query"] and "2009" not in action["query"]
    assert action["start_date"] is None and action["end_date"] is None
    state["observations"].append(
        {
            "tool": "web_search",
            "arguments": {"search": True, "publisher_domain": "opec.org"},
            "result_count": 0,
        }
    )
    assert archive_event_gap(state, validation) == gap
    second = archive_verification_search(state, gap)
    assert second["omit_publisher_domain"] == "interfax.ru"
    state["observations"].append(
        {"tool": "web_search", "arguments": {"search": True, **second}, "result_count": 0}
    )
    assert archive_event_gap(state, validation) == gap
    assert archive_verification_search(state, gap) is None


@pytest.mark.parametrize(
    "params",
    [
        {"latest_completed_event": False},
        {"fresh": False},
        {"report_only": True},
        {"event_source_limit": "только по РБК"},
        {"event_publication_window": {"quote": "в газетах за 2009 год", "end_date": "2009-12-31"}},
    ],
)
def test_historical_and_source_scoped_questions(params):
    state, validation = archived()
    state["question"] += " Проверь только по РБК в газетах за 2009 год."
    state["plan"].update(params)
    assert archive_event_gap(state, validation) is None


def test_only_sources_that_establish_the_chronology_can_resolve_an_archive_gap(web_today):
    state, validation = archived()
    state["evidence"]["W2:B1"] = {"kind": "web", "url": "https://opec.org/releases", "date": None}
    assert archive_event_gap(state, validation)
    validation["event_chronology"]["latest_evidence_ids"].append("W2:B1")
    assert archive_event_gap(state, validation) is None


def test_verification_respects_only_rbc_even_if_model_omits_source_limit():
    state, validation = archived()
    state["question"] += " Используй только РБК."
    state["evidence"]["W1:B1"]["url"] = "https://rbc.ru/meeting"
    assert archive_event_gap(state, validation) is None


def test_publication_age_boundary_is_not_an_event_date(web_today):
    state, validation = archived()
    block = state["evidence"]["W1:B1"]
    block["date"] = (web_today - timedelta(days=365)).isoformat()
    assert archive_event_gap(state, validation) is None
    block["date"] = (web_today - timedelta(days=366)).isoformat()
    assert archive_event_gap(state, validation)["event_date"] == "2009-05-27"


def test_ungrounded_or_non_whitelisted_organizer_does_not_trigger_search(monkeypatch):
    monkeypatch.setenv("WEB_DOMAINS", "interfax.ru")
    state, validation = archived()
    assert archive_event_gap(state, validation) is None


def test_unregistered_organizer_checks_independent_media_once(monkeypatch, web_today):
    monkeypatch.setenv("WEB_DOMAINS", "interfax.ru,rbc.ru")
    state, validation = archived()
    state["question"] = "Когда прошла последняя встреча газового совета?"
    gap = archive_event_gap(state, validation)
    assert gap["organizer_domain"] == ""
    first = archive_verification_search(state, gap)
    assert first["omit_publisher_domain"] == "interfax.ru" and not first.get("publisher_domain")
    state["observations"].append({"tool": "web_search", "arguments": {"search": True, **first}})
    assert archive_verification_search(state, gap) is None
    state["evidence"]["W2:B1"] = {
        "kind": "web",
        "url": "https://rbc.ru/meeting",
        "date": web_today.isoformat(),
    }
    validation["event_chronology"]["latest_evidence_ids"].append("W2:B1")
    assert archive_event_gap(state, validation) is None
