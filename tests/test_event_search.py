"""Current-event searches preserve the user's scope even after an irrelevant result."""

from datetime import date

import pytest

from src.agent import action_schema
from src.event_search import event_publication_search, event_search_context
from src.integrations.web import WEB_QUERY_LIMIT


def research(**changes):
    state = {
        "question": "Когда был последний совет по энергетике?",
        "previous": None,
        "plan": {"latest_completed_event": True, "rag_query": "совет в сентябре 2024"},
        "draft": {"complete": False},
        "observations": [{"tool": "web_search", "arguments": {"search": True}}],
        "answer_checks": [
            {
                "validation": {
                    "event_chronology": {
                        "latest_held_date": None,
                        "overdue_scheduled_dates": ["2024-09-15"],
                    }
                }
            }
        ],
        "sources": [
            {"id": "W1", "kind": "web", "url": "https://www.iea.org/news"},
            {"id": "W2", "kind": "web", "url": "https://untrusted.invalid/archive"},
        ],
    }
    state.update(changes)
    return state


def search(state, **kwargs):
    raw = (
        action_schema(("read_web", "web_search", "finish"), event_followup=True)
        .model_validate(
            {
                "action": "web_search",
                "purpose": "Проверить дату",
                "query": "совет сентябрь 2024",
                "days": 14,
                **kwargs,
            }
        )
        .model_dump()
    )
    return event_publication_search(raw, event_search_context(state), state)


def test_open_discovery_drops_invented_period_in_both_plan_and_action():
    state = research()
    actual = search(
        state, event_language="en", event_lead_date="2024-09-01", event_publisher_id="W2"
    )
    assert actual["query"] == "совет"
    assert not actual["publisher_domain"] and actual["days"] is None
    assert actual["argument_repair"]["before"]["query"] == "совет сентябрь 2024"
    fallback = search(state, query="", event_language="en")
    assert "official" in fallback["query"] and "2024" not in fallback["query"]


def test_current_month_hypothesis_is_a_search_term_not_a_date_filter(monkeypatch):
    monkeypatch.setattr("src.event_search.clock_today", lambda: date(2026, 11, 21))
    state = research(question="Когда прошла последняя встреча ОПЕК+? Сегодня 2026 год 21 ноября.")
    query = "последняя встреча ОПЕК+ ноябрь 2026"
    actual = search(state, query=query)
    assert actual["query"] == query and actual["days"] is None
    assert actual["start_date"] is actual["end_date"] is None


def test_first_search_can_target_unseen_trusted_organizer_in_its_own_language(monkeypatch):
    monkeypatch.setenv("WEB_DOMAINS", "iea.org,other.example")
    state = research(sources=[], observations=[], answer_checks=[])
    assert event_search_context(state)["trusted_publishers"] == ["iea.org", "other.example"]
    query = "IEA ministerial meeting official communiqué"
    actual = search(state, query=query, publisher_domain="iea.org", days=None)
    assert (actual["query"], actual["publisher_domain"]) == (query, "iea.org")
    assert not search(state, query=query, publisher_domain="untrusted.invalid")["publisher_domain"]


def test_verified_lead_supports_focused_check_and_does_not_set_publication_cutoff():
    actual = search(research(), event_lead_date="2024-09-15", event_publisher_id="W1")
    assert "2024-09-15" in actual["query"] and actual["publisher_domain"] == "www.iea.org"
    assert actual["days"] is actual["start_date"] is actual["end_date"] is None


@pytest.mark.parametrize(
    "period", ["September 1st, 2024", "1.09.2024", "2024-09-01", "1 сентября 2024"]
)
def test_explicit_historical_period_in_question_remains_a_search_term(period):
    query = f"Gas Council press release {period}"
    assert search(research(), query=query)["query"] == "Gas Council press release"
    question = "Какое последнее заседание состоялось к 30 сентября 2024 года?"
    assert search(research(question=question), query=query)["query"] == query


@pytest.mark.parametrize(
    "changes",
    [
        {"plan": {"latest_completed_event": False}},
        {"plan": {"latest_completed_event": True, "report_only": True}},
        {"draft": {"complete": True}},
    ],
)
def test_other_scopes_and_completed_answers_are_unaffected(changes):
    assert event_search_context(research(**changes)) is None


def test_direct_quote_is_not_a_completed_publication_search():
    state = research(observations=[{"tool": "web_search", "arguments": {"search": False}}])
    assert not event_search_context(state)["followup"]


def test_untrusted_bounds_dont_override_user_wording():
    question = "Когда прошла последняя встреча? Используй публикации за последние 14 дней."
    actual = search(research(question=question), start_date="2020-01-01", days=360)
    assert question not in actual["query"]
    assert actual["start_date"] is actual["end_date"] is actual["days"] is None


def test_excerpts_from_one_url_stay_grouped_for_publication_selection():
    state = research()
    state["sources"].append({**state["sources"][0], "id": "W8", "text": "other excerpt"})
    context = event_search_context(state)
    assert context["publisher_pages"] == {"https://www.iea.org/news": ["W1", "W8"]}
    assert search(state, event_publisher_id="W8")["publisher_domain"] == "www.iea.org"
    state["answer_checks"][-1]["validation"]["event_relevance"] = {
        "excluded_source_ids": ["W1", "W8"]
    }
    assert not event_search_context(state)["publisher_pages"]
    assert not search(state, event_publisher_id="W8")["publisher_domain"]
    assert "iea.org" in event_search_context(state)["trusted_publishers"]


def test_elliptical_question_restores_user_grounded_entity_from_previous_turn():
    state = research(
        question="Когда они встречались в последний раз?", previous="Меня интересует Газовый форум."
    )
    assert "Газовый форум" in search(state, event_subject="Газовый форум")["query"]
    assert "Октябрь" not in search(state, event_subject="Октябрь")["query"]


def test_long_question_keeps_focal_subject_publisher_and_date_in_actual_provider_limit():
    state = research(question="У меня есть подробный вопрос. " * 20 + "Когда был Энергофорум?")
    actual = search(
        state,
        query="council meeting " * 60,
        event_subject="Энергофорум",
        event_publisher_id="W1",
        event_lead_date="2024-09-15",
    )
    q = actual["query"]
    assert len(q) <= WEB_QUERY_LIMIT and "Энергофорум" in q
    assert q.endswith("2024-09-15") and actual["publisher_domain"] == "www.iea.org"
