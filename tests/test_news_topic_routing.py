"""Current-news selection at the boundary between the model and the tool."""

import pytest
from test_agent import plan
from test_agent_loop import action, event

from src.agent import ask


@pytest.mark.parametrize(
    "question, plan_fields, options, expected_topic",
    [
        ("Какие свежие новости о нефти?", {"news": True, "fresh": True}, {}, "news"),
        (
            "Какие свежие новости о нефти? Сверь с пресс-релизом.",
            {"news": True, "fresh": True},
            {"topic": "general"},
            "general",
        ),
        (
            "Какие новости о нефти вышли за пять дней?",
            {"news": True, "fresh": True},
            {"days": 5},
            "news",
        ),
        ("Что писали о нефти в 2024 году?", {}, {"start_date": "2024-01-01"}, "general"),
        ("Какой сейчас норматив?", {"fresh": True}, {}, "general"),
        (
            "Когда прошло последнее заседание?",
            {"news": True, "fresh": True, "latest_completed_event": True},
            {},
            "general",
        ),
    ],
    ids=["news", "official-document", "five-days", "archive", "current-rule", "last-event"],
)
def test_news_mode_uses_task_and_preserves_explicit_search_choices(
    question, plan_fields, options, expected_topic
):
    calls = []

    class Client:
        def generate(self, name, payload, schema):
            assert name == "decide"
            if calls:
                return schema.model_validate({**action("finish"), "assessment": "unavailable"})
            return schema.model_validate({**action("web_search", "oil"), **options})

    def execute(registry, name, **arguments):
        assert name == "web_search"
        calls.append(arguments)
        return {"items": [], "notices": []}, None, event(name)

    ask(question, client=Client(), planner=lambda *_: plan(**plan_fields), execute=execute)
    assert len(calls) == 1
    assert calls[0].get("topic", "general") == expected_topic
    for field in ("days", "start_date", "end_date"):
        assert calls[0].get(field) == options.get(field)
