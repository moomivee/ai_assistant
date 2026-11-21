"""Replay the public agent with three waves of search results and scripted model decisions."""

from datetime import UTC, datetime

import pytest
from test_agent import plan
from test_agent_loop import action, event

from src.agent import ask

QUESTION = "Когда последний раз собирался совет по газу? Сегодня 21 ноября 2026 года."
OFFICIAL_QUERY = "gas council meeting November 2026 official communiqué"
OUTCOME = "The Gas Council met on 17 November 2026."
CLAIM = "Заседание совета по газу состоялось 17 ноября 2026 года."
FINANCE_TITLE = "Итоги встречи глав финансовых ведомств G20"


def article(url, title, text):
    return {"kind": "web", "url": url, "title": title, "text": text, "date": None}


class DiscoveryReplay:
    def __init__(self):
        self.checked_titles = []
        self.chronology_ids = []
        self.answers = 0

    def generate(self, name, payload, schema):
        if name == "plan":
            return plan(fresh=True, latest_completed_event=True, rag_query="совет по газу")
        if name == "decide":
            step = len(payload["observations"])
            query = ("совет по газу", "заседание совета итоги ноябрь 2026", OFFICIAL_QUERY)[step]
            decision = action("web_search", query)
            if step == 2:
                decision["publisher_domain"] = "council.example"
            return schema.model_validate(decision)
        if name == "event_relevance":
            pages = payload["publications"]
            self.checked_titles.extend([p["title"] for p in pages])
            return schema.model_validate(
                {"off_topic": [p["index"] for p in pages if p["title"] == FINANCE_TITLE]}
            )
        if name == "event_chronology":
            blocks = []
            for key, row in payload["evidence"].items():
                self.chronology_ids.append(key)
                assert FINANCE_TITLE != row["title"]
                events = []
                if row["text"] == OUTCOME:
                    events.append(
                        dict(
                            kind="held",
                            date="2026-11-17",
                            quote=OUTCOME,
                            scope="совет по газу",
                            match="in_scope",
                        )
                    )
                blocks.append(dict(evidence_id=key, status="assessed", events=events))
            return schema.model_validate({"blocks": blocks})
        if name == "cited_answer":
            self.answers += 1
            assert payload["event_chronology"]["latest_held_date"] == "2026-11-17"
            assert "W2:B1" not in payload["evidence"]
            assert all(source["id"] != "W2" for source in payload["sources"])
            return schema.model_validate(
                {"claims": [{"text": CLAIM, "evidence_ids": ["W3:B1"]}], "complete": True}
            )
        assert name == "citation_review"
        return schema.model_validate(
            dict(supported=[0], coverage="sufficient", complete=True, feedback="")
        )


@pytest.mark.parametrize("has_minutes", [True, False])
def test_recover_from_irrelevant_batch_and_seek_unseen_organizer(monkeypatch, has_minutes):
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    monkeypatch.setenv("WEB_DOMAINS", "council.example,news.example,finance.example")
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    monkeypatch.setattr("src.clock._utc_now", lambda: datetime(2026, 11, 21, tzinfo=UTC))
    model = DiscoveryReplay()
    searches = []
    initial = article("https://news.example/agenda", "Совет по газу", "Повестка ещё обсуждается.")
    unrelated = article(
        "https://finance.example/meeting", FINANCE_TITLE, "Финансисты G20 обсудили ставки."
    )
    minutes = article("https://council.example/minutes", "Gas Council communiqué", OUTCOME)
    results = [[initial], [unrelated], [minutes] if has_minutes else []]

    def lookup(registry, name, **kwargs):
        assert name == "web_search"
        searches.append(kwargs)
        return {"items": results[len(searches) - 1], "notices": []}, None, event(name)

    response = ask(QUESTION, client=model, execute=lookup)
    assert searches[-1]["query"] == OFFICIAL_QUERY
    assert searches[-1]["publisher_domain"] == "council.example"
    assert model.checked_titles == [
        initial["title"],
        unrelated["title"],
        *([minutes["title"]] if has_minutes else []),
    ]
    assert "W2:B1" not in model.chronology_ids
    assert model.chronology_ids.count("W1:B1") == 1
    if has_minutes:
        assert response["status"] == "ok"
        assert response["sources"][0]["url"] == minutes["url"]
        assert CLAIM in response["answer"] and FINANCE_TITLE not in response["answer"]
        assert model.answers == 1 and model.chronology_ids.count("W3:B1") == 1
    else:
        assert response["status"] == "partial" and not response["claims"]
        assert model.answers == 0 and "17 ноября" not in response["answer"]
