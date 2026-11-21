"""Public graph replay with scripted model outputs and six invented source fragments."""

from datetime import UTC, datetime

import pytest
from test_agent import plan
from test_agent_loop import action, event

from src.agent import ask

JAN = "Eight participating producers met on 4 January 2026."
NOV = "Seven participating producers met on 1 November 2026."
NEXT = "The next meeting of those producers is scheduled for 29 November 2026."
MARCH = "Participants met on 1 March 2026 to agree on April targets."
COMMITTEE = "The auditing committee held its separate meeting on 15 November 2026."
SCOPE = "страны, участвующие в добровольных ограничениях"
QUESTION = (
    "Когда состоялась последняя встреча стран ОПЕК+ по добровольным ограничениям и что решили?"
)


def fragments():
    rows = [
        ("January meeting", "2026-01-04", JAN + " They kept the previous targets for February."),
        (
            "Participant bulletin",
            None,
            NOV + " They retained September production targets for December 2026. " + NEXT,
        ),
        ("Oversight committee bulletin", "2026-11-16", COMMITTEE),
        ("Spring report", "2026-03-01", MARCH),
        ("January dispatch", "2026-01-04", JAN + " February production was on the agenda."),
        ("January follow-up", "2026-01-05", JAN + " A new meeting was announced for February."),
    ]
    return [
        {
            "kind": "web",
            "url": f"https://example.org/bulletin/{i}",
            "date": day,
            "date_basis": "provider_estimate" if day else "unknown",
            "title": title,
            "text": text,
        }
        for i, (title, day, text) in enumerate(rows, 1)
    ]


def occurrence(date, quote, *, scheduled=False, other_scope=False):
    return {
        "kind": "scheduled" if scheduled else "held",
        "date": date,
        "quote": quote,
        "scope": "отдельный контрольный комитет" if other_scope else SCOPE,
        "match": "different_scope" if other_scope else "in_scope",
    }


class MeetingReplay:
    def __init__(self, draft_mode):
        self.draft_mode = draft_mode
        self.drafts = 0
        self.chronologies = 0

    def generate(self, name, payload, schema):
        if name == "plan":
            return plan(fresh=True, latest_completed_event=True, rag_query="latest meeting")
        if name == "decide":
            selected = (
                action("finish") if payload["observations"] else action("web_search", "meetings")
            )
            return schema.model_validate(selected)
        if name == "event_chronology":
            self.chronologies += 1
            jan = occurrence("2026-01-04", JAN)
            events = [
                [jan],
                [occurrence("2026-11-01", NOV), occurrence("2026-11-29", NEXT, scheduled=True)],
                [occurrence("2026-11-15", COMMITTEE, other_scope=True)],
                [occurrence("2026-03-01", MARCH)],
                [jan],
                [jan],
            ]
            return schema.model_validate(
                {
                    "blocks": [
                        {"evidence_id": f"W{i}:B1", "status": "assessed", "events": block}
                        for i, block in enumerate(events, 1)
                    ]
                }
            )
        if name == "cited_answer":
            self.drafts += 1
            if self.draft_mode == "repair" and self.drafts > 1:
                claims = [
                    {
                        "text": "В найденном сообщении описана встреча семи стран 1 ноября 2026 года.",
                        "evidence_ids": ["W2:B1"],
                    },
                    {
                        "text": "На декабрь сохранили сентябрьские уровни добычи.",
                        "evidence_ids": ["W2:B1"],
                    },
                ]
            elif self.draft_mode == "future":
                claims = [
                    {
                        "text": "Последняя встреча прошла 29 ноября 2026 года.",
                        "evidence_ids": ["W2:B1"],
                    }
                ]
            else:
                claims = [
                    {
                        "text": "Последняя встреча состоялась 4 января 2026 года.",
                        "evidence_ids": ["W1:B1", "W5:B1"],
                    }
                ]
            return schema.model_validate({"claims": claims, "complete": True})
        if name == "citation_review":
            # Replay the original bug: blindly approve any supplied claims.
            return schema.model_validate(
                {
                    "supported": list(range(len(payload["claims"]))),
                    "complete": True,
                    "coverage": "sufficient",
                    "feedback": "",
                }
            )
        raise AssertionError(name)


@pytest.mark.parametrize(
    "question,draft_mode",
    [
        (QUESTION, "repair"),
        (QUESTION, "stale"),
        (QUESTION, "future"),
        ("Когда была последняя встреча ОПЕК+ и что на ней решили?", "repair"),
    ],
)
def test_graph_uses_latest_held_event_even_if_naive_reviewer_would_approve(
    question, draft_mode, monkeypatch
):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    monkeypatch.setattr("src.clock._utc_now", lambda: datetime(2026, 11, 21, tzinfo=UTC))
    model = MeetingReplay(draft_mode)
    calls = []

    def lookup(_registry, name, **_args):
        calls.append(name)
        return {"items": fragments(), "notices": []}, None, event(name)

    response = ask(question, client=model, execute=lookup)
    assert calls == ["web_search"]
    assert model.chronologies == 1
    assert model.drafts == 2
    assert "4 января" not in response["answer"]
    assert "29 ноября" not in response["answer"]
    if draft_mode == "repair":
        assert response["status"] == "ok"
        assert "1 ноября 2026" in response["answer"] and "декабрь" in response["answer"]
        assert "семи стран" in response["answer"]
        assert {source["id"] for source in response["sources"]} == {"W2"}
        assert response["sources"][0]["date"] is None
    else:
        assert response["status"] == "partial" and not response["claims"]
