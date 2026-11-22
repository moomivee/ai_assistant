"""Replay the repeated archival searches in trace 34c346. All event dates are fixtures."""

import httpx
import pytest
from test_agent import plan
from test_agent_loop import action

from src.agent import ask

QUESTION = "когда была последняя встреча опэк?"
QUERY = "последняя встреча ОПЕК"
OFFICIAL_URL = "https://www.opec.org/fixture-conference.html"
OFFICIAL = "Конференция стран-членов ОПЕК состоялась 12 ноября 2026 года."
OLD = "Заседание стран-членов ОПЕК состоялось 27 мая 2009 года."
OLD_AS_LATEST = "Последняя встреча ОПЕК состоялась 27 мая 2009 года."
WRONG_SCOPE = "Восемь стран ОПЕК+ провели встречу 19 ноября 2026 года."
ARCHIVE = [
    (
        "ОПЕК осталась при своих",
        "На очередном заседании ОПЕК сохранила прежние квоты.",
        "2009-05-28",
    ),
    (
        "ОПЕК и страны вне картеля",
        "Представители России и ОПЕК обсуждали сотрудничество.",
        "2015-05-12",
    ),
    (
        "Россия готова участвовать во встрече",
        "Заседание ОПЕК запланировано на 4 декабря.",
        "2015-11-19",
    ),
    ("Восемь стран ОПЕК+ провели встречу", WRONG_SCOPE, "2026-11-20"),
    ("Страны ОПЕК и не-ОПЕК", "Участники договорились мониторить экспорт.", "2017-10-03"),
    ("Россия и ОПЕК", "Россия обсудила контакты с нефтяным альянсом.", "2009-12-22"),
]


class RepeatedSearch:
    def __init__(self, archive_has_event):
        self.archive_has_event = archive_has_event

    def generate(self, name, payload, schema):
        if name == "plan":
            return plan(fresh=True, latest_completed_event=True, rag_query=QUERY)
        if name == "decide":
            step = len(payload["observations"])
            if step and self.archive_has_event:
                return schema.model_validate(action("finish"))
            query = QUERY + {0: " ноябрь 2026", 4: " 2026"}.get(step, "")
            return schema.model_validate(action("web_search", query))
        if name == "event_relevance":
            # The real relevance model admitted every result, including OPEC+.
            return schema.model_validate({"off_topic": []})
        if name == "event_chronology":
            blocks = []
            for key, row in payload["evidence"].items():
                # An explicit meeting of the eight countries must already be excluded.
                assert WRONG_SCOPE not in row["text"]
                events = []
                for quote, date in ((OFFICIAL, "2026-11-12"), (OLD, "2009-05-27")):
                    if quote not in row["text"]:
                        continue
                    events.append(
                        dict(
                            kind="held",
                            date=date,
                            quote=quote,
                            scope="Конференция стран-членов ОПЕК",
                            match="in_scope",
                        )
                    )
                blocks.append(dict(evidence_id=key, status="assessed", events=events))
            return schema.model_validate({"blocks": blocks})
        if name == "cited_answer":
            chronology = payload["event_chronology"]
            ids = chronology["latest_evidence_ids"]
            text = OLD_AS_LATEST if chronology["latest_held_date"] == "2009-05-27" else OFFICIAL
            return schema.model_validate(
                dict(claims=[dict(text=text, evidence_ids=ids)], complete=True)
            )
        assert name == "citation_review"
        return schema.model_validate(
            dict(supported=[0], coverage="sufficient", complete=True, feedback="")
        )


@pytest.mark.parametrize("coverage", ["organizer", "independent", "archive-only"])
@pytest.mark.parametrize("archive_has_event", [False, True], ids=["archive-leads", "archive-date"])
def test_archival_monopoly_triggers_organizer_search(
    monkeypatch, web_today, coverage, archive_has_event
):
    monkeypatch.setenv("AGENT_MAX_STEPS", "6")
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    monkeypatch.setenv("WEB_DOMAINS", "interfax.ru,opec.org,rbc.ru")
    monkeypatch.setenv("TAVILY_API_KEY", "fixture")
    archive = [
        dict(
            url=f"https://www.interfax.ru/business/{i}",
            title=title,
            content=text,
            published_date=day,
        )
        for i, (title, text, day) in enumerate(ARCHIVE, 1)
    ]
    if archive_has_event:
        archive[0]["content"] = OLD
    official = dict(
        url=OFFICIAL_URL,
        title="Итоги конференции ОПЕК",
        content=OFFICIAL,
        published_date="2026-11-13",
    )
    news = {**official, "url": "https://www.rbc.ru/fixture-conference"}
    articles = {row["url"]: row for row in [*archive, official, news]}
    calls = []

    def post(url, *, json, **kwargs):
        operation = url.rsplit("/", 1)[-1]
        calls.append((operation, json))
        if operation == "search":
            domains = json["include_domains"]
            if domains == ["opec.org"]:
                rows = [official] if coverage == "organizer" else []
            elif coverage == "independent" and "rbc.ru" in domains and "interfax.ru" not in domains:
                rows = [news]
            else:
                rows = archive if "interfax.ru" in domains else []
        else:
            assert operation == "extract"
            rows = [dict(url=url, raw_content=articles[url]["content"]) for url in json["urls"]]
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": rows})

    monkeypatch.setattr("src.integrations.web.httpx.post", post)
    result = ask(QUESTION, client=RepeatedSearch(archive_has_event))
    assert result["diagnostics"]["time_context"]["today"] == web_today.isoformat()
    targets = [
        i
        for i, (op, req) in enumerate(calls)
        if op == "search" and req["include_domains"] == ["opec.org"]
    ]
    assert targets and targets[0] <= 2, calls
    assert sum(op == "extract" for op, _ in calls[: targets[0]]) <= 1
    assert WRONG_SCOPE not in result["answer"]
    assert OLD_AS_LATEST not in result["answer"]
    if coverage != "archive-only":
        assert result["status"] == "ok" and OFFICIAL in result["answer"]
        expected = official if coverage == "organizer" else news
        assert {row["url"] for row in result["sources"]} == {expected["url"]}
    else:
        assert result["status"] == "partial"
        assert OFFICIAL not in result["answer"]
        assert all(row["url"] != OFFICIAL_URL for row in result["sources"])
        if archive_has_event:
            assert any(
                date in result["answer"] for date in ("27 мая 2009", "27.05.2009", "2009-05-27")
            )
            assert archive[0]["url"] in {row["url"] for row in result["sources"]}
        else:
            assert not result["claims"]
