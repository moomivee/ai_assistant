"""Reduced replay of trace 4d71e0ee using its question, dates and short archive excerpts."""

from functools import partial

import pytest
from fastapi.testclient import TestClient
from test_agent import plan
from test_agent_loop import action, event

from src import api
from src.agent import ask
from src.integrations.llm import LLMUnavailable
from src.integrations.web import SEARCH_DATE_NOTICE

QUESTION = (
    "Когда страны ОПЕК+, добровольно корректирующие добычу, встречались в последний раз? "
    "А когда они планируют встретиться снова?"
)
SCHEDULE_URL = "https://www.interfax.ru/business/1046054"
MEETING_URL = "https://www.interfax.ru/business/1046051"
EMPTY_NOTICE = "В доверенных источниках не найдено подходящей публикации с заданными параметрами."
ARCHIVE_DAY = "2025-09-07"
SCHEDULE = 'Следующая встреча "восьмерки" ОПЕК+ пройдет 5 октября.'
MEETING = (
    "Москва. 7 сентября. INTERFAX.RU - Министры восьми стран ОПЕК+, участвующих "
    "в добровольных ограничениях добычи нефти, начали встречу по обсуждению плана "
    "производства на октябрь."
)
TITLES = {
    SCHEDULE_URL: '"Восьмерка" ОПЕК+ в октябре увеличит добычу нефти на 137 тыс. б/с',
    MEETING_URL: "Министры восьми стран ОПЕК+ начали встречу по обсуждению плана добычи нефти на октябрь",
}


class ArchiveReplay:
    def __init__(self, outcome="archive"):
        self.outcome = outcome

    def generate(self, name, payload, schema):
        if name == "plan":
            if self.outcome == "failure":
                raise LLMUnavailable("Fixture provider unavailable")
            return plan(fresh=True, latest_completed_event=True, rag_query="ОПЕК+ встречи 2026")
        if name == "decide":
            step = len(payload["observations"])
            if step < 2:
                terms = ["ОПЕК+ планы на 2026 год", "ОПЕК+ последняя встреча и планы"][step]
                return schema.model_validate(action("web_search", terms))
            url = [SCHEDULE_URL, MEETING_URL][step - 2]
            return schema.model_validate({**action("read_web"), "url": url})
        if name == "event_relevance":
            return schema.model_validate(
                {
                    "off_topic": [p["index"] for p in payload["publications"]]
                    if self.outcome == "unrelated"
                    else []
                }
            )
        assert name == "event_chronology"
        return schema.model_validate(
            {
                "blocks": [
                    {"status": "uncertain", "evidence_id": key, "events": []}
                    for key in payload["evidence"]
                ],
                "requested_context": ["previous_held", "next_scheduled"],
            }
        )

    def execute(self, registry, name, **args):
        if name == "read_web":
            url = args["url"]
            items = [self.article(url, SCHEDULE if url == SCHEDULE_URL else MEETING, "extracted")]
            return {"items": items, "notices": []}, None, event(name)
        assert name == "web_search"
        if self.outcome == "empty" or "планы на" in args["query"]:
            return {"items": [], "notices": [EMPTY_NOTICE]}, None, event(name)
        if self.outcome == "unrelated":
            items = [
                dict(
                    self.article(
                        MEETING_URL, "Совет обсуждал заседание финансового комитета.", "snippet"
                    ),
                    title="Отчёт финансового комитета",
                )
            ]
        else:
            items = [
                self.article(SCHEDULE_URL, SCHEDULE + " ... Министры обсудили добычу.", "snippet"),
                self.article(
                    MEETING_URL, "В число стран добровольцев ОПЕК+ входит восемь стран.", "snippet"
                ),
            ]
        return {"items": items, "notices": [SEARCH_DATE_NOTICE]}, None, event(name)

    @staticmethod
    def article(url, text, mode):
        return dict(
            kind="web",
            url=url,
            title=TITLES[url],
            text=text,
            date=ARCHIVE_DAY if mode == "extracted" else None,
            date_basis="provider_estimate" if mode == "extracted" else "unknown",
            content_type=mode,
            provider="tavily",
        )


def ask_on_workbench(monkeypatch, outcome):
    monkeypatch.setenv("AGENT_MAX_STEPS", "4" if outcome == "archive" else "2")
    monkeypatch.setenv("WEB_DOMAINS", "interfax.ru")
    model = ArchiveReplay(outcome)
    monkeypatch.setattr(api, "ask", partial(ask, client=model, execute=model.execute))
    response = TestClient(api.app).post("/api/ask", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    return response.json()


def test_partial_api_shows_archive_excerpt_but_keeps_both_current_dates_open(
    monkeypatch, web_today
):
    result = ask_on_workbench(monkeypatch, "archive")
    assert result["status"] == "partial" and result["stop_reason"] == "budget_exhausted"
    assert not result["claims"]
    diagnostics = result["diagnostics"]
    assert diagnostics["time_context"]["today"] == web_today.isoformat()
    chronology = diagnostics["answer_checks"][-1]["validation"]["event_chronology"]
    assert chronology["latest_held_date"] is None and not chronology["scheduled_leads"]
    texts = result["answer"]
    for notice in (EMPTY_NOTICE, SEARCH_DATE_NOTICE):
        assert any(notice in step["notices"] for step in diagnostics["steps"])
        assert notice not in texts
    leads = result["preliminary_evidence"]
    sources = {source["id"]: source for source in result["sources"]}
    assert leads and {item["source_id"] for item in leads} == set(sources)
    assert len({source["url"] for source in sources.values()}) == len(sources)
    for item in leads:
        sid, quote = item["source_id"], item["quote"]
        assert sources[sid]["date"] == ARCHIVE_DAY and sources[sid]["content_type"] == "extracted"
        assert quote in sources[sid]["text"] and quote in texts and f"[{sid}]" in texts
        assert item["evidence_id"].startswith(sid + ":")
    # The archive must be recognizable as historical even in the plain JSON answer.
    assert "2025" in texts
    assert {item["kind"] for item in leads} == {"held_report", "schedule_report"}
    assert any("5 октября" in item["quote"] for item in leads)
    assert not any("2026" in item["quote"] for item in leads)


@pytest.mark.parametrize("outcome", ["unrelated", "empty", "failure"])
def test_no_preliminary_evidence_for_unrelated_or_unavailable_materials(monkeypatch, outcome):
    result = ask_on_workbench(monkeypatch, outcome)
    assert result["status"] == ("unavailable" if outcome == "failure" else "partial")
    assert not result["preliminary_evidence"] and not result["claims"] and not result["sources"]
    assert "финансового комитета" not in result["answer"]
