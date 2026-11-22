"""Reproduce the three undated leads and empty extractions in trace ce10910a."""

import httpx
import pytest
from test_agent import plan
from test_agent_loop import action
from test_opec_official_archive import INDEX, article, card

from src.agent import ask

QUESTION = "когда была последняя встреча опэк+?"
QUERY = "последняя встреча ОПЕК+"
RBC = {
    "url": "https://www.rbc.ru/business/22/05/2022/628a8f1c9a79478c64a7dae4",
    "title": "Саудовская Аравия выразила надежду на соглашение ОПЕК+ с ...",
    "content": "5 мая состоялась последняя встреча министров стран ОПЕК+, по результатам которой "
    "было заявлено, что до июня страны продолжат следовать правилу увеличения добычи",
}
TAG = {
    "url": "https://ru.euronews.com/tag/opec",
    "title": "ОПЕК - Euronews.com",
    "content": 'Россия благодарит ОАЭ за "посредничество в чувствительных вопросах". Встреча '
    "прошла после того, как в рамках ОПЕК+, в котором участвуют и Россия, и Эмираты",
}
BACKGROUND = {
    "url": "https://minenergo.gov.ru/activity/international-cooperation/cooperation/cooperation-with-opec",
    "title": "Сотрудничество с ОПЕК - Министерство энергетики РФ",
    "content": "Благодаря последовательным усилиям российской стороны 10 декабря 2016 г. в Вене "
    "состоялось подписание соглашения между странами ОПЕК и 11 странами, не входящими",
}
# This organizer publication and its date are synthetic counterfactuals.
MINUTES = {
    "url": "https://www.opec.org/pr-detail/fixture-minutes.html",
    "title": "OPEC and non-OPEC Ministerial Meeting",
    "content": "Совместная встреча министров стран ОПЕК+ состоялась 17 ноября 2026 года.",
    "published_date": "2026-11-18",
}
WIRE = {**MINUTES, "url": "https://www.reuters.com/fixture-ministerial-meeting"}


class RepeatsSearch:
    def __init__(self, early_finish):
        self.early_finish = early_finish

    def generate(self, name, payload, schema):
        if name == "plan":
            return plan(fresh=True, latest_completed_event=True, news=False, rag_query=QUERY)
        if name == "decide":
            if payload["observations"] and (
                self.early_finish
                or any(s["url"] in {MINUTES["url"], WIRE["url"]} for s in payload["sources"])
            ):
                return schema.model_validate(action("finish"))
            return schema.model_validate(action("web_search", QUERY))
        if name == "event_relevance":
            return schema.model_validate({"off_topic": []})
        if name == "event_chronology":
            rows = []
            for key, block in payload["evidence"].items():
                text = block["text"]
                events = []
                if text == MINUTES["content"]:
                    events = [
                        dict(
                            kind="held",
                            date="2026-11-17",
                            quote=text,
                            scope="ОПЕК+",
                            match="in_scope",
                        )
                    ]
                rows.append(
                    dict(
                        evidence_id=key,
                        status="uncertain" if text == RBC["content"] else "assessed",
                        events=events,
                    )
                )
            return schema.model_validate({"blocks": rows})
        if name == "cited_answer":
            ledger = payload["event_chronology"]
            assert ledger["latest_held_date"] == "2026-11-17"
            return schema.model_validate(
                dict(
                    claims=[
                        dict(text=MINUTES["content"], evidence_ids=ledger["latest_evidence_ids"])
                    ],
                    complete=True,
                )
            )
        assert name == "citation_review"
        return schema.model_validate(
            dict(supported=[0], coverage="sufficient", complete=True, feedback="")
        )


@pytest.mark.parametrize(
    "coverage",
    ["organizer-direct", "organizer-via-tavily", "independent-news", "only-undated-lead"],
)
@pytest.mark.parametrize("early_finish", [False, True], ids=["repeated-query", "premature-finish"])
def test_undated_mixed_publishers_and_empty_extracts(
    monkeypatch, web_today, coverage, early_finish
):
    monkeypatch.setenv("TAVILY_API_KEY", "fixture")
    monkeypatch.setenv("WEB_DOMAINS", "rbc.ru,euronews.com,minenergo.gov.ru,opec.org,reuters.com")
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    calls = []

    if coverage == "organizer-direct":
        pages = {
            INDEX: card(MINUTES["title"], "18 November 2026", "/pr-detail/fixture-minutes.html"),
            MINUTES["url"]: article(MINUTES["title"], MINUTES["content"]),
        }

        async def read(client, url, deadline):
            calls.append(("archive", {"url": url}))
            return pages.get(url, "").encode()

        monkeypatch.setattr("src.integrations.opec_publications._fetch_html", read)

    def post(url, *, json, **kwargs):
        op = url.rsplit("/", 1)[-1]
        calls.append((op, json))
        if op == "extract":
            data = {"results": [], "failed_results": [{"url": json["urls"][0]}]}
        else:
            assert op == "search"
            if json["include_domains"] == ["opec.org"]:
                data = {"results": [MINUTES] if coverage == "organizer-via-tavily" else []}
            elif (
                coverage == "independent-news"
                and any(req.get("include_domains") == ["opec.org"] for _, req in calls)
                and "reuters.com" in json["include_domains"]
                and (
                    json["query"] != calls[0][1]["query"]
                    or json["include_domains"] != calls[0][1]["include_domains"]
                )
            ):
                data = {"results": [WIRE]}
            else:
                data = {"results": [RBC, TAG, BACKGROUND]}
        return httpx.Response(200, request=httpx.Request("POST", url), json=data)

    monkeypatch.setattr("src.integrations.web.httpx.post", post)
    response = ask(QUESTION, client=RepeatsSearch(early_finish))
    assert response["diagnostics"]["time_context"]["today"] == web_today.isoformat()
    primary = next(
        (
            i
            for i, (op, req) in enumerate(calls)
            if op == "archive" or op == "search" and req["include_domains"] == ["opec.org"]
        ),
        None,
    )
    assert primary is not None, [(op, req.get("query"), req.get("urls")) for op, req in calls]
    assert sum(op == "extract" for op, _ in calls[:primary]) <= 1
    if coverage == "organizer-direct":
        assert [req["url"] for op, req in calls if op == "archive"] == [INDEX, MINUTES["url"]]
        assert not any(
            op == "search" and req["include_domains"] == ["opec.org"] for op, req in calls
        )
        assert response["sources"][0]["provider"] == "opec_press"
    else:
        assert calls[primary][1]["query"] != calls[0][1]["query"]
    assert "2016" not in response["answer"]
    assert "2022" not in response["answer"]
    if coverage != "only-undated-lead":
        assert MINUTES["content"] in response["answer"]
        expected = WIRE if coverage == "independent-news" else MINUTES
        assert {s["url"] for s in response["sources"]} == {expected["url"]}
        assert response["claims"]
    else:
        assert response["status"] == "partial"
        assert not response["claims"]
        assert {s["url"] for s in response["sources"]} == {RBC["url"]}
        assert [entry["source_id"] for entry in response["preliminary_evidence"]] == ["W1"]
        assert "5 мая" in response["answer"] and "год" in response["answer"].lower()
        assert "5 мая 2026" not in response["answer"]
