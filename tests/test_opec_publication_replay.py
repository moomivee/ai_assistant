"""Reproduce the 22 November OPEC+ trace through the graph and the Tavily HTTP adapter.

The question and the model's search window come from the trace. The page body
and event dates are synthetic fixtures, not a record of an actual OPEC+ meeting.
"""

import httpx
import pytest
from test_agent import plan
from test_agent_loop import action

from src.agent import ask

QUESTION = (
    "Когда страны ОПЕК+, добровольно корректирующие добычу, встречались в последний раз? "
    "А когда они планируют встретиться снова?"
)
QUERY = "страны ОПЕК+ последняя встреча и плановые встречи ноябрь 2026"
ARTICLE = "https://www.interfax.ru/business/fixture"
SNIPPET = "В число стран-добровольцев ОПЕК+ входят несколько государств."
HELD = "Страны ОПЕК+, добровольно корректирующие добычу, встретились 17 ноября 2026 года."
SCHEDULED = "Следующая встреча этих участников назначена на 2 декабря 2026 года."


class RepeatingModel:
    def generate(self, name, payload, schema):
        if name == "plan":
            return plan(fresh=True, latest_completed_event=True, rag_query="встречи ОПЕК+")
        if name == "decide":
            decision = action("web_search", QUERY)
            if not payload["observations"]:
                decision.update(
                    start_date="2026-11-01", end_date="2026-11-30", search_depth="advanced"
                )
            return schema.model_validate(decision)
        if name == "event_relevance":
            return schema.model_validate({"off_topic": []})
        if name == "event_chronology":
            blocks = []
            for key, row in payload["evidence"].items():
                events = []
                for quote, kind, day in (
                    (HELD, "held", "2026-11-17"),
                    (SCHEDULED, "scheduled", "2026-12-02"),
                ):
                    if quote in row["text"]:
                        events.append(
                            dict(
                                kind=kind,
                                date=day,
                                quote=quote,
                                scope="страны-добровольцы ОПЕК+",
                                match="in_scope",
                            )
                        )
                blocks.append(dict(evidence_id=key, status="assessed", events=events))
            return schema.model_validate(dict(blocks=blocks, requested_context=["next_scheduled"]))
        if name == "cited_answer":
            ids = payload["event_chronology"]["latest_evidence_ids"]
            return schema.model_validate(
                dict(claims=[dict(text=HELD, evidence_ids=ids)], complete=True)
            )
        assert name == "citation_review"
        return schema.model_validate(
            dict(
                supported=list(range(len(payload["claims"]))),
                coverage="sufficient",
                complete=True,
                feedback="",
            )
        )


@pytest.mark.parametrize("has_dates", [True, False], ids=["page-confirms-dates", "no-confirmation"])
def test_month_hypothesis_and_repeated_search_recover_an_undated_lead(
    monkeypatch, web_today, has_dates
):
    monkeypatch.setenv("AGENT_MAX_STEPS", "4")
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    monkeypatch.setenv("WEB_DOMAINS", "interfax.ru,opec.org")
    monkeypatch.setenv("TAVILY_API_KEY", "fixture")
    calls = []

    def post(url, *, json, **kwargs):
        calls.append((url.rsplit("/", 1)[-1], json))
        if url.endswith("/search"):
            items = [
                dict(url=ARTICLE, title="Министры стран ОПЕК+ начали встречу", content=SNIPPET),
                dict(
                    url=ARTICLE + "-future",
                    title="Архив",
                    content=HELD,
                    published_date="2026-11-23",
                ),
            ]
        else:
            assert url.endswith("/extract") and json["urls"] == [ARTICLE]
            items = [dict(url=ARTICLE, raw_content=f"{HELD} {SCHEDULED}" if has_dates else SNIPPET)]
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": items})

    monkeypatch.setattr("src.integrations.web.httpx.post", post)
    result = ask(QUESTION, client=RepeatingModel())
    assert result["diagnostics"]["time_context"]["today"] == web_today.isoformat()
    first = result["diagnostics"]["steps"][0]
    assert first["status"] == "ok" and first["result_count"] == 1
    assert first["argument_repair"]["before"]["end_date"] == "2026-11-30"
    operation, query = calls[0]
    assert operation == "search" and "ноябрь 2026" not in query["query"]
    assert "start_date" not in query and "end_date" not in query and "days" not in query
    assert calls[1][0] == "extract"
    assert sum(operation == "extract" for operation, _ in calls) == 1
    recovery = result["decisions"][1]
    assert recovery["model_action"] == "web_search" and recovery["action"] == "read_web"
    assert recovery["origin"] == "event_recovery"
    first_check = result["diagnostics"]["answer_checks"][0]["validation"]
    assert first_check["reason"] == "no_confirmed_event"
    assert not any(
        row["url"].endswith("-future") for row in result["diagnostics"]["retrieved_sources"]
    )

    if has_dates:
        assert result["status"] == "ok"
        assert HELD in result["answer"] and "02.12.2026" in result["answer"]
        assert {row["url"] for row in result["sources"]} == {ARTICLE}
        assert all(row["content_type"] == "extracted" for row in result["sources"])
    else:
        assert result["status"] == "partial" and not result["claims"]
        assert result["stop_reason"] in {"budget_exhausted", "no_progress"}
        assert HELD not in result["answer"] and "02.12.2026" not in result["answer"]
