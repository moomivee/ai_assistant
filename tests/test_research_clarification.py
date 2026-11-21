import pytest
from test_agent import REPORT, WEB, plan
from test_agent_loop import Scripted, action, event

from src.agent import ask
from src.config import settings
from src.validation import validate_plan

ORIGINAL = "Какой рейтинг у российской нефти?"
CLARIFICATION = "Какой показатель сравнить: место России по добыче или по экспорту?"
OPTIONS = ["Место по добыче", "Место по экспорту"]


class ResearchClient(Scripted):
    def __init__(self, planned, actions=()):
        super().__init__(actions)
        self.planned = planned

    def generate(self, name, payload, schema):
        if name == "plan":
            self.payloads.append((name, payload))
            return self.planned
        return super().generate(name, payload, schema)


def test_ambiguous_ranking_prompts_for_a_measure_before_searching():
    client = ResearchClient(plan(clarify=CLARIFICATION, research_options=OPTIONS))
    result = ask(ORIGINAL, client=client)
    assert result["status"] == "clarification"
    assert result["answer"] == CLARIFICATION
    assert result["trace"] == [] and result["sources"] == []
    assert [name for name, _ in client.payloads] == ["plan"]


@pytest.mark.parametrize(
    "question, intent, clarification, options",
    [
        ("Чем отличаются Urals и ESPO по плотности и сере?", "research", "", []),
        ("Какое место у РФ по объёму экспорта в 2025 году?", "research", "", []),
        (ORIGINAL, "research", ORIGINAL, OPTIONS),
        (ORIGINAL, "research", "Уточните запрос", []),
        (ORIGINAL, "research", CLARIFICATION, ["Добыча", "добыча."]),
        (ORIGINAL, "research", CLARIFICATION, [" ", "Добыча"]),
        ("Как получить работу в нефтяной компании?", "outside", CLARIFICATION, OPTIONS),
    ],
)
def test_clarification_needs_a_research_goal_and_distinct_meanings(
    question, intent, clarification, options
):
    parsed = validate_plan(
        plan(intent=intent, clarify=clarification, research_options=options), question
    )
    assert parsed.clarify == "" and parsed.research_options == []


def test_choice_after_clarification_reaches_planning_decision_answer_and_review(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()
    previous = ORIGINAL + "\nУточнение агента: " + CLARIFICATION
    question = "Второй вариант, за 2020 год."
    query = "Russia 2020 crude oil export ranking"
    client = ResearchClient(plan(rag_query=query), [action("report_search", query)])
    source = {
        **REPORT,
        "text": "Условный отчёт: в 2020 году Россия заняла второе место по экспорту.",
    }

    def execute(_registry, name, **args):
        assert name == "report_search" and args["query"] == query
        return [source], None, event(name)

    result = ask(question, previous, client=client, execute=execute)
    assert result["status"] == "ok" and result["sources"][0]["id"] == "R1"
    for name in ("plan", "decide", "cited_answer", "citation_review"):
        payload = next(payload for step, payload in client.payloads if step == name)
        assert payload["question"] == question
        assert payload["previous"] == previous


@pytest.mark.parametrize("is_followup", [False, True])
def test_review_can_request_a_metric_without_retrying_irrelevant_search(is_followup, monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()
    original = "Какой рейтинг у российской нефти на международной арене?"
    previous = original + "\nУточнение агента: " + CLARIFICATION if is_followup else None
    question = "По объёму экспорта за 2025 год." if is_followup else original
    query = "рейтинг российской нефти международные рынки"
    sources = [
        {**WEB, "url": f"https://www.eia.gov/reports/{i}", "text": text}
        for i, text in enumerate(
            (
                "Опубликован прогноз экспортной цены.",
                "В отчёте оценили мировое предложение нефти.",
                "Спрос на нефть снизился.",
                "Изменение предложения по регионам.",
                "Оценка доходов от экспорта.",
                "Опубликован обзор рынка.",
            )
        )
    ]

    class RankingReview(ResearchClient):
        def generate(self, name, payload, schema):
            if name == "cited_answer":
                self.payloads.append((name, payload))
                return schema.model_validate(
                    {
                        "claims": [
                            {"evidence_ids": [key], "text": payload["evidence"][key]}
                            for key in ("W1:B1", "W2:B1")
                        ],
                        "complete": False,
                    }
                )
            if name == "citation_review":
                self.payloads.append((name, payload))
                return schema.model_validate(
                    {
                        "supported": [],
                        "complete": False,
                        "coverage": "unverified",
                        "missing_aspects": ["Нет подтверждённого места России в рейтинге."],
                        "feedback": "",
                        "research_options": OPTIONS,
                    }
                )
            return super().generate(name, payload, schema)

    client = RankingReview(plan(rag_query=query), [action("web_search", query), action("finish")])

    def search(_registry, name, **kwargs):
        assert name == "web_search" and kwargs["query"] == query
        return {"items": sources, "notices": []}, None, event(name)

    response = ask(question, previous, client=client, execute=search)
    assert response["claims"] == response["sources"] == []
    assert len(response["trace"]) == 1
    diagnostics = response["diagnostics"]
    assert diagnostics["plan"]["clarify"] == ""
    assert len(diagnostics["retrieved_sources"]) == 6
    (check,) = diagnostics["answer_checks"]
    assert check["validation"]["semantic_rejected"] == 2
    stages = [name for name, _ in client.payloads]
    if is_followup:
        assert response["status"] == "partial" and not response["clarification"]
        assert "research_clarification" not in check["validation"]
        assert stages[-1] == "decide"
    else:
        assert response["status"] == "clarification" and response["clarification"]
        assert response["stop_reason"] == "research_clarification"
        assert all(option in response["answer"] for option in OPTIONS)
        assert response["answer"] == check["validation"]["research_clarification"]["clarify"]
        assert stages == ["plan", "decide", "cited_answer", "citation_review"]
