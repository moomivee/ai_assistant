import pytest
from test_agent import REPORT, plan
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
