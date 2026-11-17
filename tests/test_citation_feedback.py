"""A failed reference and a lack of evidence require different user feedback."""

import pytest
from test_agent import WEB, plan
from test_agent_loop import Scripted, action, event

from src.agent import ask
from src.integrations.web import SEARCH_DATE_NOTICE
from src.models.schemas import CitedAnswer


@pytest.mark.parametrize(
    "reason,candidates,bad_ids,repair,shows_reference_failure",
    [
        ("local_validation_rejected", 2, 2, False, True),
        ("local_validation_rejected", 2, 2, True, True),
        ("local_validation_rejected", 2, 1, False, False),
        ("no_candidate_claims", 0, 0, True, False),
    ],
)
def test_partial_answer_uses_final_citation_outcome_and_preserves_tool_notices(
    monkeypatch, reason, candidates, bad_ids, repair, shows_reference_failure
):
    technical_notice = "Источник котировки временно недоступен."

    def answer(*args, diagnostics, **kwargs):
        diagnostics.update(
            reason=reason,
            candidate_claims=candidates,
            citation_rejected=bad_ids,
            numeric_rejected=candidates - bad_ids,
            accepted_claims=0,
            citation_repair_attempted=repair,
        )
        return CitedAnswer(claims=[], complete=False), {}, "retry"

    monkeypatch.setattr("src.agent.generate_answer", answer)
    data = {"items": [{**WEB, "date": None}], "notices": [SEARCH_DATE_NOTICE, technical_notice]}
    do_search = action("web_search", "traded crude grades")
    result = ask(
        "Какими сортами нефти торгует страна?",
        client=Scripted([do_search] * 3),
        planner=lambda *_: plan(fresh=True),
        execute=lambda registry, name, **kwargs: (data, None, event(name)),
    )
    assert result["status"] == "partial" and result["stop_reason"] == "no_progress"
    assert not result["claims"] and not result["sources"]
    message = result["answer"]
    assert ("ссылки модели не совпали" in message) == shows_reference_failure
    assert ("Подтверждённых данных для ответа недостаточно" in message) != shows_reference_failure
    assert (SEARCH_DATE_NOTICE in message) != shows_reference_failure
    assert technical_notice in message
    assert SEARCH_DATE_NOTICE in result["diagnostics"]["steps"][0]["notices"]
