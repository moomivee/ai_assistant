"""Regression checks on what the four-question gate actually counts as success."""

import json
from pathlib import Path

from eval.core_smoke import score


def mixed_answer():
    fixture = json.loads((Path(__file__).parent / "fixtures/qwen-review-failure.json").read_text())
    claim = fixture["draft"]["claims"][0]
    source = fixture["sources"][0]
    quote = {
        "kind": "web",
        "id": "W1",
        "grade": "Brent",
        "date": "2026-11-13",
        "price": 85.6,
        "url": "https://fred.stlouisfed.org/series/DCOILBRENTEU",
    }
    return {
        "status": "ok",
        "answer": claim["text"] + " [R1]\n\nBrent 13 ноября 2026 — 85,60 USD/барр. [W1]",
        "claims": [claim],
        "sources": [source, quote],
        "diagnostics": {"time_context": {"today": "2026-11-17"}},
    }


def test_mixed_answer_requires_both_parts_even_if_sources_were_retrieved():
    response = mixed_answer()
    assert score("mixed", response)["passed"]
    response["answer"] = response["answer"].split("\n\n")[0]
    assert not score("mixed", response)["passed"]
    response = mixed_answer()
    response["answer"] = response["answer"].split("\n\n")[1]
    assert not score("mixed", response)["passed"]


def test_quote_with_future_date_or_partial_status_is_not_counted():
    response = mixed_answer()
    response["status"] = "partial"
    assert not score("quote", response)["passed"]
    response["status"] = "ok"
    response["diagnostics"]["time_context"]["today"] = "2026-11-12"
    assert not score("quote", response)["passed"]


def test_reversed_grade_values_do_not_pass():
    response = mixed_answer()
    for field in [response, response["claims"][0]]:
        key = "answer" if field is response else "text"
        field[key] = field[key].replace("30-32", "34-37")
    assert not score("mixed", response)["passed"]


def test_explicit_provider_error_is_distinguishable_from_a_tool_call_error():
    response = mixed_answer()
    response["diagnostics"]["steps"] = [{"error": "Ouroboros: TOOL_ERROR"}]
    assert not score("quote", response)["quote_provider_error"]
    response["diagnostics"]["steps"] = [
        {"notices": ["Не удалось получить котировку Brent из EIA/FRED: тайм-аут."]}
    ]
    assert score("quote", response)["quote_provider_error"]


def test_swapping_density_and_sulfur_units_does_not_pass():
    response = mixed_answer()
    for field in (response, response["claims"][0]):
        key = "answer" if field is response else "text"
        text = field[key].replace("API 30-32", "API 1.3-1.5").replace("1.3%-1.5%", "30%-32%")
        field[key] = text.replace("API 34-37", "API 0.4-0.6").replace("0.4%-0.6%", "34%-37%")
    assert not score("comparison", response)["passed"]


def test_stray_price_match_in_answer_cannot_mask_wrong_quoted_price():
    response = mixed_answer()
    response["answer"] = "Brent 13 ноября 2026 — 42,12 USD/барр., открытый интерес 85,60. [W1]"
    assert not score("quote", response)["passed"]
