from datetime import timedelta

import httpx

from src.calculations.forecast import forecast
from src.config import settings
from src.integrations.web import allowed, search_web


def test_web_domain_and_date_filters(monkeypatch, web_today):
    assert allowed("https://www.eia.gov/report", ("eia.gov",))
    assert not allowed("https://eia.gov.fake.example/report", ("eia.gov",))
    assert not allowed("https://user@eia.gov/report", ("eia.gov",))
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    settings.cache_clear()
    rows = [
        {
            "url": "https://eia.gov/" + str(i),
            "content": "Oil supplies",
            "title": "EIA",
            "published_date": d,
        }
        for i, d in enumerate(
            [
                web_today.isoformat(),
                (web_today + timedelta(days=1)).isoformat(),
                (web_today - timedelta(days=90)).isoformat(),
                None,
            ]
        )
    ]
    rows.append(
        {
            "url": "https://untrusted.example/oil",
            "content": "Oil",
            "published_date": web_today.isoformat(),
        }
    )

    def post(*a, **k):
        return httpx.Response(200, request=httpx.Request("POST", a[0]), json={"results": rows})

    monkeypatch.setattr("src.integrations.web.httpx.post", post)
    result = search_web("Oil", True, days=30)
    assert len(result) == 1 and result[0]["url"] == "https://eia.gov/0"


def test_forecast_scenario_shift():
    base = forecast("Brent", 3)
    scenario = forecast("Urals", 3, discount=10, cut=1, sensitivity=3)
    for a, b in zip(base["rows"], scenario["rows"]):
        assert abs(b["point"] - (a["point"] - 7)) < 0.02
        assert b["lower"] <= b["point"] <= b["upper"]
        assert b["mae"] == a["mae"]
