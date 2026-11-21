"""Browser checks with explicit UI fixtures, not a live LLM quality evaluation.

uv run --with playwright python tools/check_trace_ui.py
Install Chromium with `playwright install chromium`, or set CHROMIUM_PATH.
"""

import argparse
import asyncio
import base64
import copy
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
SEED_REPORTS = json.loads((ROOT / "data/reports.json").read_text())
QUESTION = "Тест интерфейса: сравни образцы Orion и Vega по тестовому отчёту."
TEXT = "Тестовые данные: Orion — 42° API; Vega — 28° API. Чем выше API, тем легче нефть."
SOURCE = dict(
    id="R1",
    kind="report",
    title="Тестовый отчёт · образцы нефти",
    date="2026-11-11",
    page=1,
    url="https://example.invalid/fixture",
    text=TEXT,
    used_in_answer=True,
)
DECISIONS = [
    dict(
        origin="plan_policy",
        action="report_search",
        purpose="Найти характеристики двух образцов в отчёте.",
        query="Orion Vega properties",
        missing_facts=["Значения API для обоих образцов"],
    ),
    dict(
        action="report_search",
        purpose="Проверить поиск после пустого результата.",
        query="Orion Vega properties",
    ),
    dict(
        action="report_search",
        purpose="Уточнить запрос: искать показатели API.",
        query="Orion Vega API",
    ),
    dict(
        action="finish",
        purpose="В найденном фрагменте есть значения для сравнения.",
        source_ids=["R1"],
        evidence_checks=[dict(fact="Найдены значения API для обоих образцов", source_ids=["R1"])],
        missing_facts=[],
        assessment_references_valid=True,
    ),
]
STEPS = [
    dict(
        step=1,
        tool="report_search",
        purpose=DECISIONS[0]["purpose"],
        arguments={"query": "Orion Vega properties"},
        status="ok",
        ms=210,
        result_count=0,
        source_ids=[],
        new_evidence=False,
        no_progress_steps=0,
    ),
    dict(
        step=2,
        tool="report_search",
        purpose=DECISIONS[1]["purpose"],
        arguments={"query": "Orion Vega properties"},
        status="rejected",
        ms=None,
        error="Duplicate action rejected. Change the query.",
        duplicate_of=1,
        reused_source_ids=[],
        new_evidence=False,
        no_progress_steps=1,
    ),
    dict(
        step=3,
        tool="report_search",
        purpose=DECISIONS[2]["purpose"],
        arguments={"query": "Orion Vega API"},
        status="ok",
        ms=180,
        result_count=1,
        source_ids=["R1"],
        new_evidence=True,
        no_progress_steps=0,
    ),
]
RESPONSE = dict(
    question=QUESTION,
    answer="По тестовым данным, у Orion 42° API, у Vega — 28° API. Orion легче, поскольку более высокое значение API соответствует меньшей плотности. [R1]\n\nЭто контролируемый пример для проверки интерфейса, а не новый ответ Qwen.",
    status="ok",
    clarification=False,
    intent="research",
    mode="UI fixture · без LLM",
    sources=[SOURCE],
    claims=[],
    forecast=None,
    trace=[
        dict(tool="report_search", engine="ui.fixture", status="ok", ms=210),
        dict(tool="report_search", engine="ui.fixture", status="ok", ms=180),
    ],
    trace_id="ui-fixture-not-a-live-llm-run",
    agent_mode="loop",
    stop_reason="completed",
    rejected_actions=1,
    decisions=DECISIONS,
    diagnostics=dict(
        time_context=dict(
            today="2026-11-17",
            current_year=2026,
            current_time="2026-11-17T23:14:00+03:00",
            timezone="Europe/Moscow",
        ),
        plan=dict(
            intent="research",
            rag_query="Orion Vega",
            wants_context=True,
            report_only=True,
            fresh=False,
            quotes=[],
        ),
        steps=STEPS,
        retrieved_sources=[SOURCE],
        answer_feedback="",
        failure="",
        no_progress_limit=2,
        no_progress_steps=0,
        timeout_seconds=180,
        answer_reserve_seconds=60,
        answer_checks=[
            dict(
                after_step=3,
                source_ids=["R1"],
                status="ok",
                claims=1,
                complete=True,
                ms=300,
                feedback="",
                validation=dict(
                    candidate_claims=1,
                    citation_rejected=0,
                    numeric_rejected=0,
                    semantic_rejected=0,
                    accepted_claims=1,
                    reason="complete",
                ),
            )
        ],
    ),
)
LEGACY = next(
    r["response"]
    for r in json.loads((ROOT / "eval/results/agent-audit-7b-dev.json").read_text())["rows"]
    if r["response"].get("forecast")
)


FINISH_STREAM = threading.Event()


def stream_event(kind, **fields):
    return (json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n").encode()


class Handler(BaseHTTPRequestHandler):
    def send(self, body, mime, status=200, *, cache_control=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/fixture/previous-workbench":
            return self.send(b'<link rel="stylesheet" href="/static/site.css">', "text/html")
        if self.path == "/static/site.css":
            # Reproduce a browser that still has a fresh copy from the previous release.
            return self.send(
                b":root{--previous-workbench:yes}.workspace-grid{display:block}",
                "text/css",
                cache_control="public, max-age=86400",
            )
        if self.path == "/api/status":
            data = dict(
                model="GigaChat-2",
                provider="gigachat",
                llm_configured=True,
                tavily=False,
                agent_mode="loop",
                agent_max_steps=6,
                reports=SEED_REPORTS,
                history=[
                    dict(month="2025-01", Brent=70, WTI=65),
                    dict(month="2025-02", Brent=72, WTI=67),
                ],
            )
            return self.send(json.dumps(data).encode(), "application/json")
        if self.path.partition("?")[0] == "/api/knowledge/documents":
            documents = [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "filename": row["file"],
                    "status": "ready",
                    "builtin": True,
                    "active_version_id": "fixture-v1",
                    "current_version_id": "fixture-v1",
                }
                for row in SEED_REPORTS
            ]
            return self.send(json.dumps({"documents": documents}).encode(), "application/json")
        if self.path.endswith("/update-capability"):
            return self.send(
                b'{"supported":true,"discovers_new_releases":false}', "application/json"
            )
        if self.path == "/api/rag/status":
            return self.send(b'{"model":"minilm-multi","top_k":6}', "application/json")
        if path == "/static/gigachat-mark.png":
            return self.send((ROOT / "src/static/gigachat-mark.png").read_bytes(), "image/png")
        files = {
            "/": ("index.html", "text/html"),
            "/static/site.js": ("site.js", "application/javascript"),
            "/static/trace.js": ("trace.js", "application/javascript"),
            "/static/journey.js": ("journey.js", "application/javascript"),
            "/static/knowledge.js": ("knowledge.js", "application/javascript"),
            "/static/site.css": ("site.css", "text/css"),
        }
        if path in files:
            name, mime = files[path]
            return self.send((ROOT / "src/static" / name).read_bytes(), mime + "; charset=utf-8")
        self.send(b"Not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/ask/stream":
            return self.send(b"{}", "application/json", 404)
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        question = payload["question"]
        response = copy.deepcopy(RESPONSE)
        if question == "legacy":
            response = LEGACY
        elif question == "web-only":
            web_source = dict(
                id="W4",
                kind="web",
                title="Тестовая веб-страница о сортах нефти",
                date=None,
                url="https://example.invalid/web-fixture",
                text="В тестовом источнике названы образцы Orion и Vega.",
                content_type="snippet",
                provider="tavily",
                cache_hit=True,
                used_in_answer=True,
            )
            decisions = [
                dict(action="web_search", purpose="Пустая попытка не выполнена.", query=""),
                dict(
                    action="web_search",
                    purpose="Проверить перечень в интернете.",
                    query="oil grades",
                ),
            ]
            response.update(
                answer="В тестовом веб-источнике названы Orion и Vega. [W4]",
                status="partial",
                sources=[web_source],
                decisions=decisions,
                trace=[dict(tool="web_search", status="ok", ms=200)],
            )
            response["diagnostics"].update(
                plan=dict(intent="research", report_only=False, wants_context=True, fresh=False),
                steps=[
                    dict(step=1, tool="web_search", status="rejected", error="Empty query"),
                    dict(step=2, tool="web_search", status="ok", ms=200, source_ids=["W4"]),
                ],
                retrieved_sources=[web_source],
                semantic_review_enabled=False,
                answer_checks=[
                    dict(
                        after_step=2,
                        source_ids=["W4"],
                        status="ok",
                        complete=False,
                        claims=1,
                        ms=10,
                        validation=dict(
                            coverage="unverified",
                            missing_aspects=["Полнота перечня не подтверждена"],
                        ),
                    )
                ],
            )
        elif question in {"latest-event", "latest-event-unconfirmed"}:
            confirmed = question == "latest-event"
            source = dict(
                id="W2",
                kind="web",
                title="Сообщение о заседании",
                date=None,
                url="https://example.invalid/meeting-fixture",
                used_in_answer=confirmed,
                text="Заседание прошло 1 ноября 2026 года."
                if confirmed
                else "Следующее заседание запланировано на 29 ноября 2026 года.",
            )
            steps = []
            retrieved = []
            if confirmed:
                for source_id, title, excerpt in (
                    ("W1", "Календарь заседаний", "Заседание запланировано на 1 ноября."),
                    ("W3", "Встреча банкиров", "Обсудили регулирование банков."),
                ):
                    retrieved.append(
                        dict(
                            source,
                            id=source_id,
                            title=title,
                            text=excerpt,
                            used_in_answer=False,
                            url=f"https://news.invalid/{source_id}",
                        )
                    )
                steps.append(
                    dict(step=1, tool="web_search", status="ok", ms=18, source_ids=["W1", "W3"])
                )
            retrieved.append(source)
            steps.append(
                dict(
                    step=len(steps) + 1,
                    tool="web_search",
                    status="ok",
                    ms=24,
                    source_ids=["W2"],
                    arguments={"publisher_domain": "example.invalid"} if confirmed else {},
                )
            )
            response.update(
                answer=source["text"] + " [W2]" if confirmed else "Найден только анонс.",
                sources=[source] if confirmed else [],
                status="ok" if confirmed else "partial",
                stop_reason="completed" if confirmed else "no_progress",
                decisions=[
                    dict(action="web_search", purpose="Найти итоги заседания.") for _ in steps
                ],
                trace=[dict(tool=s["tool"], status=s["status"], ms=s["ms"]) for s in steps],
                rejected_actions=0,
            )
            response["diagnostics"].update(
                plan=dict(intent="research", fresh=True, latest_completed_event=True),
                steps=steps,
                retrieved_sources=retrieved,
                answer_checks=[
                    dict(
                        after_step=len(steps),
                        source_ids=["W1", "W2"] if confirmed else ["W2"],
                        status="ok",
                        complete=confirmed,
                        claims=int(confirmed),
                        validation=dict(
                            event_relevance=dict(excluded_source_ids=["W3"] if confirmed else []),
                            event_chronology=dict(
                                latest_held_date="2026-11-01" if confirmed else None,
                                latest_evidence_ids=["W2:B1"] if confirmed else [],
                                extraction=dict(reused_evidence_ids=["W1:B1"] if confirmed else []),
                            ),
                        ),
                    ),
                ],
            )
        elif question in {"two-checks", "answer-refusal"}:
            diagnostics = response["diagnostics"]
            source = dict(SOURCE, evidence={"R1:B1": TEXT})
            response["sources"] = [source]
            response["claims"] = [
                dict(source="R1", text="Orion легче Vega.", quote=TEXT, evidence_ids=["R1:B1"])
            ]
            diagnostics["retrieved_sources"] = [source]
            diagnostics["answer_checks"].insert(
                0,
                dict(
                    after_step=1,
                    source_ids=[],
                    status="ok",
                    complete=False,
                    claims=0,
                    validation={"reason": "no_evidence"},
                ),
            )
            final_check = diagnostics["answer_checks"][-1]
            final_check.update(
                supplied_source_ids=["R1"],
                evidence_ids=["R1:B1"],
                claim_results=[
                    dict(
                        index=1,
                        evidence_ids=["R1:B1"],
                        citation="passed",
                        numeric="passed",
                        semantic="passed",
                        accepted=True,
                    ),
                    dict(
                        index=2,
                        evidence_ids=["R1:B1"],
                        citation="passed",
                        numeric="failed",
                        semantic="not_checked",
                        missing_number_count=1,
                        accepted=False,
                    ),
                ],
            )
            final_check["validation"].update(candidate_claims=2, numeric_rejected=1)
            if question == "answer-refusal":
                reason = "GigaChat отказался обрабатывать этот запрос."
                response.update(
                    answer=reason,
                    status="unavailable",
                    stop_reason="llm_unavailable",
                    claims=[],
                    sources=[],
                )
                diagnostics["failure"] = reason
                diagnostics["retrieved_sources"] = [dict(source, used_in_answer=False)]
                diagnostics["answer_checks"][-1] = dict(
                    after_step=3, source_ids=["R1"], status="error", error=reason, claim_results=[]
                )
        elif question == "failed-only":
            response.update(
                status="unavailable",
                answer="Поиск недоступен.",
                sources=[],
                decisions=[dict(action="report_search", purpose="Найти факт в отчёте.")],
                trace=[dict(tool="report_search", status="error", ms=20)],
            )
            response["diagnostics"].update(
                steps=[dict(step=1, tool="report_search", status="error", error="Unavailable")],
                retrieved_sources=[],
                answer_checks=[],
            )
        elif question == "Что ты можешь?":
            response.update(
                question=question,
                intent="capabilities",
                status="ok",
                answer="Могу найти данные в отчёте и рассчитать прогноз цены Brent.",
                sources=[],
                claims=[],
                trace=[],
                decisions=[],
            )
            response["diagnostics"].update(
                plan={"intent": "capabilities"},
                steps=[],
                retrieved_sources=[],
                answer_checks=[],
            )
        elif question in {"clarification", "unavailable", "outside"}:
            response.update(
                status=question,
                clarification=question == "clarification",
                answer={
                    "clarification": "Укажите дисконт к Brent в USD за баррель.",
                    "unavailable": "Модель недоступна.",
                    "outside": "Вопрос вне специализации.",
                }[question],
                sources=[],
                decisions=[],
                trace=[],
            )
            response["diagnostics"].update(
                plan=None,
                steps=[],
                retrieved_sources=[],
                failure="Модель недоступна" if question == "unavailable" else "",
            )
        elif question == "partial":
            response.update(
                status="partial",
                stop_reason="budget_exhausted",
                answer="Данных недостаточно. Попытки поиска завершены.",
            )
        elif question in {"no_progress", "time_budget_exhausted", "research_budget_exhausted"}:
            response.update(status="partial", stop_reason=question)
        elif question == "controller":
            response["decisions"][-1].update(
                model_action="report_search", controller_note="Referenced assessment says enough."
            )
        elif question == "unknown-reference":
            response["decisions"][-1].update(
                unknown_source_ids=["R999"],
                evidence_checks=[dict(fact="Непроверенная ссылка", source_ids=["R999"])],
                assessment_references_valid=False,
            )
        elif question == "http-error":
            return self.send(b"{}", "application/json", 503)
        elif question == "hostile":
            text = '<img src=x onerror="window.injected=true">'
            response["answer"] = text
            response["decisions"][0]["purpose"] = text
            response["decisions"][0]["missing_facts"] = [text]
            response["decisions"][0]["evidence_checks"] = [dict(fact=text, source_ids=[text])]
            response["sources"][0].update(
                title=text, url="javascript:window.injected=true", text=text
            )
            response["diagnostics"]["retrieved_sources"] = response["sources"]
            response["diagnostics"]["steps"][0].update(
                purpose=text, arguments={"query": text}, notices=[text]
            )
            response["diagnostics"]["answer_checks"][0]["feedback"] = text
        response["question"] = question
        final = stream_event("result", result=response)
        if question not in {"quote-fallback", "direct-quote", "incomplete-stream"}:
            return self.send(final, "application/x-ndjson")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if question != "direct-quote":
            self.wfile.write(stream_event("progress", stage="quote_fallback"))
            self.wfile.flush()
        if question != "incomplete-stream":
            FINISH_STREAM.wait(timeout=10)
            self.wfile.write(final)

    def log_message(self, *_):
        pass


async def check_quote_progress(page):
    status = page.locator("#ask-status")
    for question, pending in [
        ("quote-fallback", "Проверяю другие источники"),
        ("direct-quote", "Агент готовит ответ"),
    ]:
        FINISH_STREAM.clear()
        try:
            await page.locator("#question").fill(question)
            await page.locator("#question").press("Control+Enter")
            await status.filter(has_text=pending).wait_for(timeout=5000)
            assert await page.locator("button.submit").is_disabled()
            assert await page.locator(".answer-panel").count() == 0
        finally:
            FINISH_STREAM.set()
        await page.locator(".answer-panel").wait_for()
        assert await status.is_hidden()
    await page.locator("#question").fill("incomplete-stream")
    await page.locator("#question").press("Control+Enter")
    await page.locator("#result").filter(has_text="Соединение закрылось").wait_for()
    assert await status.is_hidden()
    assert not await page.locator("button.submit").is_disabled()


async def check_dialogue_continuations(page):
    requests = []
    token = "fixture-only-secret"
    ranking = "Какой рейтинг у российской нефти?"
    clarification = "Какое место России сравнить — по добыче или по экспорту?"
    choice = "По экспорту за 2025 год."

    async def reply(route):
        payload = route.request.post_data_json
        requests.append(payload)
        question = payload["question"]
        if question in {"busy", "stale"}:
            code = "dialogue_busy" if question == "busy" else "dialogue_conflict"
            if question == "stale":
                return await route.fulfill(status=409, json={"detail": {"code": code}})
            return await route.fulfill(
                content_type="application/x-ndjson",
                body=stream_event("error", status=409, detail={"code": code}),
            )
        pending = question in {"Рассчитай Urals", "На три месяца"}
        response = dict(
            status="clarification" if pending else "unavailable" if question == "retry" else "ok",
            clarification=pending,
            intent="forecast",
            question=question,
            answer="Уточните параметры." if pending else "Результат",
        )
        if question == ranking:
            response.update(
                status="clarification",
                clarification=True,
                intent="research",
                answer=clarification,
                stop_reason="research_clarification",
                trace=[{"tool": "web_search", "status": "ok", "ms": 90}],
            )
        elif question == choice:
            response.update(intent="research", answer="Тестовый ответ про экспорт.")
        elif payload["remember_clarification"] and (pending or question == "retry"):
            response["dialogue_token"] = (
                token + str(len(requests)) if pending else payload["dialogue_token"]
            )
        await route.fulfill(
            content_type="application/x-ndjson", body=stream_event("result", result=response)
        )

    async def send(question):
        await page.locator("#question").fill(question)
        await page.locator("#question").press("Control+Enter")
        await page.locator('#result[aria-busy="false"]').wait_for()
        return requests[-1]

    await page.route("**/api/ask/stream", reply)
    first = await send("Рассчитай Urals")
    assert first["remember_clarification"] and "dialogue_token" not in first
    second = await send("На три месяца")
    assert second["dialogue_token"] == token + "1" and "previous" not in second
    assert token not in await page.locator("#result").text_content()
    for question in ("busy", "retry", "12 долларов за баррель"):
        assert (await send(question))["dialogue_token"] == token + "2"
        if question == "busy":
            assert "ещё обрабатывается" in await page.locator("#result").inner_text()
    assert await page.locator("#newquestion").is_hidden()
    fresh = await send("Рассчитай Urals")
    assert "dialogue_token" not in fresh and not fresh.get("previous")
    await send("stale")
    assert "устарело" in await page.locator("#result").inner_text()
    assert await page.locator("#newquestion").is_hidden()
    await send("Рассчитай Urals")
    await page.locator("#newquestion").click()
    assert "dialogue_token" not in await send("Рассчитай Urals")
    await page.locator("#query-settings > summary").click()
    await page.locator("#report-only").check()
    assert await page.locator("#newquestion").is_hidden()
    scoped = await send("Рассчитай Urals")
    assert scoped["report_only"] and "dialogue_token" not in scoped
    await page.locator("#remember-forecast").uncheck()
    await send("Рассчитай Urals")
    legacy = await send("На три месяца")
    assert not legacy["remember_clarification"] and legacy["previous"] == "Рассчитай Urals"
    assert "dialogue_token" not in legacy
    await page.locator("#newquestion").click()
    await send(ranking)
    await page.locator("details.trace-panel > summary").click()
    assert await page.locator(".response-status").inner_text() == "Нужно уточнение"
    assert "После проверки источников" in await page.locator(".trace-stop").inner_text()
    assert (
        await page.locator("#question").get_attribute("placeholder")
        == "Ответьте на уточняющий вопрос…"
    )
    continuation = await send(choice)
    assert continuation["previous"] == ranking + "\nУточнение агента: " + clarification
    assert "dialogue_token" not in continuation and await page.locator("#newquestion").is_hidden()
    await page.unroute("**/api/ask/stream", reply)


async def check_cached_workbench(page, url, *, capture=False):
    await page.set_viewport_size({"width": 1440, "height": 900})
    await page.goto(url + "/fixture/previous-workbench")
    assert await page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--previous-workbench').trim() === 'yes'"
    )
    resources = []

    def record_request(request):
        resources.append(request.url)

    page.on("request", record_request)
    await page.goto(url)
    assert not await page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--previous-workbench')"
    )
    await page.locator('#model-badge[data-provider="gigachat"]').wait_for()
    await page.locator("#mode").filter(has_text="GigaChat").wait_for()
    mark_width = await page.locator("#model-badge .model-mark").evaluate(
        r"""async mark => {
            const image = new Image();
            const background = getComputedStyle(mark).backgroundImage;
            const match = /^url\(["']?(.+?)["']?\)$/.exec(background);
            if (!match) throw new Error("Missing model badge image: " + background);
            image.src = match[1];
            await image.decode();
            return image.naturalWidth;
        }"""
    )
    assert mark_width > 0
    assert await page.locator(".workspace-grid").evaluate(
        """grid => getComputedStyle(grid).display === 'grid' &&
            grid.querySelector('form').getBoundingClientRect().right <=
            grid.querySelector('aside').getBoundingClientRect().left"""
    )
    assert await page.locator("#model-badge").evaluate(
        "badge => parseFloat(getComputedStyle(badge).columnGap) > 0"
    )
    page.remove_listener("request", record_request)
    expected = {
        "/static/site.css",
        "/static/site.js",
        "/static/trace.js",
        "/static/journey.js",
        "/static/knowledge.js",
        "/static/gigachat-mark.png",
    }
    assets = [urlsplit(resource) for resource in resources if urlsplit(resource).path in expected]
    assert {asset.path for asset in assets} == expected, resources
    assert all(parse_qs(asset.query).get("v") for asset in assets), resources
    await page.wait_for_function(
        "count => document.querySelector('#knowledge-count').textContent === String(count)",
        arg=len(SEED_REPORTS),
    )
    capabilities = page.locator("#panel-test .capability-query")
    assert await capabilities.count() == 6
    assert all([await button.is_visible() for button in await capabilities.all()])
    if capture:
        await page.screenshot(path=str(OUT / "trace-workbench-home-desktop.jpg"), quality=85)
    await page.set_viewport_size({"width": 390, "height": 844})
    assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert await page.locator(".workspace-grid").evaluate(
        """grid => grid.querySelector('form').getBoundingClientRect().bottom <=
            grid.querySelector('aside').getBoundingClientRect().top"""
    )
    if capture:
        await page.screenshot(path=str(OUT / "trace-workbench-home-mobile.jpg"), quality=85)


async def expand_detail(details):
    """Exercise native disclosure controls through the keyboard."""
    if not await details.evaluate("element => element.open"):
        await details.locator(":scope > summary").press("Enter")
    assert await details.evaluate("element => element.open")


async def check_journey(page, submit):
    journey = page.locator("details.request-journey")
    stages = journey.locator("ol.journey-timeline > li.journey-stage")

    def stage(kind, attrs=""):
        return journey.locator(f'.journey-timeline > li[data-kind="{kind}"]{attrs}')

    async def open_stage(kind, attrs=""):
        entry = stage(kind, attrs)
        await expand_detail(entry.locator("details.journey-node").first)
        return entry

    await submit("two-checks", open_technical=False)
    assert await journey.count() == 1
    assert not await journey.evaluate("element => element.open")
    await expand_detail(journey)
    assert await journey.locator(".journey-summary").evaluate("el => el === document.activeElement")
    kinds = await stages.evaluate_all(
        "rows => rows.map(r => [r.dataset.kind, r.dataset.step, r.dataset.afterStep])"
    )
    assert kinds[0][0] == "plan" and kinds[-1][0] == "finish", kinds
    for anchor in ["1", "3"]:
        index = kinds.index(["action", anchor, None])
        assert kinds[index + 1] == ["check", None, anchor], kinds
    await open_stage("plan")
    assert "Только загруженные отчёты" in await stage("plan").inner_text()
    rejected = await open_stage("action", '[data-step="2"]')
    assert "повтор" in (await rejected.inner_text()).lower()
    assert "отклон" in (await rejected.inner_text()).lower()
    retrieved = await open_stage("action", '[data-step="3"]')
    evidence = retrieved.locator('details.journey-source[data-source-id="R1"]')
    await expand_detail(evidence)
    assert TEXT in await evidence.inner_text()
    verified = await open_stage("check", '[data-after-step="3"]')
    claims = verified.locator("details.journey-claim")
    assert await claims.count() == 2
    assert "1 · принято" in await claims.first.inner_text()
    assert "2 · не принято" in await claims.last.inner_text()
    await expand_detail(claims.last)
    rejected_numbers = claims.last.locator("dt").filter(has_text="Проверка чисел")
    numeric_verdict = rejected_numbers.locator("xpath=following-sibling::dd[1]")
    assert await numeric_verdict.inner_text() == "Отклонено"
    assert "R1:B1" in await claims.last.inner_text()
    first_check = await open_stage("check", '[data-after-step="1"]')
    assert await first_check.locator("details.journey-claim").count() == 0
    finish = await open_stage("finish")
    answer_claim = finish.locator("details.journey-claim")
    await expand_detail(answer_claim)
    citation = answer_claim.locator('details.journey-source[data-source-id="R1"]')
    await expand_detail(citation)
    assert TEXT in await citation.inner_text()
    assert not await page.locator("details.trace-panel").evaluate("element => element.open")

    await page.set_viewport_size({"width": 390, "height": 844})
    assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    await journey.locator(":scope > summary").press("Space")
    assert not await journey.evaluate("element => element.open")
    await expand_detail(journey)
    assert await evidence.evaluate("element => element.open")

    await submit("answer-refusal", open_technical=False)
    await expand_detail(journey)
    succeeded = stage("action", '[data-step="3"]')
    assert await succeeded.locator('summary.journey-node-heading [data-tone="error"]').count() == 0
    failed = await open_stage("check", '[data-after-step="3"]')
    assert "GigaChat отказался" in await failed.inner_text()
    assert await failed.locator('summary.journey-node-heading [data-tone="error"]').count() == 1
    finish_badge = stage("finish").locator("summary.journey-node-heading .journey-badge")
    assert await finish_badge.get_attribute("data-tone") == "error"

    await submit("web-only", open_technical=False)
    await expand_detail(journey)
    web = await open_stage("action", '[data-step="2"]')
    snippet = web.locator('details.journey-source[data-source-id="W4"]')
    await expand_detail(snippet)
    assert "Orion и Vega" in await snippet.inner_text()
    assert await snippet.locator('a[href="https://example.invalid/web-fixture"]').count() == 1
    assert await finish_badge.get_attribute("data-tone") == "warning"

    await submit("legacy", open_technical=False)
    await expand_detail(journey)
    assert await stage("legacy").count() >= 1
    assert await stage("check").count() == 0
    await page.set_viewport_size({"width": 1280, "height": 1000})


async def check_warm_cache_webkit(url):
    async with async_playwright() as playwright:
        browser = await playwright.webkit.launch()
        try:
            page = await browser.new_page()
            await check_cached_workbench(page, url)
            await check_quote_progress(page)
        finally:
            await browser.close()
    print("WebKit: previous stylesheet superseded on desktop and mobile.")


async def check(url):
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            **(
                {"executable_path": os.environ["CHROMIUM_PATH"]}
                if os.getenv("CHROMIUM_PATH")
                else {}
            ),
            args=["--no-sandbox"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 1000},
            permissions=["clipboard-read", "clipboard-write"],
        )
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await check_cached_workbench(page, url, capture=True)
        await page.set_viewport_size({"width": 1280, "height": 1000})

        async def submit(question, *, open_technical=True):
            if await page.locator("#newquestion").is_visible():
                await page.locator("#newquestion").click()
            await page.locator("#question").fill(question)
            await page.locator("#question").press("Control+Enter")
            await page.locator('#result[aria-busy="false"]').wait_for()
            if question != "http-error":
                technical = page.locator("details.trace-panel")
                assert await technical.count() == 1
                assert not await technical.evaluate("panel => panel.open")
                if open_technical:
                    await technical.locator(":scope > summary").click()

        await check_journey(page, submit)
        await submit(QUESTION)
        assert "база знаний (1)" in await page.locator(".route-used-sources").inner_text()
        assert "только база знаний" in await page.locator(".route-policy").inner_text()
        assert "Поиск в отчётах" in await page.locator(".route-executed").inner_text()
        assert "Цель шага контроллера" in await page.locator(".route-purpose").inner_text()
        assert "2026-11-17" in await page.locator(".trace-time").inner_text()
        assert "Europe/Moscow" in await page.locator(".trace-time").inner_text()
        assert await page.locator(".trace-step").count() == 4
        assert await page.locator(".step-state.rejected").count() == 1
        assert "Найдено фрагментов: 0" in await page.locator(".trace-panel").inner_text()
        assert await page.locator(".trace-metrics strong").all_text_contents() == [
            await page.locator(".trace-metrics strong").first.inner_text(),
            "2",
            "3",
        ]
        await (
            page.locator(".inspector-detail > summary").filter(has_text="Как понят запрос").click()
        )
        assert "Orion Vega" in await page.locator(".trace-fields").inner_text()
        await (
            page.locator(".inspector-detail > summary").filter(has_text="Как понят запрос").click()
        )
        arguments = (
            page.locator(".step-detail > summary")
            .filter(has_text="Аргументы и результат шага")
            .first
        )
        await arguments.click()
        assert "Orion Vega properties" in await arguments.locator("..").inner_text()
        await arguments.click()
        assessment = (
            page.locator(".step-detail > summary").filter(has_text="Какие сведения найдены").first
        )
        await assessment.click()
        assert "Не хватает: Значения API" in await assessment.locator("..").inner_text()
        await assessment.click()
        assert "Повтор шага 1" in await page.locator(".trace-panel").inner_text()
        assert "Новые данные: да" in await page.locator(".trace-panel").inner_text()
        assert "отдельного вызова LLM" in await page.locator(".step-origin").inner_text()
        assert await page.locator(".trace-metrics strong").last.inner_text() == "3"
        handoff = page.locator(".inspector-detail > summary").filter(
            has_text="Как источники переданы в ответ"
        )
        await handoff.click()
        assert "переданы источники R1" in await page.locator(".answer-check").inner_text()
        assert await page.locator(".event-chronology").count() == 0
        assert (
            "Модель предложила утверждений: 1" in await page.locator(".answer-check").inner_text()
        )
        await handoff.click()
        await page.locator(".text-button").click()
        assert await page.evaluate("navigator.clipboard.readText()") == RESPONSE["trace_id"]
        await page.locator("#result").screenshot(path=str(OUT / "trace-workbench-desktop.png"))
        await (
            page.locator(".inspector-detail > summary")
            .filter(has_text="Найденные фрагменты")
            .click()
        )
        await page.locator(".retrieved-source > summary").click()
        assert TEXT in await page.locator(".source-preview").inner_text()
        await (
            page.locator(".inspector-detail > summary").filter(has_text="Полный ответ API").click()
        )
        async with page.expect_download() as info:
            await page.locator(".download-button").click()
        download = await info.value
        saved = json.loads(Path(await download.path()).read_text())
        assert saved == RESPONSE
        await page.set_viewport_size({"width": 390, "height": 844})
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        await (
            page.locator(".inspector-detail > summary")
            .filter(has_text="Найденные фрагменты")
            .click()
        )
        await (
            page.locator(".inspector-detail > summary").filter(has_text="Полный ответ API").click()
        )
        await page.locator("#result").screenshot(path=str(OUT / "trace-workbench-mobile.png"))
        # The API fixture includes price history. The active workbench does
        # not expose a chart, while the forecast table below must still work.
        assert await page.locator("#market, #historychart").count() == 0
        await submit("partial")
        assert "Достигнут лимит шагов" in await page.locator(".trace-stop").inner_text()
        await submit("no_progress")
        assert (
            "Повторные попытки не добавили данных" in await page.locator(".trace-stop").inner_text()
        )
        await submit("time_budget_exhausted")
        assert "Достигнут общий лимит времени" in await page.locator(".trace-stop").inner_text()
        await submit("research_budget_exhausted")
        assert "резервом времени на ответ" in await page.locator(".trace-stop").inner_text()
        await submit("controller")
        assert (
            "Контроллер перевёл запрос к ответу" in await page.locator(".trace-panel").inner_text()
        )
        await submit("unknown-reference")
        await (
            page.locator(".step-detail > summary")
            .filter(has_text="Какие сведения найдены")
            .last.click()
        )
        assert "R999" in await page.locator(".trace-panel").inner_text()
        for scenario in ("clarification", "unavailable", "outside"):
            await submit(scenario)
            assert await page.locator(".trace-step").count() == 0
            assert "Вызовов инструментов нет" in await page.locator(".empty-trace").inner_text()
            assert "Вызовов инструментов нет" in await page.locator(".route-executed").inner_text()
            assert await page.locator(".route-purpose").count() == 0
        await submit("Что ты можешь?")
        assert await page.locator(".answer-panel h2").inner_text() == "Возможности агента"
        assert await page.locator(".response-status").inner_text() == "Ответ получен"
        assert "рассчитать прогноз" in await page.locator(".answer-panel").inner_text()
        assert not await page.locator(".route-summary, .trace-footnote, .save-note").count()
        assert "встроенную справку" in await page.locator(".trace-panel").inner_text()
        assert await page.locator(".trace-metrics strong").nth(1).inner_text() == "0"
        await page.locator(".trace-panel").get_by_text("Как понят запрос", exact=True).click()
        assert "Вопрос о возможностях агента" in await page.locator(".trace-fields").inner_text()
        assert await page.locator(".download-button").count() == 1
        await submit("web-only")
        assert "интернет (1)" in await page.locator(".route-used-sources").inner_text()
        assert "Веб-поиск" in await page.locator(".route-executed").inner_text()
        assert "Поиск в отчётах" not in await page.locator(".route-executed").inner_text()
        assert "Обязательной очереди нет" in await page.locator(".route-policy").inner_text()
        assert "Отклонённых попыток: 1" in await page.locator(".route-attempts").inner_text()
        assert (
            "Цель модели: Проверить перечень в интернете."
            in await page.locator(".route-purpose").inner_text()
        )
        assert "Пустая попытка" not in await page.locator(".route-summary").inner_text()
        caveats = await page.locator(".route-caveat").all_text_contents()
        assert any(
            "[W4] дата публикации неизвестна" in line and "поисковой выдачи" in line
            for line in caveats
        )
        assert any("отключена в настройках" in line for line in caveats)
        await (
            page.locator(".inspector-detail > summary")
            .filter(has_text="Как источники переданы в ответ")
            .click()
        )
        assert "Покрытие вопроса: не проверено" in await page.locator(".answer-check").inner_text()
        assert "Полнота перечня не подтверждена" in await page.locator(".answer-check").inner_text()
        await submit("latest-event")
        await handoff.click()
        publisher = page.locator(".trace-step .trace-publisher")
        excluded = page.locator(".answer-check .event-excluded")
        reused = page.locator(".answer-check .event-reused")
        assert await publisher.inner_text() == "Поиск ограничен сайтом example.invalid."
        assert "Из проверки дат исключены публикации W3." in await excluded.inner_text()
        assert "результаты предыдущих шагов для фрагментов W1:B1." in await reused.inner_text()
        chronology = page.locator(".answer-check .event-chronology")
        assert await chronology.is_visible()
        assert (
            await chronology.inner_text()
            == "По найденным материалам последнее подтверждённое событие состоялось 2026-11-01 · W2:B1."
        )
        await submit("latest-event-unconfirmed")
        await handoff.click()
        assert await publisher.count() == await excluded.count() == await reused.count() == 0
        assert (
            await chronology.inner_text()
            == "По найденным материалам пока не удалось подтвердить дату состоявшегося события."
        )
        await submit("failed-only")
        assert (
            "Успешных вызовов инструментов нет"
            in await page.locator(".route-executed").inner_text()
        )
        assert "Вызовов с ошибкой: 1" in await page.locator(".route-attempts").inner_text()
        assert await page.locator(".route-purpose").count() == 0
        await submit("legacy")
        assert await page.locator(".trace-legacy").count() == 1
        assert "отключена в настройках" not in await page.locator(".route-summary").inner_text()
        calculation_summary = await page.evaluate(
            """() => window.TraceView.render({agent_mode: 'loop', status: 'ok',
              sources: [{id: 'M1', kind: 'model'}], trace: [],
              diagnostics: {semantic_review_enabled: false,
                plan: {intent: 'forecast', wants_context: false}, answer_checks: []}
            }, 0).querySelector('.route-summary').textContent"""
        )
        assert "отключена в настройках" not in calculation_summary
        assert "расчёт (1)" in calculation_summary
        await page.locator(".forecast-details > summary").click()
        assert await page.locator(".forecast-details tbody tr").count() == len(
            LEGACY["forecast"]["rows"]
        )
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        await submit("hostile")
        hostile_journey = page.locator("details.request-journey")
        await expand_detail(hostile_journey)
        for detail in await hostile_journey.locator("details.journey-node").all():
            await expand_detail(detail)
        assert "<img src=x" in await hostile_journey.inner_text()
        assert await page.locator("#result img, #result script").count() == 0
        assert not await page.evaluate("window.injected === true")
        assert await page.locator('a[href^="javascript:"]').count() == 0
        await submit("http-error")
        assert "503" in await page.locator("#result").inner_text()
        assert await page.locator(".trace-panel, .request-journey").count() == 0
        assert not await page.locator("button.submit").is_disabled()
        await check_quote_progress(page)
        await check_dialogue_continuations(page)
        assert not errors, errors
        await browser.close()
    archive = ROOT / "archive/user-interface-2026-11-11"
    manifest = json.loads((archive / "manifest.json").read_text())
    assert all(
        hashlib.sha256((archive / name).read_bytes()).hexdigest() == sha
        for name, sha in manifest["sha256"].items()
    )
    result = dict(
        kind="browser_fixture_check",
        new_llm_run=False,
        scenarios=[
            "new stylesheet and scripts with a cached previous stylesheet",
            "interactive request journey: chronology, keyboard, sources and claim verdicts",
            "model refusal after successful retrieval, partial response and legacy journey",
            "steps and duplicate rejection",
            "plan and arguments",
            "retrieved evidence",
            "copy and download",
            "mobile",
            "price history chart removed, forecast table preserved",
            "partial",
            "evidence assessment and missing facts",
            "no progress and time budget stops",
            "controller override and unknown references",
            "evidence handoff and answer reserve",
            "latest completed event date and supporting block within the retrieved evidence",
            "policy step separate from model choices",
            "source route summary: reports, web and failed attempts",
            "used web source caveats and disabled semantic review",
            "clarification",
            "clarification after research and continuation with the chosen metric",
            "quote fallback visible during search, direct quote and interrupted stream",
            "forecast dialogue continuation, retry, scope reset and opt-out",
            "unavailable",
            "outside",
            "capabilities help without source warnings and with downloadable trace",
            "legacy forecast",
            "untrusted text and URLs",
            "HTTP error",
        ],
        page_errors=errors,
        archive_hashes_valid=True,
    )
    (OUT / "trace-workbench-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))
    # Optional text transport for review clients that cannot download binary
    # Actions artifacts. These screenshots contain UI fixtures only.
    if os.getenv("TRACE_UI_PREVIEW") == "true":
        print("::group::UI fixture screenshots (base64)")
        for name in (
            "trace-workbench-home-desktop.jpg",
            "trace-workbench-home-mobile.jpg",
            "trace-workbench-desktop.png",
            "trace-workbench-mobile.png",
        ):
            print("UI_PREVIEW " + name + " " + base64.b64encode((OUT / name).read_bytes()).decode())
        print("::endgroup::")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warm-cache-webkit", action="store_true", help="Run the WebKit cache upgrade check only."
    )
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        run = check_warm_cache_webkit if args.warm_cache_webkit else check
        asyncio.run(run(f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()
