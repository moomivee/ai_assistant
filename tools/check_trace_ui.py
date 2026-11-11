"""Browser checks with explicit UI fixtures, not a live LLM quality evaluation.

uv run --with playwright python tools/check_trace_ui.py
Install Chromium with `playwright install chromium`, or set CHROMIUM_PATH.
"""

import asyncio
import base64
import copy
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
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
        action="report_search",
        purpose="Найти характеристики двух образцов в отчёте.",
        query="Orion Vega properties",
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
    ),
    dict(
        step=2,
        tool="report_search",
        purpose=DECISIONS[1]["purpose"],
        arguments={"query": "Orion Vega properties"},
        status="rejected",
        ms=None,
        error="Duplicate action rejected. Change the query.",
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
    ),
)
LEGACY = next(
    r["response"]
    for r in json.loads((ROOT / "eval/results/agent-audit-7b-dev.json").read_text())["rows"]
    if r["response"].get("forecast")
)


class Handler(BaseHTTPRequestHandler):
    def send(self, body, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            data = dict(
                model="UI fixture · без LLM",
                tavily=False,
                agent_mode="loop",
                agent_max_steps=6,
                reports=[],
                history=[
                    dict(month="2025-01", Brent=70, WTI=65),
                    dict(month="2025-02", Brent=72, WTI=67),
                ],
            )
            return self.send(json.dumps(data).encode(), "application/json")
        files = {
            "/": ("index.html", "text/html"),
            "/static/site.js": ("site.js", "application/javascript"),
            "/static/trace.js": ("trace.js", "application/javascript"),
            "/static/site.css": ("site.css", "text/css"),
        }
        if self.path in files:
            name, mime = files[self.path]
            return self.send((ROOT / "src/static" / name).read_bytes(), mime + "; charset=utf-8")
        self.send(b"Not found", "text/plain", 404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        question = payload["question"]
        response = copy.deepcopy(RESPONSE)
        if question == "legacy":
            response = LEGACY
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
        elif question == "http-error":
            return self.send(b"{}", "application/json", 503)
        elif question == "hostile":
            text = '<img src=x onerror="window.injected=true">'
            response["answer"] = text
            response["decisions"][0]["purpose"] = text
            response["sources"][0].update(
                title=text, url="javascript:window.injected=true", text=text
            )
            response["diagnostics"]["retrieved_sources"] = response["sources"]
        response["question"] = question
        self.send(json.dumps(response, ensure_ascii=False).encode(), "application/json")

    def log_message(self, *_):
        pass


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
        await page.goto(url)
        await page.locator("#mode").filter(has_text="UI fixture").wait_for()

        async def submit(question):
            if await page.locator("#newquestion").is_visible():
                await page.locator("#newquestion").click()
            await page.locator("#question").fill(question)
            await page.locator("#question").press("Control+Enter")
            await page.locator('#result[aria-busy="false"]').wait_for()

        await submit(QUESTION)
        assert await page.locator(".trace-step").count() == 4
        assert await page.locator(".step-state.rejected").count() == 1
        assert "Найдено фрагментов: 0" in await page.locator(".trace-panel").inner_text()
        assert await page.locator(".trace-metrics strong").all_text_contents() == [
            await page.locator(".trace-metrics strong").first.inner_text(),
            "2",
            "4",
        ]
        await (
            page.locator(".inspector-detail > summary").filter(has_text="Как понят запрос").click()
        )
        assert "Orion Vega" in await page.locator(".trace-fields").inner_text()
        await (
            page.locator(".inspector-detail > summary").filter(has_text="Как понят запрос").click()
        )
        await page.locator(".step-detail > summary").first.click()
        assert "Orion Vega properties" in await page.locator(".step-detail").first.inner_text()
        await page.locator(".step-detail > summary").first.click()
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
        await page.locator("#market > summary").click()
        await page.locator("#historychart svg").wait_for()
        await submit("partial")
        assert "Достигнут лимит шагов" in await page.locator(".trace-stop").inner_text()
        for scenario in ("clarification", "unavailable", "outside"):
            await submit(scenario)
            assert await page.locator(".trace-step").count() == 0
            assert "Вызовов инструментов нет" in await page.locator(".empty-trace").inner_text()
        await submit("legacy")
        assert await page.locator(".trace-legacy").count() == 1
        await page.locator(".forecast-details > summary").click()
        assert await page.locator(".forecast-details tbody tr").count() == len(
            LEGACY["forecast"]["rows"]
        )
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        await submit("hostile")
        assert await page.locator("#result img, #result script").count() == 0
        assert not await page.evaluate("window.injected === true")
        assert await page.locator('a[href^="javascript:"]').count() == 0
        await submit("http-error")
        assert "503" in await page.locator("#result").inner_text()
        assert await page.locator(".trace-panel").count() == 0
        assert not await page.locator("button.submit").is_disabled()
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
            "steps and duplicate rejection",
            "plan and arguments",
            "retrieved evidence",
            "copy and download",
            "mobile",
            "chart on demand",
            "partial",
            "clarification",
            "unavailable",
            "outside",
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
        print("::group::UI fixture screenshots (base64 PNG)")
        for name in ("trace-workbench-desktop.png", "trace-workbench-mobile.png"):
            print("UI_PREVIEW " + name + " " + base64.b64encode((OUT / name).read_bytes()).decode())
        print("::endgroup::")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        asyncio.run(check(f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()
