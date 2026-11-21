"""Browser fixture checks for the v3 workbench; no live model/API evaluation.

uv run --with playwright python tools/check_knowledge_ui.py
Set CHROMIUM_PATH if Chromium is not on PATH; screenshots go to a temporary
folder (or --out PATH), never to the repository's documentation assets.
"""

import argparse
import asyncio
import copy
import json
import os
import shutil
import tempfile
import threading
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from check_trace_ui import RESPONSE as TRACE_RESPONSE
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
QUOTE = "Образец Orion: плотность 42° API, содержание серы 0,2%."
HOSTILE = '<img src=x onerror="window.injected=true">'
DOCUMENT = dict(
    id="fixture-doc",
    document_id="fixture-doc",
    title="Обзор качества нефти",
    publisher="Тестовый издатель",
    source_url="https://example.invalid/report.pdf",
    published_date="2026-11-10",
    builtin=True,
    archived=False,
    status="ready",
    current_version_id="v1",
    active_version_id="v1",
    versions_count=1,
    filename="report.pdf",
    size_bytes=12400,
    block_count=2,
    warnings=[],
    index_status="pending",
)
STATE = {
    "documents": [copy.deepcopy(DOCUMENT)],
    "asks": [],
    "notes": [],
    "uploads": [],
    "polls": 0,
    "list_error": False,
}


class Handler(BaseHTTPRequestHandler):
    def respond(self, data, status=200):
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode(), "application/json", status)

    def send_bytes(self, body, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            return self.respond(
                dict(
                    model="GigaChat-2",
                    provider="gigachat",
                    llm_configured=True,
                    tavily=True,
                    agent_mode="loop",
                    agent_max_steps=6,
                    reports=[],
                    history=[],
                )
            )
        if path == "/api/rag/status":
            return self.respond(dict(model="multilingual-e5-small", mode="hybrid", top_k=6))
        if path == "/api/knowledge/documents":
            if STATE["list_error"]:
                return self.respond({"detail": "Тестовая ошибка каталога"}, 503)
            STATE["polls"] += 1
            for doc in STATE["documents"]:
                if doc["status"] == "queued" and STATE["polls"] > doc.get("ready_after", 0):
                    doc.update(
                        status="ready", active_version_id=doc["current_version_id"], block_count=2
                    )
            return self.respond({"documents": STATE["documents"]})
        if path == "/api/notes":
            return self.respond({"notes": [n for n in STATE["notes"] if n["state"] == "active"]})
        if path.startswith("/api/notes/"):
            note = next((n for n in STATE["notes"] if n["note_id"] == path.split("/")[-1]), None)
            return self.respond(note or {"detail": "Не найдено"}, 200 if note else 404)
        if path.startswith("/api/knowledge/documents/"):
            parts = path.split("/")
            doc = next((d for d in STATE["documents"] if d["id"] == parts[4]), None)
            if not doc:
                return self.respond({"detail": "Документ не найден"}, 404)
            if path.endswith("/update-capability"):
                return self.respond({"supported": doc["builtin"], "discovers_new_releases": False})
            if path.endswith("/blocks"):
                return self.respond(
                    {
                        "blocks": [
                            dict(
                                block_id="B1",
                                text=QUOTE,
                                page=3,
                                method="native",
                                locator={"label": "Страница PDF 3"},
                            ),
                            dict(
                                block_id="B2",
                                text=HOSTILE,
                                page=4,
                                method="ocr",
                                locator={"label": "Страница PDF 4"},
                            ),
                            dict(
                                block_id="B3",
                                text="Сорт | Плотность\nOrion | 42",
                                page=None,
                                method="native",
                                locator={
                                    "type": "xlsx",
                                    "sheet": "Качество",
                                    "cell_range": "A1:B2",
                                },
                            ),
                            dict(
                                block_id="B4",
                                text="Условия сравнения образцов.",
                                page=None,
                                method="native",
                                locator={"type": "docx", "paragraph": 5},
                            ),
                        ],
                        "warnings": [],
                    }
                )
            if path.endswith("/file"):
                return self.send_bytes(b"%PDF-1.4 fixture", "application/pdf")
            if len(parts) == 5:
                versions = [dict(id="v1", status="ready", created_at="2026-11-10T12:00:00Z")]
                if doc["versions_count"] > 1:
                    versions.insert(
                        0, dict(id="v2", status=doc["status"], created_at="2026-11-14T12:00:00Z")
                    )
                return self.respond({**doc, "versions": versions})
        if path == "/static/gigachat-mark.png":
            return self.send_bytes(
                (ROOT / "src/static/gigachat-mark.png").read_bytes(), "image/png"
            )
        file = "index.html" if path == "/" else path.removeprefix("/static/")
        if file in {"index.html", "site.js", "site.css", "trace.js", "knowledge.js"}:
            mime = (
                "text/html"
                if file.endswith("html")
                else "text/css"
                if file.endswith("css")
                else "application/javascript"
            )
            return self.send_bytes(
                (ROOT / "src/static" / file).read_bytes(), mime + "; charset=utf-8"
            )
        self.respond({"detail": "Не найдено"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path == "/api/knowledge/documents":
            message = BytesParser(policy=default).parsebytes(
                ("Content-Type: " + self.headers["Content-Type"] + "\r\n\r\n").encode() + body
            )
            fields = {}
            filename = ""
            for part in message.iter_parts():
                key = part.get_param("name", header="content-disposition")
                if key == "file":
                    filename = part.get_filename()
                else:
                    fields[key] = part.get_payload(decode=True).decode()
            STATE["uploads"].append({**fields, "filename": filename})
            if filename == "reject.pdf":
                return self.respond({"detail": "Файл превышает допустимый размер"}, 413)
            if fields.get("document_id"):
                doc = next(d for d in STATE["documents"] if d["id"] == fields["document_id"])
                doc.update(
                    current_version_id="v2",
                    versions_count=2,
                    status="queued",
                    ready_after=STATE["polls"] + 1,
                )
            else:
                doc = {
                    **DOCUMENT,
                    "id": "uploaded",
                    "document_id": "uploaded",
                    "title": fields.get("title") or filename,
                    "filename": filename,
                    "builtin": False,
                    "status": "queued",
                    "active_version_id": None,
                    "ready_after": STATE["polls"] + 1,
                }
                STATE["documents"].append(doc)
            return self.respond({"document": doc}, 202)
        data = json.loads(body) if body else {}
        if path == "/api/ask/stream":
            STATE["asks"].append(data)
            if data["question"] == "http-error":
                return self.respond({"detail": "Тестовая ошибка"}, 503)
            response = copy.deepcopy(TRACE_RESPONSE)
            response.update(
                question=data["question"],
                answer="Плотность образца Orion составляет 42° API. [R1]\n\nСодержание серы — 0,2%. [R1]",
                claims=[
                    {
                        "text": "Плотность образца Orion составляет 42° API.",
                        "source": "R1",
                        "quote": QUOTE,
                        "evidence_ids": ["R1:B1"],
                        "citations": [
                            {
                                "evidence_id": "R1:B1",
                                "source": "R1",
                                "quote": QUOTE,
                                "page": 3,
                                "locator": {"label": "Страница PDF 3"},
                                "local_url": "/api/knowledge/documents/fixture-doc/versions/v1/file#page=3",
                            }
                        ],
                    }
                ],
                sources=[
                    dict(
                        id="R1",
                        kind="report",
                        title=DOCUMENT["title"],
                        text=QUOTE,
                        date="2026-11-10",
                        page=3,
                        url=DOCUMENT["source_url"],
                        evidence={"R1:B1": QUOTE},
                        local_url="/api/knowledge/documents/fixture-doc/versions/v1/file#page=3",
                    )
                ],
            )
            if data["question"] == "hostile":
                response.update(answer=HOSTILE + " [R1]", claims=[])
                response["sources"][0].update(
                    title=HOSTILE, url="javascript:window.injected=true", text=HOSTILE
                )
            event = json.dumps({"type": "result", "result": response}, ensure_ascii=False)
            return self.send_bytes((event + "\n").encode(), "application/x-ndjson")
        if path == "/api/notes":
            note = {
                **data,
                "note_id": "note-fixture",
                "revision": "a" * 64,
                "state": "active",
                "verified": False,
                "updated_at": "2026-11-14T12:00:00Z",
            }
            STATE["notes"].append(note)
            return self.respond(note)
        if path.startswith("/api/notes/") and path.endswith("/archive"):
            if data.get("expected_revision") != STATE["notes"][0]["revision"]:
                return self.respond({"detail": {"message": "Версия заметки изменилась"}}, 409)
            STATE["notes"][0]["state"] = "archived"
            return self.respond(STATE["notes"][0])
        if path.endswith("/archive"):
            doc = next(d for d in STATE["documents"] if d["id"] == path.split("/")[4])
            doc.update(archived=True, status="archived", active_version_id=None)
            return self.respond({"document": doc})
        if path.endswith("/check-update"):
            return self.respond({"status": "unchanged", "notices": ["Новые URL не проверяются."]})
        self.respond({"detail": "Не найдено"}, 404)

    def log_message(self, *_):
        pass


async def check(url, out):
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        executable = os.environ.get("CHROMIUM_PATH") or shutil.which("chromium")
        browser = await p.chromium.launch(
            **({"executable_path": executable} if executable else {}), args=["--no-sandbox"]
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(url)
        await page.locator('#model-badge[data-provider="gigachat"]').wait_for()
        assert "GigaChat" in await page.locator("#mode").inner_text()
        await page.locator(".document-card").wait_for(state="attached")
        assert await page.get_by_role("tab").count() == 2
        capabilities = page.locator("#panel-test #capabilities .capability-query")
        assert await capabilities.count() == 6
        assert await page.locator("#question").is_visible()
        assert await page.locator("#tab-capabilities, #market, #historychart").count() == 0
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        await page.set_viewport_size({"width": 390, "height": 844})
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        await page.set_viewport_size({"width": 1440, "height": 900})
        examples = set()
        for button in await capabilities.all():
            await button.click()
            if await page.locator("#panel-knowledge").is_visible():
                assert await page.locator("#document-file").evaluate(
                    "el => el === document.activeElement"
                )
                await page.locator("#tab-test").click()
            else:
                examples.add(await page.locator("#question").input_value())
                assert await page.locator("#question").evaluate("q => q === document.activeElement")
        assert len(examples) == 5 and "" not in examples
        assert not STATE["asks"], "Examples must leave submission to the user"
        await page.locator("#tab-test").focus()
        await page.keyboard.press("ArrowRight")
        assert await page.locator("#tab-knowledge").get_attribute("aria-selected") == "true"
        await page.keyboard.press("Home")
        assert await page.locator("#tab-test").get_attribute("aria-selected") == "true"
        await page.keyboard.press("End")
        assert await page.locator("#tab-knowledge").get_attribute("aria-selected") == "true"
        await page.get_by_role("button", name="Проверить исходный файл").click()
        await page.locator("#knowledge-feedback").filter(has_text="тот же файл").wait_for()
        await page.get_by_role("button", name="Просмотр и версии").first.click()
        await page.locator("#preview-blocks").filter(has_text=QUOTE).wait_for()
        assert (
            "Лист «Качество» · Ячейки A1:B2" in await page.locator("#preview-blocks").inner_text()
        )
        assert "Абзац 5" in await page.locator("#preview-blocks").inner_text()
        assert await page.locator("#preview-blocks img").count() == 0
        assert not await page.evaluate("window.injected === true")
        assert (
            await page.get_by_role("link", name="Открыть страницу 3").get_attribute("href")
            == url + "/api/knowledge/documents/fixture-doc/versions/v1/file#page=3"
        )
        await page.get_by_role("button", name="Закрыть", exact=True).click()
        await page.get_by_role("button", name="Задать вопрос", exact=True).click()
        assert await page.locator("#report-only").is_checked()
        await page.locator("#question").fill("Сравни плотность")
        await page.locator("#question").press("Control+Enter")
        await page.locator('#result[aria-busy="false"]').wait_for()
        assert STATE["asks"][-1]["document_ids"] == ["fixture-doc"]
        assert STATE["asks"][-1]["report_only"] is True
        assert await page.locator(".trace-panel").count() == 1
        assert await page.locator(".citation-button").count() == 2
        await page.locator(".citation-button").first.click()
        excerpt = page.locator(".citation-excerpt").first
        assert QUOTE in await excerpt.inner_text()
        assert "Фрагмент, подтверждающий" in await excerpt.inner_text()
        assert "42° API" in await page.locator(".answer-text").first.inner_text()
        assert await page.locator(".answer-text").count() == 2, (
            "Claims must not duplicate answer paragraphs"
        )
        await page.screenshot(path=str(out / "answer-desktop.png"), full_page=True)
        await page.get_by_role("button", name="Сохранить в заметки", exact=True).click()
        await page.get_by_role("button", name="Заметка сохранена").wait_for()
        assert STATE["notes"][0]["source_refs"][0]["id"] == "R1"
        await page.locator("#notes-panel > summary").click()
        await page.locator(".research-note > summary").click()
        await page.get_by_role("button", name="Продолжить исследование").click()
        assert await page.locator("#note-context").is_visible()
        await page.locator("#question").fill("Уточни вывод")
        await page.locator("#question").press("Control+Enter")
        await page.locator('#result[aria-busy="false"]').wait_for()
        assert STATE["asks"][-1]["note_id"] == "note-fixture"
        await page.locator("#clear-note-context").click()
        await (
            page.locator(".research-note").get_by_role("button", name="В архив", exact=True).click()
        )
        await page.locator(".research-note").wait_for(state="detached")
        assert STATE["notes"][0]["state"] == "archived"
        await page.get_by_role("tab", name="База знаний").click()
        await page.get_by_role("button", name="Новая версия", exact=True).click()
        await page.locator("#document-file").set_input_files(
            {"name": "revision.pdf", "mimeType": "application/pdf", "buffer": b"%PDF fixture"}
        )
        await page.locator("#upload-button").click()
        await page.locator("#upload-feedback").filter(has_text="Файл принят").wait_for()
        assert STATE["uploads"][-1]["document_id"] == "fixture-doc"
        await page.locator(".document-card").filter(has_text="предыдущая готовая версия").wait_for()
        await page.locator(".document-card").filter(has_text="Текст готов").wait_for(timeout=10000)
        assert STATE["documents"][0]["active_version_id"] == "v2"
        await page.locator("#document-file").set_input_files(
            {"name": "new.csv", "mimeType": "text/csv", "buffer": b"name,value\nBrent,70"}
        )
        await page.locator("#document-title").fill("Таблица с данными")
        await page.locator("#upload-button").click()
        await page.locator(".document-card").filter(has_text="Таблица с данными").wait_for()
        await (
            page.locator(".document-card")
            .filter(has_text="Таблица с данными")
            .filter(has_text="Текст готов")
            .wait_for(timeout=10000)
        )
        assert "document_id" not in STATE["uploads"][-1]
        await page.locator("#document-file").set_input_files(
            {"name": "reject.pdf", "mimeType": "application/pdf", "buffer": b"%PDF fixture"}
        )
        await page.locator("#upload-button").click()
        await page.locator("#upload-feedback.is-error").filter(has_text="размер").wait_for()
        assert not await page.locator("#upload-button").is_disabled()
        await page.screenshot(path=str(out / "knowledge-desktop.png"), full_page=True)
        await page.set_viewport_size({"width": 390, "height": 844})
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        await page.screenshot(path=str(out / "knowledge-mobile.png"), full_page=True)
        await (
            page.locator(".document-card")
            .filter(has_text="Таблица с данными")
            .get_by_role("button", name="В архив", exact=True)
            .click()
        )
        await (
            page.locator(".document-card")
            .filter(has_text="Таблица с данными")
            .wait_for(state="detached")
        )
        assert await page.locator(".document-card").count() == 1
        await page.locator("#show-archived").check()
        assert await page.locator(".document-card").count() == 2
        archived = page.locator(".document-card").filter(has_text="Таблица с данными")
        assert await archived.locator('input[type="checkbox"]').is_disabled()
        await page.locator("#document-filter").fill("несуществующее название")
        assert "не найдено" in await page.locator("#document-list").inner_text()
        await page.locator("#document-filter").fill("")
        STATE["list_error"] = True
        await page.locator("#refresh-knowledge").click()
        await (
            page.locator("#knowledge-feedback.is-error")
            .filter(has_text="ошибка каталога")
            .wait_for()
        )
        STATE["list_error"] = False
        await page.locator("#refresh-knowledge").click()
        await page.locator("#tab-test").click()
        for question in ["hostile", "http-error"]:
            await page.locator("#question").fill(question)
            await page.locator("#question").press("Control+Enter")
            await page.locator('#result[aria-busy="false"]').wait_for()
            assert await page.locator("#result img, #result script").count() == 0
            assert not await page.evaluate("window.injected === true")
            assert await page.locator('a[href^="javascript:"]').count() == 0
        assert "503" in await page.locator("#result").inner_text()
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")

        for provider, model, state in [
            ("openai", "local-model", "other"),
            ("gigachat", None, "unconfigured"),
        ]:
            status = {
                "provider": provider,
                "llm_configured": bool(model),
                "model": model,
                "reports": [],
            }
            await page.route("**/api/status", lambda route: route.fulfill(json=status))
            await page.reload()
            await page.locator(f'#model-badge[data-provider="{state}"]').wait_for()
            if model:
                assert model in await page.locator("#mode").inner_text()
            else:
                assert "ключ" in await page.locator("#mode").inner_text()
            await page.unroute("**/api/status")
        await page.route("**/api/status", lambda route: route.fulfill(status=503, json={}))
        await page.reload()
        await page.locator('#model-badge[data-provider="unavailable"]').wait_for()
        assert await capabilities.count() == 6
        assert not errors, errors
        await browser.close()
    result = dict(
        kind="browser_fixture_check",
        live_llm=False,
        live_tavily=False,
        scenarios=[
            "inline capabilities without autorun",
            "configured, unconfigured and unavailable model status",
            "keyboard tabs",
            "catalog and scoped ask",
            "local OCR/text preview and provenance",
            "exact per-claim citations",
            "notes save and explicit continuation",
            "version upload and bounded polling",
            "source revision check",
            "new document upload",
            "archive and filters",
            "desktop and mobile",
            "upload/catalog/API errors",
            "untrusted text and URL safety",
        ],
        page_errors=errors,
        screenshots_directory=str(out),
    )
    (out / "knowledge-ui-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    output = args.out or Path(tempfile.mkdtemp(prefix="petro-knowledge-ui-"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        asyncio.run(check(f"http://127.0.0.1:{server.server_port}", output))
    finally:
        server.shutdown()
