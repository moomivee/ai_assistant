"""Capture the archived consumer UI with a recorded, real evaluation response.

This is an interface replay, not a new model run or a latency measurement.
Run in CI or on a machine with Playwright Chromium installed.
"""

import asyncio
import csv
import hashlib
import io
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
UI = ROOT / "archive/user-interface-2026-11-11"
RESULT = ROOT / "eval/results/live-qwen-3b-2026-11-10.json"
CASE_ID = "regression-brent-explicit-001"
DATA = json.loads(RESULT.read_text())
RESPONSE = next(row["response"] for row in DATA["rows"] if row["case_id"] == CASE_ID)
QUESTION = RESPONSE["question"]
REPORTS = json.loads((ROOT / "data/reports.json").read_text())
with (ROOT / "data/prices/monthly.csv").open() as source:
    HISTORY = list(csv.DictReader(source))[-24:]
STATUS = {
    "model": RESPONSE["mode"],
    "tavily": False,
    "reports": [
        {"title": row["title"], "date": row["date"], "url": "/reports/" + row["file"]}
        for row in REPORTS
    ],
    "history": [
        {"month": row["month"], "Brent": float(row["Brent"]), "WTI": float(row["WTI"])}
        for row in HISTORY
    ],
}


class ReplayHandler(BaseHTTPRequestHandler):
    def send_body(self, body, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        self.send_body(json.dumps(data, ensure_ascii=False).encode(), "application/json", status)

    def do_GET(self):
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/static/site.css": ("site.css", "text/css; charset=utf-8"),
            "/static/site.js": ("site.js", "text/javascript; charset=utf-8"),
        }
        if self.path in files:
            name, mime = files[self.path]
            self.send_body((UI / name).read_bytes(), mime)
        elif self.path == "/api/status":
            self.send_json(STATUS)
        else:
            self.send_body(b"Not found", "text/plain", 404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if self.path == "/api/ask" and payload == {"question": QUESTION, "previous": None}:
            self.send_json(RESPONSE)
        else:
            self.send_json({"error": "Only the recorded evaluation question is available."}, 422)

    def log_message(self, *_):
        pass


async def capture(url):
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    async with async_playwright() as runtime:
        browser = await runtime.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1040, "height": 860}, device_scale_factor=1
        )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        async def frame(duration):
            shot = Image.open(io.BytesIO(await page.screenshot())).convert("RGB")
            frames.append((shot, duration))

        await page.goto(url, wait_until="networkidle")
        await page.locator("#mode").filter(has_text="Анализ с AI").wait_for()
        await page.locator("#market:not([hidden])").wait_for()
        await page.screenshot(path=str(OUT / "interface-home.png"), full_page=True)
        await frame(1600)
        await page.locator("#question").fill(QUESTION)
        await frame(1400)
        await page.locator("button.submit").click()
        await page.locator("#result h2").filter(has_text="Ответ аналитика").wait_for()
        shown = await page.locator("#result").inner_text()
        assert RESPONSE["status"] == "ok"
        assert all(paragraph in shown for paragraph in RESPONSE["answer"].split("\n\n"))
        await page.locator("#result").evaluate("node => window.scrollTo(0, node.offsetTop - 24)")
        await frame(2200)
        await page.locator(".forecast-details > summary").click()
        assert await page.locator(".forecast-details tbody tr").count() == 3
        await page.locator("#result").screenshot(path=str(OUT / "interface-forecast.png"))
        await page.locator(".forecast-details").scroll_into_view_if_needed()
        await frame(2400)
        await page.locator(".source > summary").click()
        await page.locator(".source").scroll_into_view_if_needed()
        await frame(2000)
        await page.locator(".library > summary").click()
        await page.locator("#market > summary").click()
        await page.locator("#historychart svg").wait_for()
        assert await page.locator("#historychart polyline").count() == 2
        await page.locator(".resources").screenshot(path=str(OUT / "interface-reports.png"))
        await page.locator(".resources").scroll_into_view_if_needed()
        await frame(2600)
        await page.evaluate("window.scrollTo(0, 0)")
        await frame(1200)
        assert not errors, errors
        await browser.close()

    palette_frames = [frame.quantize(colors=256) for frame, _ in frames]
    palette_frames[0].save(
        OUT / "interface-demo.gif",
        save_all=True,
        append_images=palette_frames[1:],
        duration=[duration for _, duration in frames],
        loop=0,
        optimize=True,
        disposal=2,
    )
    paths = [
        OUT / name
        for name in (
            "interface-home.png",
            "interface-forecast.png",
            "interface-reports.png",
            "interface-demo.gif",
        )
    ]
    provenance = {
        "kind": "interface_replay",
        "ui_source_directory": str(UI.relative_to(ROOT)),
        "archived_ui_commit": json.loads((UI / "manifest.json").read_text())["source_commit"],
        "ui_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "recorded_result": str(RESULT.relative_to(ROOT)),
        "case_id": CASE_ID,
        "question": QUESTION,
        "new_llm_run": False,
        "note": "Actual HTML/CSS/JS; recorded evaluation response; frame durations do not measure model latency.",
        "files": [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
    }
    (OUT / "capture.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), ReplayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        asyncio.run(capture(f"http://127.0.0.1:{server.server_address[1]}/"))
    finally:
        server.shutdown()
        server.server_close()
