import hashlib
import re
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from src.api import app


def test_workbench_serves_current_assets_under_content_versioned_urls():
    with TestClient(app) as client:
        home = client.get("/")
        direct = client.get("/static/index.html")
        assert home.status_code == direct.status_code == 200
        assert home.content == direct.content
        assert home.headers["cache-control"] == direct.headers["cache-control"] == "no-store"

        def fetch_assets(text):
            urls = re.findall(r"""["'](/static/[^"']+)["']""", text)
            assert urls
            assets = {}
            for url in urls:
                path = urlsplit(url)
                response = client.get(url)
                assert response.status_code == 200
                digest = hashlib.sha256(response.content).hexdigest()[:16]
                assert path.query == f"v={digest}", "Run python tools/version_ui_assets.py"
                assets[path.path] = response
            return assets

        assets = fetch_assets(home.text)
        css = assets["/static/site.css"]
        assert css.headers["content-type"].startswith("text/css")
        mark = fetch_assets(css.text)["/static/gigachat-mark.png"]
        assert mark.headers["content-type"] == "image/png"
