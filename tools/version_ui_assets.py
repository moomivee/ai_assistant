"""Refresh workbench asset URLs after editing src/static. Run with Python 3."""

import hashlib
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "static"
ASSET_URL = re.compile(r"""(?P<url>/static/(?P<file>[\w.-]+))(?:\?v=[0-9a-f]+)?(?=["'])""")


def refresh_versions(directory: Path):
    def versioned(match):
        contents = (directory / match["file"]).read_bytes()
        return f"{match['url']}?v={hashlib.sha256(contents).hexdigest()[:16]}"

    # The HTML references the CSS, which in turn references the provider mark.
    for name in ("site.css", "index.html"):
        path = directory / name
        before = path.read_text(encoding="utf-8")
        after = ASSET_URL.sub(versioned, before)
        if after != before:
            path.write_text(after, encoding="utf-8")


if __name__ == "__main__":
    refresh_versions(STATIC)
