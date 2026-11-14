"""Small, disposable cache of accepted public web evidence (never credentials)."""

import hashlib
import json
import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from ..config import settings

MAX_ENTRIES = 128
MAX_ENTRY_BYTES = 128_000


def _path(provider: str, parameters: dict) -> Path:
    key = json.dumps(
        {"version": 1, "provider": provider, "parameters": parameters},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return settings().cache / "web" / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def get(provider: str, parameters: dict) -> dict | None:
    path = _path(provider, parameters)
    try:
        if path.stat().st_size > MAX_ENTRY_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("expires_at", 0) <= time.time():
            path.unlink(missing_ok=True)
            return None
        result = data.get("result")
        return result if isinstance(result, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def put(provider: str, parameters: dict, result: dict, *, ttl: int) -> None:
    """Cache failures never prevent answering; replacement is atomic across workers."""
    temporary = None
    try:
        path = _path(provider, parameters)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            {"expires_at": time.time() + ttl, "result": result}, ensure_ascii=False
        ).encode("utf-8")
        if len(encoded) > MAX_ENTRY_BYTES:
            return
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.replace(temporary, path)
        entries = sorted(path.parent.glob("*.json"), key=lambda item: item.stat().st_mtime)
        for expired in entries[:-MAX_ENTRIES]:
            expired.unlink(missing_ok=True)
    except OSError:
        pass
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
