import pytest

from src.config import settings
from src.integrations.web import allowed
from tools.update_web_domains import TEMPLATE_LINES, update


def test_default_publishers_and_explicit_override(monkeypatch):
    monkeypatch.delenv("WEB_DOMAINS", raising=False)
    for host in ("www.rbc.ru", "ru.euronews.com", "www.ft.com", "www.theguardian.com"):
        assert allowed(f"https://{host}/news", settings().domains)
    monkeypatch.setenv("WEB_DOMAINS", "opec.org")
    settings.cache_clear()
    assert settings().domains == ("opec.org",)
    assert not allowed("https://www.rbc.ru/news", settings().domains)


@pytest.mark.parametrize("old", sorted(TEMPLATE_LINES))
def test_migrate_template_once_preserving_other_values_and_permissions(tmp_path, old):
    path = tmp_path / ".env"
    other = b"TAVILY_API_KEY=example\r\n"
    path.write_bytes(other + old + b"\r\n")
    path.chmod(0o600)
    update(path)
    update(path)
    assert path.read_bytes() == other + b"# " + old + b"\r\n"
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "custom", [b"WEB_DOMAINS=rbc.ru", b"WEB_DOMAINS=", b"export WEB_DOMAINS=opec.org"]
)
def test_migrate_preserves_custom_and_duplicate_definitions(tmp_path, custom):
    path = tmp_path / ".env"
    for content in (custom, next(iter(TEMPLATE_LINES)) + b"\n" + custom):
        path.write_bytes(content)
        update(path)
        assert path.read_bytes() == content
