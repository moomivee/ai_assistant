"""Switch an unchanged historical .env template to the maintained web source list."""

import os
import re
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile

OLD_DEFAULT = (
    b"WEB_DOMAINS=eia.gov,fred.stlouisfed.org,opec.org,iea.org,cbr.ru,economy.gov.ru,"
    b"minfin.gov.ru,minenergo.gov.ru,reuters.com,interfax.ru,consilium.europa.eu,"
    b"ofac.treasury.gov"
)
TEMPLATE_LINES = {OLD_DEFAULT, OLD_DEFAULT + b",ru.wikipedia.org,en.wikipedia.org"}


def update(path: Path) -> str:
    if not path.is_file():
        return "Файл .env отсутствует. При запуске применяется конфигурация окружения."
    path = path.resolve()
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    entries = [line for line in lines if re.match(rb"\s*(?:export\s+)?WEB_DOMAINS\s*=", line)]
    if not entries:
        return "В .env список не закреплён. Используются домены из конфигурации приложения."
    if len(entries) != 1 or entries[0].rstrip(b"\r\n") not in TEMPLATE_LINES:
        return "Сохранён пользовательский WEB_DOMAINS. Изменить его можно в .env."
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        tmp_path = Path(temp.name)
        try:
            for line in lines:
                temp.write(b"# " + line if line == entries[0] else line)
            temp.flush()
            os.fchmod(temp.fileno(), stat.S_IMODE(path.stat().st_mode))
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return "Прежний список в .env закомментирован. При пересоздании сервис получит новые домены."


if __name__ == "__main__":
    print(update(Path(".env")))
