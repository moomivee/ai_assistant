# Установка и обновление

Для проекта зафиксирован Python 3.12. `pyproject.toml` описывает прямые зависимости, `uv.lock` — разрешённые версии и хеши. `uv sync --locked` создаёт `.venv` и проверяет, что lock соответствует проекту; вручную ставить пакеты через pip не требуется.

## Обновление существующей копии

Сначала сохраните собственные изменения, если они есть. В каталоге проекта:

```bash
git status
git pull --ff-only origin main
uv sync --locked
```

Сравните свой `.env` с `.env.example`, добавьте новые настройки. Не заменяйте файл с ключами вслепую. 

## MacBook Air с Apple Silicon

1. Установите uv по [официальной инструкции](https://docs.astral.sh/uv/getting-started/installation/) и приложение [Ollama для macOS](https://ollama.com/download/mac). Запустите Ollama. 
2. Клонируйте репозиторий либо обновите существующий. Все команды далее выполняются из его корня.
3. Создайте конфигурацию, только если её ещё нет:

```bash
if [ ! -f .env ]; then
    cp .env.example .env
fi
uv sync --locked
ollama pull qwen2.5:3b
```

`LLM_API_KEY=local` — непустая заглушка для локального API Ollama. Для удалённого провайдера нужны его реальный ключ, base URL и модель с поддержкой JSON Schema. Одной OpenAI-совместимости endpoint недостаточно: structured outputs должны поддерживаться отдельно.

4. Соберите индекс и запустите API:

```bash
uv run python -m src.cli prepare
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Первая подготовка скачивает файлы эмбеддера. Следующие запуски используют `.cache`. Откройте http://localhost:8000. Терминал сервера оставьте открытым. Остановка — `Ctrl+C`.

5. В другом терминале можно проверить запрос:

```bash
uv run python -m src.cli ask "Спрогнозируй Brent на 3 месяца. Только расчёт"
```

Если LLM не настроена или возвращает неверную схему, ответ имеет статус `unavailable`. Проверьте работу Ollama, `ollama list`, имя модели, URL и таймаут. 

## Docker

Весь локальный комплект, включая Ollama:

```bash
docker compose -f compose.yaml -f compose.local.yaml up --build -d
docker compose -f compose.yaml -f compose.local.yaml ps
docker compose -f compose.yaml -f compose.local.yaml logs -f analyst
```

Первое скачивание модели может быть долгим: `pull-model` должен завершиться, затем стартует `analyst`. На Mac удобнее запускать Ollama нативно, а в `.env` для контейнера приложения указать:

```dotenv
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=local
```

Тогда запускается только приложение:

```bash
docker compose up --build -d
```

Остановка комплекта:

```bash
docker compose -f compose.yaml -f compose.local.yaml down
```

Не добавляйте `-v`, если хотите сохранить скачанную модель. PDF и CSV доступны из `./data`, локальные трейсы — из `./.cache/traces`. Индекс готовится при сборке; если PDF поменялись, он пересоберётся при обращении.

`/health` подтверждает жизнь HTTP-сервиса. `/ready` проверяет доступность индекса и списка моделей LLM.

## Частые команды

| Задача | Команда |
|---|---|
| Проверить код и данные | `make check` |
| Сравнить эмбеддеры | `make eval` |
| Запустить окно eval | `make eval-ui` |
| Прогнать агента с judge на dev | `make eval-agent` |
| Обновить историю EIA/FRED | `uv run python -m src.cli refresh-prices` |
| Создать версии промптов | `uv run python -m src.cli publish-prompts --label staging` |
| Добавить зависимость | `uv add имя-пакета` |
| Добавить инструмент разработки | `uv add --dev имя-пакета` |

