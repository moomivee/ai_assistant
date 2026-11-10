# AI-агент «Нефтегазовый аналитик»

Агент для работы с отраслевыми отчётами, свежими публикациями и расчётом цен нефти. Разработан на **LangGraph и Ouroboros**: сначала проверяет загруженные PDF, затем при необходимости обращается к веб-источникам. Прогноз Brent/WTI рассчитывается отдельным инструментом, Urals — как сценарий с заданным дисконтом.

**Стек:** Python 3.12, uv, FastAPI, FAISS, ONNX-эмбеддеры, OpenAI-совместимый LLM-клиент, Langfuse. Локальная Qwen 2.5 3B выбрана как доступная стартовая модель.

## Запуск

После установки [uv](https://docs.astral.sh/uv/getting-started/installation/) и [Ollama](https://ollama.com/download):

```bash
git clone https://github.com/moomivee/ai_assistant.git
cd ai_assistant
cp -n .env.example .env
uv sync --locked
ollama pull qwen2.5:3b
uv run python -m src.cli prepare
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Ollama должна быть запущена с контекстом **8192 токена**; настройка и обновление описаны в [инструкции](docs/Setup.md). Интерфейс: **http://localhost:8000**. Запуск приложения и Ollama в Docker:

```bash
docker compose -f compose.yaml -f compose.local.yaml up --build -d
```

## Примеры

- «Каково содержание серы в Sokol? Используй только загруженные отчёты».
- «Покажи последнее опубликованное дневное наблюдение Brent».
- «Сравни экспорт нефти в Индию из отчёта EIA со свежими новостями».
- «Спрогнозируй Brent на 3 месяца. Только расчёт».
- «Напиши рецепт борща» — проверка границ специализации.

## Проверки и документация

```bash
uv run python -m pytest -q
uv run python -m eval.run retrieval
uv run python -m eval.ui
```

Eval-интерфейс: **http://localhost:8001**. Набор содержит 100 вопросов, из них 40 с эталонными фрагментами PDF. Проверки кода: **80 тестов**. В отдельном прогоне трёх регрессий с реальными Qwen 3B и 7B успешно пройдены **2 из 3** для каждой модели. Сравнение Urals/ESPO остаётся известным ограничением; версия является учебным прототипом. [Результаты и ограничения](docs/Results.md).

Актуальная документация архитектуры: [схема и файлы](docs/Architecture.md), [eval](docs/Evaluation.md), [трейсы и промпты](docs/Observability.md), [установка и обновление](docs/Setup.md). 

