.PHONY: install prepare run test eval eval-agent eval-routing eval-ui check
install:
	uv sync --locked
prepare:
	uv run python -m src.cli prepare
run:
	uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
test:
	uv run python -m pytest -q
check:
	uv run ruff format --check src eval tests
	uv run ruff check src eval tests
	uv run python -m pytest -q
	uv run python -m eval.run validate-data
eval:
	uv run python -m eval.run retrieval
eval-agent:
	uv run python -m eval.run agent --split dev --judge
eval-routing:
	uv run python -m eval.run routing --split dev
eval-ui:
	uv run python -m eval.ui
