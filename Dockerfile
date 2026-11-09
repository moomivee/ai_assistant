FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-cache
ENV PATH="/app/.venv/bin:$PATH"
COPY petro ./petro
COPY data ./data
RUN python -m petro.cli prepare
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "petro.api:app", "--host", "0.0.0.0", "--port", "8000"]
