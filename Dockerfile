FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
COPY petro ./petro
COPY data ./data
RUN python -m petro.cli prepare
EXPOSE 8000
CMD ["python","-m","uvicorn","petro.api:app","--host","0.0.0.0","--port","8000"]
