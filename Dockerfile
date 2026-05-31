FROM python:3.10-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev
COPY src/ ./src/
RUN uv pip install --no-deps -e .
COPY data/ ./data/
COPY main.py .

EXPOSE 8000
ENV NAME=ProductSimilarityApp

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
