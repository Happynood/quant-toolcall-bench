FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv==0.4.30

COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/

RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["quantcall"]
CMD ["--help"]
