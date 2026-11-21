# API image: serves the live record and the battery planner. It trains nothing.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app \
    && mkdir /app && chown app /app

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

USER app
WORKDIR /app
COPY --chown=app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra api --no-install-project
COPY --chown=app src ./src
COPY --chown=app configs ./configs
# the live record as of the build; mount a newer copy over /app/docs/live to serve fresh data
COPY --chown=app docs ./docs
RUN uv sync --frozen --no-dev --extra api

ENV PATH="/app/.venv/bin:$PATH" \
    LIVE_DIR=/app/docs/live \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "gridwise.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
