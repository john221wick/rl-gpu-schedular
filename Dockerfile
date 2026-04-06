FROM python:3.12-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv run python - <<'PY'
from server.app import app

assert any(getattr(route, "path", None) == "/" for route in app.routes), "Root route is missing"
PY
RUN uv sync --frozen

EXPOSE 7860

CMD ["uv", "run", "server", "--host", "0.0.0.0", "--port", "7860"]
