FROM python:3.12-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOME=/home/user \
    PATH=/app/.venv/bin:/home/user/.local/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        curl \
        git \
        git-lfs \
        procps \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv run python - <<'PY'
from server.app import app

assert any(getattr(route, "path", None) == "/" for route in app.routes), "Root route is missing"
PY
RUN uv sync --frozen
RUN chown -R 1000:1000 /app /home/user

USER 1000

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
