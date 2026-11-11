# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv and the app code.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 fleetmaint
WORKDIR /app
COPY --from=build --chown=fleetmaint:fleetmaint /app /app
# The service caches bundle.pkl here on first start, so the non-root user must own it.
RUN mkdir -p /app/data && chown fleetmaint:fleetmaint /app/data
USER fleetmaint
ENV PATH="/app/.venv/bin:$PATH" FLEETMAINT_DATA_DIR=/app/data
EXPOSE 8884
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8884/healthz', timeout=2)"
# Build the bundle (about 20 seconds) if the data directory is empty, then serve.
CMD ["sh", "-c", "[ -f /app/data/bundle.pkl ] || fleetmaint --data-dir /app/data train; exec fleetmaint --data-dir /app/data serve --host 0.0.0.0 --port 8884"]
