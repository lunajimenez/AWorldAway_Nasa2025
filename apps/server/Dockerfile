FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copiar archivos necesarios
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project

ENV PYTHONPATH=/app/src
COPY . /app

RUN pip install -r requirements.txt || true

EXPOSE 8000
CMD ["uv", "run", "-m", "src.main"]
