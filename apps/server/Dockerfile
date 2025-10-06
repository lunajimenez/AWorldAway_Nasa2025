FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

ENV PYTHONPATH=/app/src

COPY pyproject.toml /app/
COPY src/ /app/src/
COPY models/ /app/models/
COPY assets/ /app/assets/

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip pip install -r requirements.txt


EXPOSE 8000

CMD [ "uv", "run", "-m", "src.main" ]