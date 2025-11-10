FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /bin/

# Configuración de entorno virtual
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copiar dependencias base
COPY pyproject.toml uv.lock /app/

# Instalar dependencias congeladas (sin mounts, para Railway)
RUN uv sync --frozen --no-install-project

# Variables y estructura del proyecto
ENV PYTHONPATH=/app/src
COPY src/ /app/src/
COPY models/ /app/models/
COPY assets/ /app/assets/

# Instalar dependencias adicionales si las hay
RUN uv sync

# Exponer puerto interno
EXPOSE 8000

# Comando de inicio (expande $PORT correctamente)
CMD ["sh", "-c", "uv run -m src.main --host 0.0.0.0 --port $PORT"]

