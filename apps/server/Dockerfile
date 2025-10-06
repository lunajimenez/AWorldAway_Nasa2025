# Imagen base
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /bin/

# Configuración del entorno
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copiar archivos del proyecto
COPY pyproject.toml uv.lock /app/
COPY src/ /app/src/
COPY models/ /app/models/
COPY assets/ /app/assets/

# Instalar dependencias (sin mounts, compatible con Railway)
RUN uv sync --frozen --no-install-project

# Instalar requirements si los usas (opcional)
COPY requirements.txt /app/
RUN pip install -r requirements.txt || true

# Variables de entorno
ENV PYTHONPATH=/app/src
EXPOSE 8000

# Comando de inicio
CMD ["uv", "run", "-m", "src.main"]
