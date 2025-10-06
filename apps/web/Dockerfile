# ==============================
# Base image
# ==============================
FROM python:3.13-slim

# Permite que Python imprima logs en tiempo real
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# ==============================
# Variables de entorno dinámicas
# ==============================
# ARG = valor recibido al hacer build (Railway o local)
ARG API_BASE_URL
# ENV = define la variable dentro del contenedor
ENV API_BASE_URL=${API_BASE_URL}

# (Puedes agregar más si necesitas, por ejemplo)
# ARG NUXT_PUBLIC_SITE_URL
# ENV NUXT_PUBLIC_SITE_URL=${NUXT_PUBLIC_SITE_URL}

# ==============================
# Instalar UV (gestor ultrarrápido de Python)
# ==============================
COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /bin/

# ==============================
# Configuración base
# ==============================
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# ==============================
# Dependencias del proyecto
# ==============================
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project

# ==============================
# Código fuente
# ==============================
COPY src/ /app/src/
COPY models/ /app/models/
COPY assets/ /app/assets/

# ==============================
# Variables adicionales (si aplica)
# ==============================
ENV PYTHONPATH=/app/src

# ==============================
# Exponer puerto dinámico
# ==============================
EXPOSE 8000

# ==============================
# Comando de inicio
# ==============================
CMD ["sh", "-c", "uv run -m src.main --host 0.0.0.0 --port $PORT"]
