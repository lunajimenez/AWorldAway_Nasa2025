# ==============================
# Build stage
# ==============================
FROM node:lts-alpine AS builder

WORKDIR /app

# Configuración general del entorno
ENV NODE_ENV=production
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"

# Habilitar Corepack para manejar pnpm
RUN corepack enable

# ==============================
# Variables dinámicas
# ==============================
# Recibidas desde Railway o el build local
ARG API_BASE_URL
ARG NUXT_PUBLIC_SITE_URL

# Disponibles dentro del contenedor en runtime
ENV API_BASE_URL=${API_BASE_URL}
ENV NUXT_PUBLIC_SITE_URL=${NUXT_PUBLIC_SITE_URL}

# ==============================
# Dependencias y build
# ==============================
COPY . .

# Instala solo dependencias de producción
RUN pnpm install --frozen-lockfile --prod

# Compila la app Nuxt
RUN npx nuxi build

# ==============================
# Runtime stage
# ==============================
FROM node:lts-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production

# Copiamos el resultado compilado
COPY --from=builder /app/.output /app/.output

# Exponemos el puerto dinámico asignado por Railway
EXPOSE 3000

# Comando de arranque
CMD ["node", ".output/server/index.mjs"]
