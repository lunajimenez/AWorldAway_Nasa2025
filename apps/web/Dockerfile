FROM node:lts-alpine AS builder

WORKDIR /app

ENV NODE_ENV="production"
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

# Variables dinámicas (vienen desde Railway o --build-arg)
ARG API_BASE_URL
ARG NUXT_PUBLIC_SITE_URL

# Exponerlas al entorno del contenedor
ENV API_BASE_URL=${API_BASE_URL}
ENV NUXT_PUBLIC_SITE_URL=${NUXT_PUBLIC_SITE_URL}

COPY . /app/

RUN pnpm i --prod

RUN npx nuxi build

FROM node:lts-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.output/ /app/.output/

EXPOSE 3000

CMD [ "node", ".output/server/index.mjs" ]
