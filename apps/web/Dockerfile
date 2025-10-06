FROM node:lts-alpine AS builder

WORKDIR /app

ENV NODE_ENV="production"
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
ENV API_BASE_URL="https://aworldaway-api-production.up.railway.app"
RUN corepack enable

COPY . /app/

RUN pnpm i --prod

RUN npx nuxi build

FROM node:lts-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.output/ /app/.output/

EXPOSE 3000

CMD [ "node", ".output/server/index.mjs" ]