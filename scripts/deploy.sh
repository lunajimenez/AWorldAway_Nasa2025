#!/bin/bash

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}[ INFO ]${NC} init"

echo -e "${GREEN}[ INFO ] building images"
docker compose build

echo -e "${GREEN}[ INFO ] initializing services"
docker compose up -d