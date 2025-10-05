#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'


echo -e "${GREEN}[ INFO ]${NC} init"

SOURCE_DIR="packages/analytics/.output/model/"
TARGET_DIR="apps/server/models/"

mkdir -p "$TARGET_DIR"
echo -e "${GREEN}[ INFO ]${NC} target directory: ${TARGET_DIR}"

cp -r "$SOURCE_DIR"/* "$TARGET_DIR"

echo -e "${YELLOW}[ INFO ]${NC} path: $TARGET_DIR"
ls -lh "$TARGET_DIR"