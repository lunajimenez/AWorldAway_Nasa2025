#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'


echo -e "${GREEN}[ INFO ]${NC} init"

SOURCE_DIR="packages/analytics/.output"
TARGET_DIR="apps/server/models"

mkdir -p "$TARGET_DIR"
echo -e "${GREEN}[ INFO ]${NC} target directory: ${TARGET_DIR}"

FILES=("model.joblib" "model_config.json")
COPIED=0

for file in "${FILES[@]}"; do
    SOURCE_FILE=$SOURCE_DIR/$file
    TARGET_FILE=$TARGET_DIR/$file

    if [ -f "$SOURCE_FILE" ]; then
        cp "$SOURCE_FILE" "$TARGET_FILE"
        echo -e "${GREEN}[ OK ]${NC} file: $file copied"
        ((COPIED++))
    else
        echo -e "${RED}[ ERROR ]${NC} $file not found in $SOURCE_DIR"
    fi
done

echo -e "${GREEN}[ SUCCESS ]${NC} $COPIED file(s) copied succesfully"
echo -e "${YELLOW}[ INFO ]${NC} path: $TARGET_DIR"