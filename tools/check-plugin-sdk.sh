#!/bin/bash

# Check hodu-plugin-sdk

BRIGHT_BLUE='\033[1;34m'
BRIGHT_GREEN='\033[1;32m'
BRIGHT_RED='\033[1;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[hodu-plugin-sdk] Running cargo check...${NC}"
if ! cargo check -p hodu-plugin-sdk 2>&1; then
    echo -e "${BRIGHT_RED}✗${NC} cargo check failed"
    exit 1
fi

echo ""
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[hodu-plugin-sdk] Running cargo clippy...${NC}"
if ! cargo clippy -p hodu-plugin-sdk 2>&1; then
    echo -e "${BRIGHT_RED}✗${NC} cargo clippy failed"
    exit 1
fi

echo ""
echo -e "${BOLD}${BRIGHT_GREEN}All checks passed!${NC}"
