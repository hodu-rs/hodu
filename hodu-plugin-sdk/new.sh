#!/bin/sh
set -e

SDK_VERSION="0.1.0"
PLUGIN_VERSION="0.1.0"
TEMPLATE_URL="https://raw.githubusercontent.com/daminstudio/hodu/main/hodu-plugin-sdk/template"

# Colors
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
DIM='\033[2m'
RESET='\033[0m'

echo ""
echo "${CYAN}◆ Hodu Plugin SDK${RESET}"
echo ""

# Plugin name
printf "${GREEN}?${RESET} Plugin name: "
read NAME

if [ -z "$NAME" ]; then
    echo "${RED}✗${RESET} Plugin name cannot be empty"
    exit 1
fi

if [ -d "$NAME" ]; then
    echo "${RED}✗${RESET} Directory '$NAME' already exists"
    exit 1
fi

# Plugin type
echo ""
echo "${GREEN}?${RESET} Plugin type:"
echo "  ${CYAN}1)${RESET} backend        ${DIM}- Execute/compile models on devices${RESET}"
echo "  ${CYAN}2)${RESET} model_format   ${DIM}- Load/save model files (e.g., ONNX)${RESET}"
echo "  ${CYAN}3)${RESET} tensor_format  ${DIM}- Load/save tensor files (e.g., NPY)${RESET}"
echo ""
printf "  Select ${DIM}[1-3]${RESET}: "
read TYPE_NUM

case "$TYPE_NUM" in
    1) TYPE="backend" ;;
    2) TYPE="model_format" ;;
    3) TYPE="tensor_format" ;;
    *)
        echo "${RED}✗${RESET} Invalid selection"
        exit 1
        ;;
esac

echo ""
echo "${DIM}Creating $TYPE plugin...${RESET}"

mkdir -p "$NAME/src"

# Download and process templates
curl -fsSL "$TEMPLATE_URL/Cargo.toml.template" | \
    sed "s/{{NAME}}/$NAME/g" | \
    sed "s/{{SDK_VERSION}}/$SDK_VERSION/g" > "$NAME/Cargo.toml"

curl -fsSL "$TEMPLATE_URL/manifest.$TYPE.json" | \
    sed "s/{{NAME}}/$NAME/g" | \
    sed "s/{{PLUGIN_VERSION}}/$PLUGIN_VERSION/g" > "$NAME/manifest.json"

curl -fsSL "$TEMPLATE_URL/src/main.$TYPE.rs" | \
    sed "s/{{NAME}}/$NAME/g" > "$NAME/src/main.rs"

echo ""
echo "${GREEN}✓${RESET} Created ${CYAN}$NAME${RESET}/"
echo ""
echo "  ${DIM}Next steps:${RESET}"
echo "  ${YELLOW}cd${RESET} $NAME"
echo "  ${DIM}# Edit manifest.json and src/main.rs${RESET}"
echo "  ${YELLOW}hodu plugin install${RESET} --path ."
echo ""
