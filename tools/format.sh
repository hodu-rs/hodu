#!/bin/bash

# Modern color palette
CYAN='\033[0;36m'
BRIGHT_BLUE='\033[1;34m'
BRIGHT_GREEN='\033[1;32m'
BRIGHT_YELLOW='\033[1;33m'
BRIGHT_RED='\033[1;31m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Format Rust files
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}Formatting Rust files...${NC}"
cargo fmt --all
echo -e "${BRIGHT_GREEN}✓${NC} Rust formatting complete\n"

# Format Metal files
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}Formatting Metal files...${NC}"

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null
then
    echo -e "${BRIGHT_RED}⚠${NC}  ${BRIGHT_YELLOW}Warning:${NC} clang-format is not installed"
    echo -e "${DIM}   Install with: ${MAGENTA}brew install clang-format${NC}\n"
else
    # Counter for formatted files
    count=0

    # Find and format all .metal files recursively from current directory
    while IFS= read -r -d '' file
    do
        echo -e "  ${CYAN}→${NC} ${DIM}$file${NC}"
        clang-format -i "$file"
        ((count++))
    done < <(find . -type f -name "*.metal" -print0)

    if [ $count -eq 0 ]; then
        echo -e "${DIM}  No .metal files found${NC}"
    else
        echo -e "${BRIGHT_GREEN}✓${NC} Formatted ${BOLD}${count}${NC} Metal file(s)"
    fi
fi

# Format Python files
echo -e "\n${BRIGHT_BLUE}▶${NC} ${BOLD}Formatting Python files...${NC}"

# Check if ruff is installed
if ! command -v ruff &> /dev/null
then
    echo -e "${BRIGHT_RED}⚠${NC}  ${BRIGHT_YELLOW}Warning:${NC} ruff is not installed"
    echo -e "${DIM}   Install with: ${MAGENTA}pip install ruff${NC}\n"
else
    # Counter for formatted files
    py_count=0

    # Find all .py files recursively from current directory
    while IFS= read -r -d '' file
    do
        echo -e "  ${CYAN}→${NC} ${DIM}$file${NC}"
        ruff format "$file"
        ((py_count++))
    done < <(find . -type f -name "*.py" -not -path "*/.*" -not -path "*/venv/*" -not -path "*/__pycache__/*" -not -path "*/venvs/*" -print0)

    if [ $py_count -eq 0 ]; then
        echo -e "${DIM}  No .py files found${NC}"
    else
        echo -e "${BRIGHT_GREEN}✓${NC} Formatted ${BOLD}${py_count}${NC} Python file(s)"
    fi
fi

echo -e "\n${BOLD}${BRIGHT_GREEN}All formatting complete!${NC}"
