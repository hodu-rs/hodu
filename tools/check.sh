#!/bin/bash

# Test all feature combinations for hodu
set -e

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

# Parse command line arguments
ADDITIONAL_FEATURES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--features)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^- ]]; do
                ADDITIONAL_FEATURES+=("$1")
                shift
            done
            ;;
        *)
            echo -e "${BRIGHT_RED}✗${NC} Unknown option: $1"
            echo "Usage: $0 [-f|--features feature1 feature2 ...]"
            exit 1
            ;;
    esac
done

# Check if metal is in additional features
HAS_METAL=false
for feature in "${ADDITIONAL_FEATURES[@]}"; do
    if [ "$feature" = "metal" ]; then
        HAS_METAL=true
        break
    fi
done

# Check Metal kernel syntax
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[Metal] Checking kernel syntax...${NC}"
if ! command -v xcrun &> /dev/null
then
    echo -e "${BRIGHT_RED}⚠${NC}  ${BRIGHT_YELLOW}Warning:${NC} xcrun is not available"
    echo -e "${DIM}   Skipping Metal syntax check (macOS only)${NC}\n"
else
    # Counter for checked files
    metal_count=0
    metal_failed=0

    # Find and check all .metal files recursively from current directory
    while IFS= read -r -d '' file
    do
        filename=$(basename "$file")
        kernel_dir=$(dirname "$file")

        # Check syntax
        if xcrun -sdk macosx metal -I "$kernel_dir" -fsyntax-only "$file" 2>&1 | grep -q "error:"; then
            echo -e "  ${BRIGHT_RED}✗${NC} ${filename}"
            ((metal_failed++))
        else
            echo -e "  ${CYAN}→${NC} ${DIM}${filename}${NC}"
        fi
        ((metal_count++))
    done < <(find . -type f -name "*.metal" -print0)

    if [ $metal_count -eq 0 ]; then
        echo -e "${DIM}  No .metal files found${NC}"
    else
        if [ $metal_failed -eq 0 ]; then
            echo -e "${BRIGHT_GREEN}✓${NC} Checked ${BOLD}${metal_count}${NC} Metal file(s)"
        else
            echo -e "${BRIGHT_RED}✗${NC} ${metal_failed}/${metal_count} Metal file(s) failed"
            exit 1
        fi
    fi
fi

echo ""

# Test Cargo feature combinations
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[Cargo] Testing feature combinations...${NC}"

# Base test cases: "features|description"
tests=(
    "|no features (no-std)"
    "serde|serde only (no-std)"
    "std|no features (std)"
    "std,serde|serde only (std)"
    "std,xla|xla only (std)"
    "std,serde,xla|serde, xla (std)"
)

# Add metal features only to std combinations
if [ "$HAS_METAL" = true ]; then
    # Add metal to std-based combinations
    tests+=(
        "std,metal|std + metal"
        "std,serde,metal|std + serde + metal"
        "std,xla,metal|std + xla + metal"
        "std,serde,xla,metal|std + serde + xla + metal"
    )
fi

passed=0
total=${#tests[@]}

for test in "${tests[@]}"; do
    features="${test%|*}"
    desc="${test#*|}"

    echo -ne "  ${CYAN}→${NC} ${DIM}$desc${NC} ... "

    if [ -z "$features" ]; then
        cmd="cargo check --no-default-features"
    else
        cmd="cargo check --no-default-features --features $features"
    fi

    if $cmd &>/dev/null; then
        echo -e "${BRIGHT_GREEN}✓${NC}"
        ((passed++))
    else
        echo -e "${BRIGHT_RED}✗${NC}"
    fi
done

if [ $passed -eq $total ]; then
    echo -e "${BRIGHT_GREEN}✓${NC} All ${BOLD}${total}${NC} feature combination(s) passed"
else
    echo -e "${BRIGHT_RED}✗${NC} ${passed}/${total} feature combination(s) passed"
fi

echo ""

if [ $passed -eq $total ]; then
    echo -e "${BOLD}${BRIGHT_GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${BOLD}${BRIGHT_RED}Some checks failed!${NC}"
    exit 1
fi
