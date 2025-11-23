#!/bin/bash

# Test all feature combinations for hodu

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
HAS_CUDA=false
HAS_METAL=false
for feature in "${ADDITIONAL_FEATURES[@]}"; do
    if [ "$feature" = "cuda" ]; then
        HAS_CUDA=true
    fi
    if [ "$feature" = "metal" ]; then
        HAS_METAL=true
    fi
done

# Check C/C++ kernel syntax
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[C/C++] Checking kernel syntax...${NC}"
if ! command -v clang &> /dev/null
then
    echo -e "${BRIGHT_RED}⚠${NC}  ${BRIGHT_YELLOW}Warning:${NC} clang is not available"
    echo -e "${DIM}   Skipping C/C++ syntax check${NC}\n"
else
    # Counter for checked files
    c_count=0
    c_failed=0

    # Find and check all C/C++ files recursively
    while IFS= read -r -d '' file
    do
        filename=$(basename "$file")
        kernel_dir=$(dirname "$file")

        # Check syntax
        if clang -I "$kernel_dir" -std=c11 -Wall -Wextra -fsyntax-only "$file" 2>&1 | grep -q "error:"; then
            echo -e "  ${BRIGHT_RED}✗${NC} ${filename}"
            ((c_failed++))
        else
            echo -e "  ${CYAN}→${NC} ${DIM}${filename}${NC}"
        fi
        ((c_count++))
    done < <(find . -type f \( -name "*.c" -o -name "*.cpp" \) -not -path "*/target/*" -not -path "*/.*" -not -path "*/libs/*" -print0)

    if [ $c_count -eq 0 ]; then
        echo -e "${DIM}  No C/C++ files found${NC}"
    else
        if [ $c_failed -eq 0 ]; then
            echo -e "${BRIGHT_GREEN}✓${NC} Checked ${BOLD}${c_count}${NC} C/C++ file(s)"
        else
            echo -e "${BRIGHT_RED}✗${NC} ${c_failed}/${c_count} C/C++ file(s) failed"
            exit 1
        fi
    fi
fi

echo ""

# Check CUDA kernel syntax
if [ "$HAS_CUDA" = true ]; then
    echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[CUDA] Checking kernel syntax...${NC}"
    if ! command -v nvcc &> /dev/null
    then
        echo -e "${BRIGHT_RED}⚠${NC}  ${BRIGHT_YELLOW}Warning:${NC} nvcc is not available"
        echo -e "${DIM}   Skipping CUDA syntax check (requires CUDA Toolkit)${NC}\n"
    else
        # Counter for checked files
        cuda_count=0
        cuda_failed=0

        # Find and check all .cu and .cuh files recursively
        while IFS= read -r -d '' file
        do
            filename=$(basename "$file")
            kernel_dir=$(dirname "$file")

            # Check syntax
            if nvcc -I "$kernel_dir" --cuda --device-c -x cu "$file" -o /dev/null 2>&1 | grep -q "error:"; then
                echo -e "  ${BRIGHT_RED}✗${NC} ${filename}"
                ((cuda_failed++))
            else
                echo -e "  ${CYAN}→${NC} ${DIM}${filename}${NC}"
            fi
            ((cuda_count++))
        done < <(find . -type f \( -name "*.cu" -o -name "*.cuh" \) -not -path "*/target/*" -not -path "*/.*" -not -path "*/libs/*" -print0)

        if [ $cuda_count -eq 0 ]; then
            echo -e "${DIM}  No CUDA files found${NC}"
        else
            if [ $cuda_failed -eq 0 ]; then
                echo -e "${BRIGHT_GREEN}✓${NC} Checked ${BOLD}${cuda_count}${NC} CUDA file(s)"
            else
                echo -e "${BRIGHT_RED}✗${NC} ${cuda_failed}/${cuda_count} CUDA file(s) failed"
                exit 1
            fi
        fi
    fi

    echo ""
else
    echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[CUDA] Checking kernel syntax...${NC}"
    echo -e "${DIM}  Skipping CUDA check (cuda feature not enabled)${NC}\n"
fi

# Check Metal kernel syntax
if [ "$HAS_METAL" = true ]; then
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
else
    echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[Metal] Checking kernel syntax...${NC}"
    echo -e "${DIM}  Skipping Metal check (metal feature not enabled)${NC}\n"
fi

# Test Cargo feature combinations
echo -e "${BRIGHT_BLUE}▶${NC} ${BOLD}[Cargo] Testing feature combinations...${NC}"

# Data type feature presets
DTYPE_NONE=""
DTYPE_ALL="f8e5m2,f64,u16,u64,i16,i64"

# Initialize tests array
tests=()

echo -e "${DIM}Building test matrix...${NC}"

# ============================================================================
# PHASE 1: Basic configurations (no dtype features)
# ============================================================================
tests+=(
    "|[basic] no features (no-std)"
    "serde|[basic] serde (no-std)"
    "std|[basic] std only"
    "std,serde|[basic] std + serde"
)

# ============================================================================
# PHASE 2: Data type feature combinations (no-std and std)
# ============================================================================
tests+=(
    "$DTYPE_ALL|[dtype] all dtypes (no-std)"
    "std,$DTYPE_ALL|[dtype] all dtypes (std)"
    "std,serde,$DTYPE_ALL|[dtype] all dtypes + serde (std)"
)

# ============================================================================
# PHASE 3: XLA backend combinations
# ============================================================================
tests+=(
    "std,xla|[xla] xla only (std)"
    "std,serde,xla|[xla] xla + serde (std)"
    "std,serde,xla,$DTYPE_ALL|[xla] xla + serde + all dtypes (std)"
)

# ============================================================================
# PHASE 4: CUDA backend combinations
# ============================================================================
if [ "$HAS_CUDA" = true ]; then
    tests+=(
        "std,cuda|[cuda] cuda only (std)"
        "std,serde,cuda|[cuda] cuda + serde (std)"
        "std,serde,cuda,$DTYPE_ALL|[cuda] cuda + serde + all dtypes (std)"
    )

    # CUDA + XLA combinations
    tests+=(
        "std,xla,cuda|[cuda+xla] xla + cuda (std)"
        "std,serde,xla,cuda|[cuda+xla] xla + cuda + serde (std)"
        "std,serde,xla,cuda,$DTYPE_ALL|[cuda+xla] xla + cuda + all dtypes (std)"
    )
fi

# ============================================================================
# PHASE 5: Metal backend combinations (macOS only)
# ============================================================================
if [ "$HAS_METAL" = true ]; then
    tests+=(
        "std,metal|[metal] metal only (std)"
        "std,serde,metal|[metal] metal + serde (std)"
        "std,serde,metal,$DTYPE_ALL|[metal] metal + serde + all dtypes (std)"
    )

    # Metal + XLA combinations
    tests+=(
        "std,xla,metal|[metal+xla] xla + metal (std)"
        "std,serde,xla,metal|[metal+xla] xla + metal + serde (std)"
        "std,serde,xla,metal,$DTYPE_ALL|[metal+xla] all features + all dtypes (std)"
    )
fi

echo -e "${DIM}Total test cases: ${#tests[@]}${NC}"
echo ""

passed=0
total=${#tests[@]}

for test in "${tests[@]}"; do
    features="${test%|*}"
    desc="${test#*|}"

    echo -ne "  ${CYAN}→${NC} ${DIM}$desc${NC} ... "

    if [ -z "$features" ]; then
        cmd="cargo check --no-default-features"
        clippy_cmd="cargo clippy --no-default-features"
    else
        cmd="cargo check --no-default-features --features $features"
        clippy_cmd="cargo clippy --no-default-features --features $features"
    fi

    # Capture output to check for warnings
    output=$($cmd 2>&1)
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Run clippy after successful check
        clippy_output=$($clippy_cmd 2>&1)
        clippy_exit_code=$?

        if [ $clippy_exit_code -eq 0 ]; then
            # Check for any warnings (from check or clippy)
            if echo "$output" | grep -q "warning:" || echo "$clippy_output" | grep -q "warning:"; then
                echo -e "${BRIGHT_YELLOW}△${NC}"
            else
                echo -e "${BRIGHT_GREEN}✓${NC}"
            fi
            ((passed++))
        else
            echo -e "${BRIGHT_RED}✗${NC}"
        fi
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
