#!/bin/bash

# Test all feature combinations for hodu_core

set -e

echo "==> Testing hodu feature combinations..."

# Test cases: "features|description"
tests=(
    "|no features (no-std)"
    "serde|serde only (no-std)"
    "std|no features (std)"
    "std,serde|serde only (std)"
)

passed=0
total=${#tests[@]}

for test in "${tests[@]}"; do
    features="${test%|*}"
    desc="${test#*|}"
    
    echo -n "Testing $desc... "
    
    if [ -z "$features" ]; then
        cmd="cargo check --no-default-features"
    else
        cmd="cargo check --no-default-features --features $features"
    fi
    
    if $cmd &>/dev/null; then
        echo "[ OK ]"
        ((passed++))
    else
        echo "[FAIL]"
    fi
done

echo
echo "Results: $passed/$total passed"

if [ $passed -eq $total ]; then
    echo "*** All tests passed! ***"
    exit 0
else
    echo "*** Some tests failed ***"
    exit 1
fi
