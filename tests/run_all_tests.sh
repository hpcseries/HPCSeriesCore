#!/bin/bash
# HPCSeries Core v0.7 - Master Test Runner
# =========================================
#
# Runs all tests: Python, C, and Fortran

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
PYTHON_PASSED=0
C_PASSED=0
FORTRAN_PASSED=0
TOTAL_FAILED=0

echo "========================================"
echo "HPCSeries Core v0.7 - Full Test Suite"
echo "========================================"
echo ""

# Check if library is built
if [ ! -f "../build/libhpcs_core.a" ]; then
    echo -e "${YELLOW}Warning: libhpcs_core.a not found${NC}"
    echo "Building C library first..."
    cd ..
    cmake -S . -B build
    cmake --build build -j$(nproc)
    cd tests
    echo ""
fi

#=============================================
# Python Tests
#=============================================
echo "----------------------------------------"
echo "1. Running Python Tests"
echo "----------------------------------------"

if command -v pytest &> /dev/null; then
    cd python
    if pytest -v --tb=short 2>&1; then
        PYTHON_PASSED=1
        echo -e "${GREEN}✓ Python tests PASSED${NC}"
    else
        echo -e "${RED}✗ Python tests FAILED${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
    cd ..
else
    echo -e "${YELLOW}⚠ pytest not found, skipping Python tests${NC}"
    echo "Install with: pip install pytest"
fi

echo ""

#=============================================
# C Tests
#=============================================
echo "----------------------------------------"
echo "2. Running C Tests"
echo "----------------------------------------"

cd c
if make run 2>&1; then
    C_PASSED=1
    echo -e "${GREEN}✓ C tests PASSED${NC}"
else
    echo -e "${RED}✗ C tests FAILED${NC}"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
fi
cd ..

echo ""

#=============================================
# Fortran Tests
#=============================================
echo "----------------------------------------"
echo "3. Running Fortran Tests"
echo "----------------------------------------"

if command -v gfortran &> /dev/null; then
    cd fortran
    if make run 2>&1; then
        FORTRAN_PASSED=1
        echo -e "${GREEN}✓ Fortran tests PASSED${NC}"
    else
        echo -e "${RED}✗ Fortran tests FAILED${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
    cd ..
else
    echo -e "${YELLOW}⚠ gfortran not found, skipping Fortran tests${NC}"
    echo "Install with: sudo apt-get install gfortran"
fi

echo ""

#=============================================
# Summary
#=============================================
echo "========================================"
echo "Test Suite Summary"
echo "========================================"

TOTAL_PASSED=$((PYTHON_PASSED + C_PASSED + FORTRAN_PASSED))

echo "Python:   $( [ $PYTHON_PASSED -eq 1 ] && echo -e '${GREEN}✓ PASSED${NC}' || echo -e '${RED}✗ FAILED${NC}' )"
echo "C:        $( [ $C_PASSED -eq 1 ] && echo -e '${GREEN}✓ PASSED${NC}' || echo -e '${RED}✗ FAILED${NC}' )"
echo "Fortran:  $( [ $FORTRAN_PASSED -eq 1 ] && echo -e '${GREEN}✓ PASSED${NC}' || echo -e '${RED}✗ FAILED${NC}' )"
echo ""
echo "Total: $TOTAL_PASSED/3 test suites passed"

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================"
    echo "✓ ALL TESTS PASSED"
    echo -e "========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================"
    echo "✗ $TOTAL_FAILED TEST SUITE(S) FAILED"
    echo -e "========================================${NC}"
    exit 1
fi
