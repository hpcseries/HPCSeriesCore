#!/bin/bash
# test_compiler_flags.sh - Verification Tests for Architecture-Aware Compilation
#
# Tests that:
# 1. SAFE profile builds without -ffast-math
# 2. FAST profile builds with -ffast-math
# 3. Architecture detection works correctly
#
# Usage:
#   ./test_compiler_flags.sh

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================================"
echo "HPCSeries Compiler Flags Verification Tests"
echo "============================================================================"
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    local test_name=$1
    local result=$2
    local details=$3

    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        if [ -n "$details" ]; then
            echo "  $details"
        fi
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        if [ -n "$details" ]; then
            echo "  $details"
        fi
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
}

# Test 1: Architecture Detection
echo -e "${BLUE}Test 1: Architecture Detection${NC}"
echo "-------------------------------------------------------------------"

if [ -f "$SCRIPT_DIR/detect_architecture.sh" ]; then
    DETECTED_ARCH=$("$SCRIPT_DIR/detect_architecture.sh" 2>/dev/null || echo "")

    if [ -n "$DETECTED_ARCH" ]; then
        print_result "Architecture detection script" "PASS" "Detected: $DETECTED_ARCH"
    else
        print_result "Architecture detection script" "FAIL" "Failed to detect architecture"
    fi
else
    print_result "Architecture detection script" "FAIL" "Script not found: $SCRIPT_DIR/detect_architecture.sh"
fi

# Test 2: SAFE Profile Build (Default)
echo -e "${BLUE}Test 2: SAFE Profile Build (Default)${NC}"
echo "-------------------------------------------------------------------"

BUILD_DIR_SAFE="$PROJECT_ROOT/build_test_safe"
rm -rf "$BUILD_DIR_SAFE"
mkdir -p "$BUILD_DIR_SAFE"

cd "$BUILD_DIR_SAFE"
echo "Configuring with SAFE profile (default)..."

if cmake .. -DCMAKE_BUILD_TYPE=Release > cmake_safe.log 2>&1; then
    # Check CMake output for architecture detection
    if grep -q "HPCSeries: Detected architecture" cmake_safe.log; then
        ARCH_LINE=$(grep "HPCSeries: Detected architecture" cmake_safe.log)
        print_result "CMake architecture detection" "PASS" "$ARCH_LINE"
    else
        print_result "CMake architecture detection" "FAIL" "Architecture not detected in CMake output"
    fi

    # Check for SAFE profile in output
    if grep -q "HPCSeries: Using SAFE profile" cmake_safe.log; then
        print_result "SAFE profile selection" "PASS" "SAFE profile activated by default"
    else
        print_result "SAFE profile selection" "FAIL" "SAFE profile not detected"
    fi

    # Try to build a simple target to verify flags
    echo "Building with SAFE profile..."
    if make VERBOSE=1 > build_safe.log 2>&1; then
        # Check that -ffast-math is NOT present
        if grep -q "\-ffast-math" build_safe.log; then
            print_result "SAFE profile flags" "FAIL" "-ffast-math found in SAFE build (should not be present)"
        else
            print_result "SAFE profile flags" "PASS" "No -ffast-math in SAFE build (correct)"
        fi

        # Check that optimization flags are present
        if grep -q "\-O3" build_safe.log; then
            print_result "Optimization flags" "PASS" "-O3 flag present"
        else
            print_result "Optimization flags" "FAIL" "-O3 flag not found"
        fi

        # Check for architecture-specific flags
        if grep -qE "\-(march|mcpu)=native" build_safe.log; then
            print_result "Architecture flags" "PASS" "Architecture-specific flags present"
        else
            print_result "Architecture flags" "FAIL" "Architecture-specific flags not found"
        fi
    else
        print_result "SAFE build compilation" "FAIL" "Build failed"
    fi
else
    print_result "SAFE profile CMake configuration" "FAIL" "CMake failed"
fi

cd "$PROJECT_ROOT"

# Test 3: FAST Profile Build
echo -e "${BLUE}Test 3: FAST Profile Build${NC}"
echo "-------------------------------------------------------------------"

BUILD_DIR_FAST="$PROJECT_ROOT/build_test_fast"
rm -rf "$BUILD_DIR_FAST"
mkdir -p "$BUILD_DIR_FAST"

cd "$BUILD_DIR_FAST"
echo "Configuring with FAST profile..."

export HPCS_PROFILE=FAST

if cmake .. -DCMAKE_BUILD_TYPE=Release > cmake_fast.log 2>&1; then
    # Check for FAST profile in output
    if grep -q "HPCSeries: Using FAST profile" cmake_fast.log; then
        print_result "FAST profile selection" "PASS" "FAST profile activated via environment variable"
    else
        print_result "FAST profile selection" "FAIL" "FAST profile not detected"
    fi

    # Try to build to verify flags
    echo "Building with FAST profile..."
    if make VERBOSE=1 > build_fast.log 2>&1; then
        # Check that -ffast-math IS present
        if grep -q "\-ffast-math" build_fast.log; then
            print_result "FAST profile flags" "PASS" "-ffast-math present in FAST build (correct)"
        else
            print_result "FAST profile flags" "FAIL" "-ffast-math NOT found in FAST build (should be present)"
        fi

        # Check for other fast-math related flags
        if grep -q "\-fno-math-errno" build_fast.log; then
            print_result "FAST math optimizations" "PASS" "Additional fast-math flags present"
        else
            print_result "FAST math optimizations" "FAIL" "Additional fast-math flags not found"
        fi
    else
        print_result "FAST build compilation" "FAIL" "Build failed"
    fi
else
    print_result "FAST profile CMake configuration" "FAIL" "CMake failed"
fi

unset HPCS_PROFILE
cd "$PROJECT_ROOT"

# Test 4: AWS Instance Detection (if on EC2)
echo -e "${BLUE}Test 4: AWS Instance Detection${NC}"
echo "-------------------------------------------------------------------"

if [ -f "$SCRIPT_DIR/detect_aws_instance.sh" ]; then
    AWS_INFO=$("$SCRIPT_DIR/detect_aws_instance.sh" 2>/dev/null || echo "")

    if echo "$AWS_INFO" | grep -q "not-ec2=true"; then
        print_result "AWS instance detection" "PASS" "Not on EC2 (graceful handling)"
    elif echo "$AWS_INFO" | grep -q "instance-id="; then
        INSTANCE_TYPE=$(echo "$AWS_INFO" | grep -o "instance-type=[^,]*" | cut -d= -f2)
        print_result "AWS instance detection" "PASS" "Detected EC2 instance: $INSTANCE_TYPE"
    else
        print_result "AWS instance detection" "FAIL" "Unexpected output from detection script"
    fi
else
    print_result "AWS instance detection script" "FAIL" "Script not found: $SCRIPT_DIR/detect_aws_instance.sh"
fi

# Summary
echo "============================================================================"
echo "Test Summary"
echo "============================================================================"
echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
echo -e "${RED}Failed:${NC} $TESTS_FAILED"
echo ""

# Cleanup test builds
echo "Cleaning up test build directories..."
rm -rf "$BUILD_DIR_SAFE" "$BUILD_DIR_FAST"
echo "Done."
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC} ✓"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC} Please review the output above."
    exit 1
fi
