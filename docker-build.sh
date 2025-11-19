#!/bin/bash
# Build script for use inside Docker container

set -e  # Exit on error

echo "==================================="
echo "HPC Series Core - Docker Build"
echo "==================================="

# Clean previous build
echo "Cleaning previous build..."
rm -rf build
mkdir build
cd build

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake ..

# Build
echo ""
echo "Building..."
make -j$(nproc)

# Run tests
echo ""
echo "==================================="
echo "Running Tests"
echo "==================================="
echo ""

if [ -f ./test_core_c ]; then
    echo "Running C tests..."
    ./test_core_c
    TEST_STATUS=$?

    if [ $TEST_STATUS -eq 0 ]; then
        echo ""
        echo "✅ All tests passed!"
    else
        echo ""
        echo "❌ Tests failed with status $TEST_STATUS"
        exit 1
    fi
else
    echo "⚠️  Test executable not found"
fi

echo ""
echo "==================================="
echo "Build Complete!"
echo "==================================="
echo ""
echo "Library: build/libhpcs_core.a"
echo "Test:    build/test_core_c"
echo ""
