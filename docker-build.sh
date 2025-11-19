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

# Compile and run benchmark harness if present
if [ -f ../bench/bench_core.cpp ]; then
    echo "==================================="
    echo "Building and running C++ benchmark"
    echo "==================================="
    echo ""
    # Create a directory to hold the benchmark binary
    mkdir -p bench_bin
    # Compile the benchmark using g++ with aggressive optimizations
    g++ -O3 -march=native -ffast-math -fno-math-errno -std=c++11 -I../include ../bench/bench_core.cpp ../src/hpc_series.c -o bench_bin/bench_core

    if [ $? -eq 0 ]; then
        echo "Running C++ benchmark..."
        echo ""
        ./bench_bin/bench_core || true
        echo ""
    else
        echo "❌ Benchmark compilation failed"
    fi
fi

# Run Python benchmark if Python is available
if command -v python3 >/dev/null 2>&1 && [ -f ../bench/python_benchmark.py ]; then
    echo "==================================="
    echo "Running Python benchmark"
    echo "==================================="
    echo ""
    python3 ../bench/python_benchmark.py || true
    echo ""
fi
