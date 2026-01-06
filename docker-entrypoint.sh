#!/bin/bash
# HPCSeries Core - Docker Entrypoint Script
# Builds Python extensions after volume mount

set -e

echo "========================================"
echo "HPCSeries Core Docker Environment"
echo "========================================"
echo ""

# Check if C library exists
if [ ! -f "build/libhpcs_core.a" ]; then
    echo "Building C library..."
    cmake -S . -B build
    cmake --build build -j$(nproc)
    echo "✓ C library built"
else
    echo "✓ C library found"
fi

# Build Python extensions
echo ""
echo "Building Python extensions..."
pip install -e . --no-build-isolation -v > /tmp/pip_install.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Python extensions built successfully"

    # Verify installation
    python3 -c "import hpcs; print(f'✓ HPCSeries {hpcs.__version__} ready')"
else
    echo "✗ Python extension build failed!"
    echo "Last 20 lines of build log:"
    tail -20 /tmp/pip_install.log
    exit 1
fi

echo ""
echo "========================================"
echo ""

# Execute the command passed to the container
exec "$@"
