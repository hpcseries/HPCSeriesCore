#!/bin/bash
# HPCSeries Core v0.7.0 - PyPI Quick Publication Script
# ========================================================
#
# This script automates the PyPI publication workflow.
#
# Prerequisites:
#   - Virtual environment activated: source test-env/bin/activate
#   - Native library built: build/libhpcs_core.a
#   - PyPI account with API token configured in ~/.pypirc
#
# Usage:
#   source test-env/bin/activate  # Activate virtual environment first
#   ./.github/PYPI_QUICK_START.sh [test|prod]
#
# Options:
#   test  - Upload to TestPyPI (recommended first)
#   prod  - Upload to production PyPI (IRREVERSIBLE)
#
# See .github/PYPI_PUBLICATION_GUIDE.md for detailed instructions.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Determine target repository
TARGET="${1:-test}"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  HPCSeries Core v0.7.0 - PyPI Publication Script                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Validate target
if [ "$TARGET" != "test" ] && [ "$TARGET" != "prod" ]; then
    echo -e "${RED}ERROR: Invalid target '$TARGET'. Use 'test' or 'prod'.${NC}"
    exit 1
fi

# Warning for production
if [ "$TARGET" = "prod" ]; then
    echo -e "${RED}⚠️  WARNING: Publishing to PRODUCTION PyPI${NC}"
    echo -e "${RED}   This action is IRREVERSIBLE. You cannot delete or replace v0.7.0.${NC}"
    echo ""
    read -p "Are you sure? Type 'YES' to confirm: " CONFIRM
    if [ "$CONFIRM" != "YES" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Step 1: Check prerequisites
echo -e "${BLUE}[1/8] Checking prerequisites...${NC}"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not detected${NC}"
    echo "It's recommended to activate the virtual environment first:"
    echo "  source test-env/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        echo "Aborted. Please activate virtual environment and try again."
        exit 0
    fi
else
    echo -e "${GREEN}✓ Virtual environment active: $VIRTUAL_ENV${NC}"
fi

if [ ! -f "build/libhpcs_core.a" ]; then
    echo -e "${RED}ERROR: Native library not found: build/libhpcs_core.a${NC}"
    echo "Build it first:"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build . -j\$(nproc)"
    echo "  cd .."
    exit 1
fi
echo -e "${GREEN}✓ Native library found${NC}"

# Check Python dependencies
python3 -c "import build" 2>/dev/null || {
    echo -e "${YELLOW}Installing build dependencies...${NC}"
    pip3 install --upgrade build twine
}
echo -e "${GREEN}✓ Build tools available${NC}"

# Step 2: Clean previous builds
echo -e "${BLUE}[2/8] Cleaning previous builds...${NC}"
rm -rf dist/ build/lib.* build/temp.* *.egg-info python/hpcs/*.c python/hpcs/*.so
echo -e "${GREEN}✓ Cleaned${NC}"

# Step 3: Build package
echo -e "${BLUE}[3/8] Building Python package...${NC}"
# Build wheel only (sdist would require users to have cmake/gfortran)
# For source distribution, users need build environment anyway
python3 -m build --wheel
python3 -m build --sdist
echo -e "${GREEN}✓ Package built${NC}"

# Step 4: Verify package contents
echo -e "${BLUE}[4/8] Verifying package contents...${NC}"
ls -lh dist/
echo ""
twine check dist/*
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Package verification failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Package verification passed${NC}"

# Step 5: Upload to repository
echo -e "${BLUE}[5/8] Uploading to ${TARGET}PyPI...${NC}"
if [ "$TARGET" = "test" ]; then
    twine upload --repository testpypi dist/*
    INSTALL_URL="https://test.pypi.org/simple/"
    PKG_URL="https://test.pypi.org/project/hpcs/0.7.0/"
else
    twine upload dist/*
    INSTALL_URL="https://pypi.org/simple/"
    PKG_URL="https://pypi.org/project/hpcs/0.7.0/"
fi
echo -e "${GREEN}✓ Upload complete${NC}"

# Step 6: Verify installation
echo -e "${BLUE}[6/8] Verifying installation...${NC}"
echo "Package should be available at: ${PKG_URL}"
echo ""
echo "To test installation:"
if [ "$TARGET" = "test" ]; then
    echo "  pip install --index-url ${INSTALL_URL} --no-deps hpcs==0.7.0"
else
    echo "  pip install hpcs==0.7.0"
fi
echo ""
echo "To verify:"
echo "  python3 -c 'import hpcs; print(hpcs.__version__)'"

# Step 7: Next steps
echo -e "${BLUE}[7/8] Post-upload verification${NC}"
echo "Waiting 30 seconds for PyPI to process the package..."
sleep 30

# Step 8: Final instructions
echo -e "${BLUE}[8/8] Next steps${NC}"
if [ "$TARGET" = "test" ]; then
    echo -e "${YELLOW}You uploaded to TestPyPI. If everything looks good:${NC}"
    echo "  1. Test installation from TestPyPI"
    echo "  2. Run: $0 prod"
    echo ""
    echo -e "${YELLOW}View package at: ${PKG_URL}${NC}"
else
    echo -e "${GREEN}✅ Package published to PyPI!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify package page: ${PKG_URL}"
    echo "  2. Test installation: pip install hpcs==0.7.0"
    echo "  3. Create GitHub release (use .github/RELEASE_TEMPLATE.md)"
    echo "  4. Update README with PyPI badge"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Publication workflow complete!                                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
