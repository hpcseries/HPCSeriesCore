# PyPI Publication Guide - HPCSeries Core v0.7.0

## Overview

This guide covers publishing the `hpcs` Python package (v0.7.0) to PyPI (Python Package Index).

**Package Name:** `hpcs`
**Version:** `0.7.0`
**PyPI URL:** https://pypi.org/project/hpcs/

---

## Prerequisites

### 1. PyPI Account Setup

```bash
# Create accounts (if you don't have them)
# - PyPI: https://pypi.org/account/register/
# - TestPyPI: https://test.pypi.org/account/register/

# Create API tokens (recommended over passwords)
# - PyPI: https://pypi.org/manage/account/token/
# - TestPyPI: https://test.pypi.org/manage/account/token/
```

### 2. Install Build Tools

```bash
pip install --upgrade build twine
```

### 3. Configure PyPI Credentials

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...  # Your PyPI API token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...  # Your TestPyPI API token
```

**Security:** Ensure `~/.pypirc` has restrictive permissions:
```bash
chmod 600 ~/.pypirc
```

---

## Pre-Publication Checklist

### Version Consistency

- [ ] `pyproject.toml:12` - `version = "0.7.0"`
- [ ] `CMakeLists.txt:4` - `VERSION 0.7.0`
- [ ] `CHANGELOG.md:8` - `[0.7.0] - 2025-12-17`
- [ ] `python/hpcs/__init__.py` - `__version__ = "0.7.0"`

### Package Metadata (pyproject.toml)

- [ ] Package name: `hpcs`
- [ ] Version: `0.7.0`
- [ ] Description accurate
- [ ] README.md specified
- [ ] License: MIT
- [ ] Python version: `>=3.8`
- [ ] Dependencies: `numpy>=1.20`
- [ ] URLs point to `hpcseries/HPCSeriesCore`
- [ ] Classifiers updated

### Required Files

- [ ] `README.md` - Will be used as PyPI long description
- [ ] `LICENSE` - Must be included
- [ ] `CHANGELOG.md` - Version history
- [ ] `pyproject.toml` - Package configuration
- [ ] `setup.py` - Build configuration
- [ ] `MANIFEST.in` - Source distribution files
- [ ] `python/hpcs/__init__.py` - Package entry point
- [ ] `python/hpcs/*.pyx` - Cython extensions

### Build Prerequisites

**Note:** The `hpcs` package requires pre-building the C/Fortran library:

```bash
# Build native library first
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cd ..
```

This creates `build/libhpcs_core.a` which is required by setup.py.

---

## Publication Steps

### Step 1: Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/lib.* build/temp.* dist/ *.egg-info
rm -f python/hpcs/*.c python/hpcs/*.so
```

### Step 2: Build Native Library

```bash
# Build C/Fortran library
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cd ..

# Verify library exists
ls -lh build/libhpcs_core.a
```

### Step 3: Build Python Package

```bash
# Build source distribution (sdist) and wheel (bdist_wheel)
python -m build

# This creates:
# - dist/hpcs-0.7.0.tar.gz (source distribution)
# - dist/hpcs-0.7.0-*.whl (wheel - platform-specific)
```

### Step 4: Verify Package Contents

```bash
# Check what's in the source distribution
tar -tzf dist/hpcs-0.7.0.tar.gz | head -20

# Check wheel contents
unzip -l dist/hpcs-0.7.0-*.whl | head -20

# Verify package metadata
twine check dist/*
```

Expected output:
```
Checking dist/hpcs-0.7.0.tar.gz: PASSED
Checking dist/hpcs-0.7.0-*.whl: PASSED
```

### Step 5: Test on TestPyPI (STRONGLY RECOMMENDED)

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --no-deps hpcs==0.7.0

# Verify installation
python -c "import hpcs; print(hpcs.__version__)"
```

If successful:
```
0.7.0
```

**Important:** TestPyPI is a separate instance of PyPI for testing. It's **strongly recommended** to test here first.

### Step 6: Upload to Production PyPI

**⚠️ WARNING: This action is IRREVERSIBLE. You cannot delete or replace a published version.**

```bash
# Upload to production PyPI
twine upload dist/*

# You'll see:
# Uploading distributions to https://upload.pypi.org/legacy/
# Uploading hpcs-0.7.0.tar.gz
# Uploading hpcs-0.7.0-*.whl
#
# View at:
# https://pypi.org/project/hpcs/0.7.0/
```

### Step 7: Verify Production Installation

```bash
# Install from PyPI
pip install hpcs==0.7.0

# Verify installation
python -c "import hpcs; print(hpcs.__version__); print(hpcs.simd_info())"
```

---

## Post-Publication Checklist

### Immediate Verification

- [ ] Package appears on PyPI: https://pypi.org/project/hpcs/
- [ ] Version shows as `0.7.0`
- [ ] README renders correctly on PyPI page
- [ ] Install works: `pip install hpcs`
- [ ] Import works: `import hpcs`
- [ ] Version correct: `hpcs.__version__ == "0.7.0"`

### Documentation Updates

- [ ] Update README installation instructions
- [ ] Add PyPI badge to README
- [ ] Update documentation with PyPI installation

### GitHub Release

- [ ] Mention PyPI publication in GitHub release notes
- [ ] Link to PyPI package page

### Social/Communication (Optional)

- [ ] Announce on GitHub Discussions
- [ ] Update project website (if any)
- [ ] Social media announcement

---

## Platform-Specific Notes

### Linux

- **Recommended**: Build on `manylinux` compatible systems
- **Dependencies**: gfortran, gcc, cmake, ninja-build
- **Wheels**: Platform-specific due to compiled extensions

### macOS

- **Compiler**: Requires gfortran (install via Homebrew)
- **OpenMP**: May need `libomp` from Homebrew
- **Wheels**: Separate wheels for Intel and Apple Silicon

### Windows

- **Status**: Not officially supported in v0.7.0
- **Future**: May support via MSVC or MSYS2/MinGW

---

## Troubleshooting

### Issue: `libhpcs_core.a` not found

**Solution:**
```bash
# Build the native library first
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cd ..
```

### Issue: Cython compilation errors

**Solution:**
```bash
# Ensure Cython is installed
pip install --upgrade Cython>=3.0

# Clean and rebuild
rm -rf build/lib.* build/temp.*
python -m build
```

### Issue: Missing dependencies during build

**Solution:**
```bash
# Install build dependencies
pip install numpy>=1.20 Cython>=3.0 setuptools>=65.0 wheel build
```

### Issue: Upload fails with authentication error

**Solution:**
```bash
# Ensure ~/.pypirc is correctly configured
# Or use environment variables:
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmcC...

twine upload dist/*
```

### Issue: Package already exists error

**Solution:**
- PyPI doesn't allow re-uploading the same version
- Increment version (e.g., `0.7.1`) and rebuild
- For testing, use TestPyPI first

---

## Version Management

### For Bug Fixes (Patch Release)

```bash
# Increment to 0.7.1
# Update: pyproject.toml, CMakeLists.txt, CHANGELOG.md
python -m build
twine upload dist/*
```

### For New Features (Minor Release)

```bash
# Increment to 0.8.0
# Update all version files
# Full release process (git tag, GitHub release, PyPI)
```

### For Breaking Changes (Major Release)

```bash
# Increment to 1.0.0
# Document migration guide
# Full release process
```

---

## Security Best Practices

1. **API Tokens**: Use API tokens, not passwords
2. **Permissions**: Restrict `~/.pypirc` permissions (chmod 600)
3. **2FA**: Enable two-factor authentication on PyPI account
4. **Trusted Publishers**: Consider setting up GitHub Actions trusted publishing
5. **Signing**: Consider signing packages with GPG

---

## Maintenance

### Update Frequency

- **Patch releases** (0.7.x): Bug fixes, no new features
- **Minor releases** (0.x.0): New features, backward compatible
- **Major releases** (x.0.0): Breaking changes

### Support Policy

- **Current release** (v0.7.0): Full support
- **Previous minor** (v0.6.x): Security fixes only
- **Older versions**: No support

---

## References

- **PyPI Documentation**: https://packaging.python.org/
- **Twine**: https://twine.readthedocs.io/
- **Build**: https://pypa-build.readthedocs.io/
- **Cython**: https://cython.readthedocs.io/
- **PEP 517**: https://peps.python.org/pep-0517/
- **PEP 518**: https://peps.python.org/pep-0518/

---

## Quick Command Reference

```bash
# Complete publication workflow
rm -rf dist/ build/lib.* build/temp.*
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . && cd ..
python -m build
twine check dist/*
twine upload --repository testpypi dist/*  # Test first
twine upload dist/*  # Production

# Verify
pip install hpcs==0.7.0
python -c "import hpcs; print(hpcs.__version__)"
```

---

**Status:** Ready for v0.7.0 publication
**Date:** 2025-12-17
**Package:** `hpcs`
**Version:** `0.7.0`
