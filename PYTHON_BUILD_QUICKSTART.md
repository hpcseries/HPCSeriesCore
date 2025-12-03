# HPCSeries Core v0.7 - Python Build Quickstart

## Option 1: Docker (Recommended for Development)

### Build and Run Python Development Container

```bash
# Build Docker image with Python support
docker build -f Dockerfile.python -t hpcs-python:v0.7 .

# Or use docker-compose
docker-compose -f docker-compose.python.yml up -d

# Enter container
docker exec -it hpcs-python-dev bash

# Inside container - verify installation
python3 -c "import hpcs; print(hpcs.__version__)"
hpcs version
hpcs cpuinfo
```

---

## Option 2: Manual Build (Native Linux/WSL2)

### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake gfortran gcc g++ \
    python3 python3-pip python3-dev \
    libomp-dev libgfortran5

# Fedora/RHEL
sudo dnf install -y \
    gcc gcc-c++ gcc-gfortran cmake \
    python3 python3-pip python3-devel \
    libomp-devel
```

### Step 2: Install Python Dependencies

```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install build dependencies
pip3 install -r requirements.txt
```

### Step 3: Build C Library

```bash
# Configure and build
cmake -S . -B build
cmake --build build -j$(nproc)

# Verify C library
ls -lh build/libhpcs_core.a
```

### Step 4: Build Python Extensions

```bash
# Development installation (editable)
pip3 install -e .

# Or build wheel for distribution
python3 -m build
pip3 install dist/hpcs-0.7.0-*.whl
```

### Step 5: Verify Installation

```bash
# Test Python import
python3 -c "import hpcs; print('✓ hpcs version:', hpcs.__version__)"

# Test CLI
hpcs version
hpcs cpuinfo
hpcs test

# Run benchmarks
hpcs bench --size 1000000 --iterations 10
```

---

## Option 3: Quick Test Without Installation

If you just want to test the bindings without full installation:

```bash
# Build C library
cmake -S . -B build && cmake --build build -j$(nproc)

# Build Cython extensions in-place
python3 setup.py build_ext --inplace

# Add to Python path and test
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python3 -c "import hpcs; print(hpcs.sum([1,2,3,4,5]))"
```

---

## Troubleshooting

### "No module named 'numpy'"
```bash
pip3 install numpy Cython
```

### "libhpcs_core.a not found"
```bash
# Build C library first
cmake -S . -B build
cmake --build build
```

### "libgfortran.so.5: cannot open shared object"
```bash
# Ubuntu/Debian
sudo apt-get install libgfortran5

# Fedora
sudo dnf install libgfortran
```

### Cython compilation errors
```bash
# Ensure development headers are installed
sudo apt-get install python3-dev

# Try building with verbose output
python3 setup.py build_ext --inplace --verbose
```

---

## Development Workflow

```bash
# 1. Make changes to C code
vim src/hpcs_reduce_simd.c

# 2. Rebuild C library
cmake --build build

# 3. Rebuild Python extensions
pip3 install -e . --force-reinstall --no-deps

# 4. Test changes
python3 -c "import hpcs; print(hpcs.sum([1,2,3]))"
hpcs test
```

---

## Running Tests

```bash
# Built-in CLI tests
hpcs test

# Python unit tests (when implemented)
pytest python/tests/

# Performance benchmarks
hpcs bench
```

---

## Building Distribution Wheels

```bash
# Install build tools
pip3 install build twine

# Build wheel
python3 -m build

# Check wheel
twine check dist/*.whl

# Install locally
pip3 install dist/hpcs-0.7.0-*.whl
```

---

## Current Status

✅ **Working:**
- Core reductions (sum, mean, std, var, min, max)
- Robust statistics (median, MAD)
- Rolling operations (mean, std, var, median, MAD)
- SIMD info API (simd_info, get_cpu_info)
- CLI tool (version, cpuinfo, bench, test)

⏳ **Not Yet Implemented:**
- Masked operations
- Batched/axis operations
- Anomaly detection functions
- Full test suite

---

## Performance Expectations

On a modern CPU with AVX2:
- `hpcs.sum()`: **1.56x faster** than NumPy
- `hpcs.rolling_mean()`: **3.75x faster** than pandas
- `hpcs.rolling_median()`: **21x faster** than pandas

All reductions use SIMD-accelerated kernels from v0.6!
