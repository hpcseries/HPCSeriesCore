# HPCSeries Core v0.7 - Test Suite
================================

Comprehensive test suite for HPCSeries Core across Python, C, and Fortran.

## Directory Structure

```
tests/
├── python/          # Python tests (pytest)
│   ├── conftest.py
│   ├── test_reductions.py
│   ├── test_rolling.py
│   └── test_simd.py
├── c/               # C tests
│   ├── Makefile
│   ├── test_hpcs_robust.c
│   ├── test_fortran_simd.c
│   └── ...
└── fortran/         # Fortran tests
    ├── Makefile
    └── test_fortran_basic.f90
```

## Running Tests

### Python Tests (Recommended)

```bash
# Install dependencies first
pip install -e .

# Run all Python tests
cd tests/python
pytest -v

# Run specific test file
pytest test_reductions.py -v

# Run with coverage
pytest --cov=hpcs --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Run only SIMD tests
pytest test_simd.py -v
```

### C Tests

```bash
# Build C library first
cmake -S . -B build
cmake --build build

# Run all C tests
cd tests/c
make run

# Run specific test
make test_hpcs_robust

# Clean
make clean
```

### Fortran Tests

```bash
# Build C library first (if not done)
cmake -S . -B build
cmake --build build

# Run all Fortran tests
cd tests/fortran
make run

# Run specific test
make test_fortran_basic

# Clean
make clean
```

### Docker Tests

```bash
# Build Docker image
docker build -f Dockerfile.python -t hpcs-python:v0.7 .

# Run Python tests in Docker
docker run --rm hpcs-python:v0.7 pytest /workspace/tests/python -v

# Run CLI tests
docker run --rm hpcs-python:v0.7 hpcs test
```

## Test Categories

### Python Tests

#### test_reductions.py
- **TestBasicReductions**: sum, mean, std, var, min, max
- **TestRobustStatistics**: median, MAD
- **TestTypeConversions**: List, NumPy array, different dtypes
- **TestEdgeCases**: Single element, zeros, large values

#### test_rolling.py
- **TestRollingMean**: Rolling mean with various windows
- **TestRollingStd**: Rolling standard deviation
- **TestRollingVar**: Rolling variance
- **TestRollingMedian**: Robust rolling median
- **TestRollingMAD**: Rolling MAD
- **TestRollingEdgeCases**: Edge cases for rolling ops
- **TestRollingPerformance**: Performance benchmarks

#### test_simd.py
- **TestSIMDInfo**: SIMD capability detection
- **TestCPUInfo**: CPU topology and cache information
- **TestSIMDIntegration**: Integration with operations

### C Tests

- **test_hpcs_robust.c**: Robust statistics (median, MAD, anomaly detection)
- **test_fortran_simd.c**: Fortran-SIMD integration
- **test_hpcs_gpu_*.c**: GPU acceleration tests (Phase 2)

### Fortran Tests

- **test_fortran_basic.f90**: Basic Fortran module integration

## Test Metrics

### Expected Results

```
Python Tests:   40+ tests, all passing
C Tests:        5+ tests, all passing
Fortran Tests:  4+ tests, all passing
```

### Performance Benchmarks

On modern CPU with AVX2:
- Sum (1M elements): ~1.5ms
- Rolling mean (1M, w=100): ~5ms
- Rolling median (100K, w=200): ~30ms
- Median (1M elements): ~100ms

## Continuous Integration

The test suite is designed to run in CI/CD:

```yaml
# GitHub Actions example
- name: Run Python tests
  run: |
    pip install -e .
    pytest tests/python -v

- name: Run C tests
  run: |
    cd tests/c && make run

- name: Run Fortran tests
  run: |
    cd tests/fortran && make run
```

## Test Coverage

Target coverage: >80% for Python bindings

```bash
# Generate coverage report
pytest tests/python --cov=hpcs --cov-report=term-missing
```

## Adding New Tests

### Python Test Template

```python
import pytest
import hpcs
import numpy as np

class TestNewFeature:
    def test_basic_case(self):
        """Test basic functionality."""
        data = [1.0, 2.0, 3.0]
        result = hpcs.new_function(data)
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case."""
        # ...
```

### C Test Template

```c
#include <stdio.h>
#include <assert.h>
#include "hpcs_core.h"

void test_new_function() {
    double data[] = {1.0, 2.0, 3.0};
    double result;
    int status;

    hpcs_new_function(data, 3, &result, &status);

    assert(status == HPCS_SUCCESS);
    assert(fabs(result - expected) < 1e-10);
    printf("✓ test_new_function passed\n");
}

int main() {
    test_new_function();
    return 0;
}
```

## Troubleshooting

### Python Tests Fail with Import Error

```bash
# Ensure package is installed
pip install -e .

# Check PYTHONPATH
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```

### C Tests Fail to Link

```bash
# Rebuild library
cmake -S . -B build
cmake --build build

# Check library exists
ls -lh build/libhpcs_core.a
```

### Fortran Tests Can't Find Modules

```bash
# Ensure Fortran modules are built
cmake --build build

# Check module directory
ls -lh build/fortran_modules/
```

## Test Philosophy

1. **Fast by default**: Most tests complete in <1s
2. **Comprehensive**: Cover all public API functions
3. **Isolated**: Tests don't depend on each other
4. **Clear failures**: Descriptive assertion messages
5. **Reproducible**: Fixed random seeds where needed

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure all existing tests pass
3. Add tests for edge cases
4. Document expected behavior
5. Run full test suite before submitting
