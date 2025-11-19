# HPC Series Core Library

High-performance computational kernels for time series analysis and statistical operations, implemented in modern Fortran with C/C++ interoperability.

## Overview

HPCSeriesCore provides optimized implementations of:
- Rolling window operations (mean, std, min, max, etc.)
- Statistical transformations (z-score, rank, etc.)
- Group-based and simple reductions
- Efficient data processing for large-scale numerical computations

## Project Structure

```
/HPCSeriesCore/
├── CMakeLists.txt         # Top-level build configuration
├── include/               # Public C headers
│   └── hpcs_core.h        # C-accessible kernel prototypes
├── src/
│   ├── fortran/           # Modern Fortran implementations
│   └── cpp/               # C++ wrappers (future)
├── tests/                 # Test suites
├── bench/                 # Performance benchmarks
├── docs/                  # Documentation
└── build/                 # Build directory (out-of-source)
```

## Building

### Option 1: Docker (Recommended for WSL/Linux)

**Quick Start:**
```bash
# In WSL, navigate to project
cd "/mnt/c/Users/Samkelo/OneDrive - iqlab/Clients/HPCSeriesCore"

# Start container
docker-compose up -d

# Enter container and build
docker-compose exec hpcs-dev bash
./docker-build.sh
```

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for details.

### Option 2: Native Build

**Prerequisites:**
- CMake 3.18 or higher
- Modern Fortran compiler (GFortran 9+, Intel Fortran, etc.)
- C/C++ compiler (GCC, Clang, MSVC, etc.)

**Build Instructions:**
```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build .

# Run tests
ctest

# Install (optional)
cmake --install . --prefix /desired/install/path
```

**Build Options:**
- `BUILD_TESTS`: Enable/disable test suite (default: ON)
- `BUILD_BENCHMARKS`: Enable/disable benchmarks (default: ON)

```bash
cmake -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=ON ..
```

**Windows Setup:**
See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for Fortran compiler installation options.

## Usage

Link against `libhpcs_core.a` and include `hpcs_core.h`:

```c
#include <hpcs_core.h>

// Example: Rolling mean
double input[1000];
double output[1000];
int window = 20;
int status;

hpcs_rolling_mean(input, 1000, window, output, &status);

if (status == HPCS_SUCCESS) {
    // Process output...
} else {
    // Handle error (HPCS_ERR_INVALID_ARGS, etc.)
}
```

**Available Functions:**
- Rolling operations: `hpcs_rolling_sum`, `hpcs_rolling_mean`
- Group reductions: `hpcs_group_reduce_sum`, `hpcs_group_reduce_mean`
- Simple reductions: `hpcs_reduce_sum`, `hpcs_reduce_min`, `hpcs_reduce_max`
- Statistical transforms: `hpcs_zscore`
- Utilities: `hpcs_fill_value`, `hpcs_copy`, `hpcs_fill_missing`, `hpcs_where`

## Testing

**Status**: ✅ ALL TESTS PASSING (26/26)
- **C Test Suite**: 8/8 tests passing (basic functionality)
- **C++ Test Suite**: 18/18 tests passing (comprehensive QA with reference implementations)

Run the test suites:
```bash
# Using Docker (runs both C and C++ tests)
docker compose run --rm hpcs-dev bash -c "./docker-build.sh"

# Or run tests individually
cd build
./test_core_c     # C test suite (basic)
./test_core_cpp   # C++ comprehensive test suite
```

**Test Coverage**:
- Rolling operations: basic, edge cases (window=1, window=n), NaN propagation, error handling
- Grouped reductions: basic, invalid groups, empty groups, error handling
- Simple reductions: basic, NaN propagation, error handling
- Z-score normalization: basic, constant arrays, error handling
- Array utilities: `fill_value`, `copy` operations

**Test Methods**:
- C tests: Basic smoke tests with hardcoded expected values
- C++ tests: Reference implementations for numerical validation (tolerance: 1e-12)

See [docs/QUICKSTART_RESULTS.md](docs/QUICKSTART_RESULTS.md) for latest test results.

## Documentation

See [docs/](docs/) for:
- [Test Summary](docs/test_summary_v0.1.md) - Comprehensive testing report
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Feature overview
- [Calling Conventions](docs/calling_conventions.md) - C API documentation
- [Kernel Specifications](docs/hpcseries_core_kernel_spec.md) - Algorithm details
- [Build Instructions](BUILD_AND_TEST.md) - Detailed build guide

## Version

Current version: **0.1.0** (Released: 2025-11-19)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
