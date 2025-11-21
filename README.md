# HPC Series Core Library

High-performance computational kernels for time series analysis and statistical operations, implemented in modern Fortran with C/C++ interoperability.

## Overview

HPCSeriesCore provides optimized implementations of:
- **Robust statistics** (median, MAD, quantiles, robust z-scores)
- **Fast rolling operations** (O(n log w) rolling median/MAD using balanced BST)
- **Parallel statistics** (OpenMP-accelerated median, MAD, quantiles for large arrays)
- Rolling window operations (mean, std, min, max, etc.)
- Data quality functions (clipping, winsorization)
- Statistical transformations (z-score, rank, etc.)
- Group-based and simple reductions
- Efficient data processing for large-scale numerical computations

### Performance Highlights (v0.3)

- **4-5x faster** rolling median using C++ heap-based algorithm
- **2-3x faster** rolling MAD computation
- **2x faster** clipping operations via branchless optimization
- **Up to 2x speedup** for parallel statistics on large arrays (1M+ elements)

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

// Example 1: Fast rolling median (v0.3 - 4-5x faster!)
double timeseries[1000000];
double rolling_out[1000000];
int window = 100;
int status;

hpcs_rolling_median_fast(timeseries, 1000000, window, rolling_out, &status);

if (status == HPCS_SUCCESS) {
    // Process output... (first 99 elements are NaN)
}

// Example 2: Parallel robust statistics (v0.3)
double data[5000000];
double median, mad;
export OMP_NUM_THREADS=4  // Set in environment

hpcs_median_parallel(data, 5000000, &median, &status);
hpcs_mad_parallel(data, 5000000, &mad, &status);

// Example 3: Data quality operations
double values[1000];
hpcs_clip(values, 1000, 0.0, 100.0, &status);  // Branchless optimization applied
```

**Available Functions:**

**Robust Statistics (v0.2/v0.3):**
- Robust location/scale: `hpcs_median`, `hpcs_mad`, `hpcs_quantile`
- Rolling robust: `hpcs_rolling_median_fast`, `hpcs_rolling_mad_fast` (⚡ 4-5x faster)
- Parallel versions: `hpcs_median_parallel`, `hpcs_mad_parallel`, `hpcs_quantile_parallel`
- Quality checks: `hpcs_robust_zscore`, `hpcs_robust_zscore_parallel`
- Data cleaning: `hpcs_clip`, `hpcs_winsorize_by_quantiles`

**Rolling Operations (v0.1):**
- `hpcs_rolling_sum`, `hpcs_rolling_mean`, `hpcs_rolling_std`, `hpcs_rolling_var`
- `hpcs_rolling_min`, `hpcs_rolling_max`

**Reductions & Transforms:**
- Group reductions: `hpcs_group_reduce_sum`, `hpcs_group_reduce_mean`
- Simple reductions: `hpcs_reduce_sum`, `hpcs_reduce_min`, `hpcs_reduce_max`
- Statistical transforms: `hpcs_zscore`

**Array Utilities:**
- `hpcs_fill_value`, `hpcs_copy`, `hpcs_fill_missing`, `hpcs_where`

## Testing

**Status**: ✅ ALL TESTS PASSING
- **v0.1 Test Suite**: 26/26 tests passing (basic + comprehensive QA)
- **v0.2 Test Suite**: 10/10 tests passing (robust statistics)
- **v0.3 Test Suite**: 10/10 tests passing (optimized versions)

Run the test suites:
```bash
# Using Docker
docker compose run --rm hpcs-dev bash -c "cd build && ./test_core_c && ./test_core_cpp && ./test_v03"

# Or run tests individually
cd build
./test_core_c     # v0.1 C test suite
./test_core_cpp   # v0.1 C++ comprehensive tests
./test_v03        # v0.3 robust statistics tests

# Run performance benchmarks
export OMP_NUM_THREADS=4
./bench_v03                  # Original v0.3 benchmarks
./bench_v03_optimized        # Compare original vs optimized
```

**Test Coverage**:
- **v0.1**: Rolling operations, reductions, z-score, array utilities
- **v0.2**: Robust statistics (median, MAD, quantiles, robust z-score, clipping, winsorization)
- **v0.3**: Parallel statistics, fast rolling operations, branchless clipping

**Performance Benchmarks**:
- Array sizes: 100K, 500K, 1M, 5M, 10M elements
- Window sizes: 100 (configurable)
- Thread counts: 1-4 cores
- Output: CSV format for analysis

See [V03_COMPLETE.md](V03_COMPLETE.md) for v0.3 performance results.

## Documentation

### v0.3 Documentation (Performance Optimizations)
- [V03_COMPLETE.md](V03_COMPLETE.md) - Complete v0.3 implementation summary
- [V03_OPTIMIZATION_PLAN.md](V03_OPTIMIZATION_PLAN.md) - Optimization roadmap and strategy
- [V03_OPTIMIZATIONS_COMPLETE.md](V03_OPTIMIZATIONS_COMPLETE.md) - OpenMP and C++ rolling results
- [V03_BRANCHLESS_CLIPPING.md](V03_BRANCHLESS_CLIPPING.md) - Branchless optimization results

### v0.2 Documentation (Robust Statistics)
- [PHASE2_COMPLETE.md](PHASE2_COMPLETE.md) through [PHASE6_COMPLETE.md](PHASE6_COMPLETE.md) - Implementation phases

### v0.1 Documentation (Core Kernels)
- [Test Summary](docs/test_summary_v0.1.md) - Comprehensive testing report
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Feature overview
- [Calling Conventions](docs/calling_conventions.md) - C API documentation
- [Kernel Specifications](docs/hpcseries_core_kernel_spec.md) - Algorithm details
- [Build Instructions](BUILD_AND_TEST.md) - Detailed build guide

### Quick Reference
- [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) - Docker setup guide
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - Windows Fortran compiler setup
- [BENCHMARKING.md](BENCHMARKING.md) - Performance benchmarking guide

## Version

Current version: **0.3.0** (Released: 2025-11-20)

### Version History
- **0.3.0** (2025-11-20) - Performance optimizations (parallel statistics, fast rolling, branchless clipping)
- **0.2.0** (2025-11-19) - Robust statistics (median, MAD, quantiles, clipping, winsorization)
- **0.1.0** (2025-11-19) - Initial release (rolling operations, reductions, utilities)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
