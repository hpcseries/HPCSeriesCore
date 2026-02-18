# HPCSeries Core

**High-Performance Statistical Computing for Large-Scale Data Analysis**

[![Version](https://img.shields.io/badge/version-0.8.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/hpcseries-core/badge/?version=latest)](https://hpcseriescore.readthedocs.io/)
[![Architecture](https://img.shields.io/badge/arch-x86%20%7C%20ARM-orange.svg)](docs/PERFORMANCE.md)

---

## Overview

HPCSeries Core is a CPU-optimized statistical computing library for massive datasets (10M+ records). Provides **2-100x speedup** over NumPy/Pandas through SIMD vectorization (AVX2/AVX-512/NEON), OpenMP parallelization, and cache-optimized algorithms.

Built with Fortran, C, and C++ for maximum performance, with zero-copy Python bindings via Cython.

### Key Features

- **SIMD-Accelerated Operations**: sum, mean, std, min, max, median, MAD, quantile
- **Fast Rolling Windows**: 50-100x faster than Pandas for rolling operations
- **Anomaly Detection**: Statistical and robust outlier detection
- **Axis/Masked Operations**: Efficient 2D array and missing data handling
- **Auto-Tuning**: One-time calibration for optimal hardware performance
- **Architecture-Aware**: Automatic optimization for x86 (Intel/AMD) and ARM (Graviton)

### Performance Highlights

| Operation | Array Size | NumPy/Pandas | HPCSeries | Speedup |
|-----------|-----------|-------------|-----------|---------|
| `sum` | 1M | 0.45 ms | 0.12 ms | **3.8x** |
| `rolling_mean` | 100K (w=50) | 45 ms | 0.8 ms | **56x** |
| `rolling_median` | 100K (w=50) | 850 ms | 7.2 ms | **118x** |

Target use cases: 10M-1B records, time-series analysis, sensor data, financial analytics.

---

## Installation

### Quick Install

```bash
pip install hpcs
```

Verify:
```python
import hpcs
print(hpcs.__version__)  # 0.8.0
```

### Build from Source

```bash
git clone https://github.com/hpcseries/HPCSeriesCore.git
cd HPCSeriesCore

# Build C/Fortran library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Install Python bindings
pip install -e .
```

**Requirements**: Python 3.8+, NumPy 1.20+, GCC/gfortran 7+, CMake 3.18+

See [Build Guide](docs/BUILD_AND_TEST.md) for details.

---

## Quick Start

```python
import hpcs
import numpy as np

x = np.random.randn(1_000_000)

# Reductions (2-5x faster than NumPy)
hpcs.sum(x), hpcs.mean(x), hpcs.std(x)

# Rolling operations (50-100x faster than Pandas)
rolling_median = hpcs.rolling_median(x, window=100)

# Anomaly detection
anomalies = hpcs.detect_anomalies_robust(x, threshold=3.0)

# Composable pipelines for multi-stage processing
pipe = hpcs.pipeline(mode='fast')
pipe.diff(order=1).ewma(alpha=0.2).robust_zscore()
result = pipe.execute(x)

# Auto-tuning (run once)
hpcs.calibrate()
hpcs.save_calibration_config()
```

---

## C++ Integration

**NEW in v0.8.0**: HPCSeries Core now provides a **stable C ABI** for direct consumption from C++ projects, enabling high-performance computation without Python dependencies.

### Why Use Core from C++?

- **Zero Python Overhead**: Direct C++ → C calls with no interpreter
- **Production Deployment**: Ship C++ binaries without Python runtime
- **SignalCore Compatible**: Designed for SignalCore and similar C++ libraries
- **Stable ABI**: Version-tracked compatibility guarantees

### Quick Example

```cpp
#include <hpcs_core.h>
#include <vector>

int main() {
    // Check library version
    std::cout << "HPCSeries Core " << hpcs_get_version() 
              << " (ABI " << hpcs_get_abi_version() << ")\n";

    // Compute rolling mean directly from C++
    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> output(signal.size());
    int status;

    hpcs_rolling_mean(signal.data(), signal.size(), 3, 
                      output.data(), HPCS_MODE_FAST, &status);

    if (status == HPCS_SUCCESS) {
        // Use output...
    }

    return 0;
}
```

### Installation for C++

```bash
# Clone and build
git clone https://github.com/hpcseries/HPCSeriesCore.git
cd HPCSeriesCore
mkdir build && cd build

# Configure with shared library
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

# Build and install
make -j$(nproc)
sudo make install
```

This installs:
- `libhpcs_core.so.0.8.0` - Shared library with SONAME versioning
- `hpcs_core.h` - Public C API header
- CMake package config (for `find_package(hpcs_core)`)
- pkg-config support (for non-CMake projects)

### Integration Methods

**Option A: CMake find_package() (Recommended)**

```cmake
find_package(hpcs_core 0.8 CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE hpcs::hpcs_core)
```

**Option B: CMake Submodule**

```cmake
add_subdirectory(external/HPCSeriesCore)
target_link_libraries(myapp PRIVATE hpcs_core)
```

**Option C: pkg-config**

```bash
gcc myapp.c $(pkg-config --cflags --libs hpcs_core) -o myapp
```

### Performance

All 31 Core kernels support three execution modes:

| Mode | Use Case | Speedup |
|------|----------|---------|
| **SAFE** | Development, debugging | 1.0x (baseline) |
| **FAST** | Production (recommended) | **1.1-33x** |
| **DETERMINISTIC** | Testing, reproducibility | 0.5-1.0x |

Example benchmark (rolling_sum, n=10M, window=100):
- SAFE: 145ms
- FAST: 4.4ms (**32.8x speedup**)

### Documentation

- **[C API Reference](docs/C_API.md)** - Complete function reference
- **[Integration Guide](docs/INTEGRATION.md)** - Build system integration
- **[C++ Example](examples/cpp/signalcore_example.cpp)** - Full working example

### Execution Modes

```cpp
// Development: Full validation
hpcs_rolling_mean(data, n, w, out, HPCS_MODE_SAFE, &status);

// Production: Maximum performance
hpcs_rolling_mean(data, n, w, out, HPCS_MODE_FAST, &status);

// Testing: Reproducible results
hpcs_rolling_mean(data, n, w, out, HPCS_MODE_DETERMINISTIC, &status);
```

### Available Functions (31 Total)

**Rolling Operations**: `rolling_sum`, `rolling_mean`, `rolling_variance`, `rolling_std`

**Reductions**: `reduce_sum`, `reduce_mean`, `reduce_std`, `reduce_min`, `reduce_max`

**2D Operations**: `reduce_sum_axis1`, `reduce_mean_axis1`, `median_axis1`, `mad_axis1`

**Masked Operations**: `reduce_sum_masked`, `reduce_mean_masked`, `median_masked`

**Transformations**: `zscore`, `normalize_minmax`, `fill_forward`, `fill_backward`

**Specialized**: `detect_anomalies`, `prefix_sum`, `where`

See [C_API.md](docs/C_API.md) for complete reference.

---

## Pipeline API (v0.8.0)

**NEW in v0.8.0**: Composable kernel execution for multi-stage data processing. Chain multiple operations with automatic intermediate buffer management.

### Python Usage

```python
import hpcs
import numpy as np

x = np.random.randn(100_000)

# Create pipeline with execution mode
pipe = hpcs.pipeline(mode='fast')

# Chain operations (fluent API)
pipe.diff(order=1)          # First difference
pipe.ewma(alpha=0.2)        # Exponential smoothing
pipe.robust_zscore()        # MAD-based normalization

# Execute pipeline
result = pipe.execute(x)

# View pipeline summary
print(pipe.summary())
# Pipeline summary (3 stages):
#   1) diff(order=1)
#   2) ewma(alpha=0.2000)
#   3) robust_zscore(eps=1.00e-12)
```

### C API Usage

```c
#include <hpcs_core.h>

int status;
double x[1000], result[1000];

// Create pipeline
pipeline_t *pipe = pipeline_create(NULL, &status);

// Add stages
pipeline_add_diff(pipe, 1, &status);
pipeline_add_ewma(pipe, 0.2, &status);
pipeline_add_robust_zscore(pipe, 1e-12, &status);

// Execute
pipeline_execute(pipe, x, 1000, result, &status);

// Cleanup
pipeline_free(pipe);
```

### Available Pipeline Stages (12 Operations)

| Stage | Description | Parameters |
|-------|-------------|------------|
| `diff` | Finite differencing | `order` (lag) |
| `ewma` | Exponential weighted moving average | `alpha` ∈ (0,1] |
| `ewvar` | Exponential weighted variance | `alpha` ∈ (0,1] |
| `ewstd` | Exponential weighted std deviation | `alpha` ∈ (0,1] |
| `rolling_mean` | Rolling window mean | `window` size |
| `rolling_std` | Rolling window std deviation | `window` size |
| `rolling_median` | Rolling window median | `window` size |
| `rolling_mad` | Rolling window MAD | `window` size |
| `zscore` | Global z-score normalization | none |
| `robust_zscore` | MAD-based z-score | `eps` (stability) |
| `normalize_minmax` | Scale to [0,1] range | none |
| `clip` | Clamp values | `min_val`, `max_val` |

### Workspace for Memory-Intensive Pipelines

```python
# Pre-allocate workspace for large arrays
ws = hpcs.workspace(128 * 1024 * 1024)  # 128MB

# Use workspace with pipeline
pipe = hpcs.pipeline(ws=ws, mode='fast')
pipe.rolling_median(window=200)  # Memory-intensive
result = pipe.execute(large_array)
```

### Custom Operations (Before/After Pipeline)

The pipeline supports 12 predefined operations. For custom transformations, apply them before or after:

```python
# Custom pre-processing
x_log = np.log1p(np.abs(x)) * np.sign(x)

# Standard pipeline
pipe = hpcs.pipeline(mode='fast')
pipe.diff(1).ewma(0.2).robust_zscore()
result = pipe.execute(x_log)

# Custom post-processing
anomalies = np.abs(result) > 3.0
```

---

## Performance Configuration

### Optimal OpenMP Settings

```bash
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=2
```

**Why 2 threads?** Empirical testing on AMD EPYC Genoa, Intel Ice Lake, and ARM Graviton3 shows HPCSeries Core saturates memory bandwidth at 2 threads. Using 4+ threads degrades performance by 5-18% due to cache contention.

See [Performance Methodology](docs/PERFORMANCE.md) for full analysis.

### Additional Tips

- Ensure C-contiguous arrays: `np.ascontiguousarray(x)`
- Use robust functions (`median`, `robust_zscore`) for data with outliers
- Run calibration once: `hpcs.calibrate()` and `hpcs.save_calibration_config()`

---

## Documentation

### Official Documentation

**📚 [Read the Docs](https://hpcseriescore.readthedocs.io/)** - Complete API reference, user guides, and tutorials

### Core Documentation

- [Performance Methodology](docs/PERFORMANCE.md) - Empirical benchmarks and thread scaling
- [AWS Deployment Guide](docs/AWS_DEPLOYMENT_GUIDE.md) - Production deployment on EC2
- [Calibration Guide](docs/CALIBRATION_GUIDE.md) - Performance auto-tuning
- [NUMA Affinity Guide](docs/NUMA_AFFINITY_GUIDE.md) - Multi-socket optimization
- [Build & Test Guide](docs/BUILD_AND_TEST.md) - Compilation and testing

### Examples & Tutorials

- [Jupyter Notebooks](notebooks/) - 12 comprehensive tutorials covering:
  - Getting started and basic usage
  - Rolling operations and anomaly detection
  - Climate data, IoT sensors, financial analytics
  - NumPy/Pandas migration guide
  - Kaggle competition examples

See [Notebooks README](notebooks/README.md) for full list.

---

## Version History

### v0.8.0 (Current - 2026-01)
- **Pipeline API**: Composable kernel execution with 12 predefined stages
- **Workspace API**: Pre-allocated memory pools for SIMD/cache efficiency
- **Feature Discovery**: `build_features()` and `last_error()` for runtime introspection
- **Extended Transforms**: EWMA, EWVAR, EWSTD, differencing, convolution
- **Robust Statistics**: Trimmed mean, winsorized mean
- **Thread-local Error Handling**: Detailed error messages per thread

### v0.7.0 (2025-12-17)
- Architecture-aware compilation (x86 and ARM)
- AWS deployment infrastructure
- Comprehensive performance validation
- Thread scaling optimization (OMP_NUM_THREADS=2 universal)

See [CHANGELOG.md](CHANGELOG.md) for complete history.

---

## Project Structure

```
HPCSeriesCore/
├── src/                      # C/Fortran/C++ source
│   ├── fortran/              # HPC kernels (OpenMP)
│   └── hpcs_*.c              # SIMD implementations
├── include/                  # C API headers
├── python/hpcs/              # Python bindings (Cython)
├── cmake/                    # CMake modules (architecture detection)
├── notebooks/                # Jupyter tutorials
├── docs/                     # Documentation
├── tests/                    # Test suites
└── bench/                    # Benchmarks
```

---

## Support

- **Documentation**: [Read the Docs](https://hpcseriescore.readthedocs.io/)
- **Contributing**: [Contributing Guide](https://hpcseriescore.readthedocs.io/en/latest/contributing.html)
- **Bug Reports**: [GitHub Issues](https://github.com/hpcseries/HPCSeriesCore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hpcseries/HPCSeriesCore/discussions)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

HPCSeries Core is licensed under the Apache License, Version 2.0, which provides:
- **Patent protection** for both contributors and users
- **Commercial-friendly** licensing for enterprise adoption
- **Flexibility** for commercial use, modification, and distribution
- **Trademark protection** preserving the HPCSeries brand

This license enables both open-source collaboration and commercial applications, including use in proprietary software.

---

## Citation

If you use HPCSeries Core in your research, please cite:

```bibtex
@software{hpcseries_core_2026,
  title = {HPCSeries Core: High-Performance Statistical Computing for Large-Scale Data Analysis},
  author = {HPCSeries Core Contributors},
  year = {2026},
  month = {1},
  version = {0.8.0},
  url = {https://github.com/hpcseries/HPCSeriesCore},
  license = {Apache-2.0}
}
```

Or use GitHub's **"Cite this repository"** button (auto-generated from [CITATION.cff](CITATION.cff)).

---

**⭐ Star us on GitHub if HPCSeries Core accelerates your data analysis!**
