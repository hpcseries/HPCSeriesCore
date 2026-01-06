# HPCSeries Core

**High-Performance Statistical Computing for Large-Scale Data Analysis**

[![Version](https://img.shields.io/badge/version-0.7.0-blue.svg)](CHANGELOG.md)
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
print(hpcs.__version__)  # 0.7.0
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

# Auto-tuning (run once)
hpcs.calibrate()
hpcs.save_calibration_config()
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

### v0.7.0 (Current - 2025-12-17)
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
@software{hpcseries_core_2025,
  title = {HPCSeries Core: High-Performance Statistical Computing for Large-Scale Data Analysis},
  author = {HPCSeries Core Contributors},
  year = {2025},
  month = {12},
  version = {0.7.0},
  url = {https://github.com/hpcseries/HPCSeriesCore},
  license = {MIT}
}
```

Or use GitHub's **"Cite this repository"** button (auto-generated from [CITATION.cff](CITATION.cff)).

---

**⭐ Star us on GitHub if HPCSeries Core accelerates your data analysis!**
