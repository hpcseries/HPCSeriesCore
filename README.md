# HPCSeries Core

**High-Performance Statistical Computing for Large-Scale Data Analysis**

[![Version](https://img.shields.io/badge/version-0.7.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/arch-x86%20%7C%20ARM-orange.svg)](docs/PERFORMANCE.md)

---

## What is HPCSeries Core?

HPCSeries Core is a **CPU-optimized** statistical computing library designed for **massive datasets (10M+ records)**. It provides **2-100x faster** operations than NumPy/Pandas through SIMD vectorization (AVX2/AVX-512), OpenMP parallelization, and cache-friendly algorithms.

Built with Fortran, C, and C++ for maximum performance, with **zero-copy Python bindings** for ease of use.

### Why HPCSeries?

**Problem**: NumPy and Pandas are too slow for large-scale time-series analysis and rolling operations.

**Solution**: HPCSeries provides:
- ✅ **2-5x faster** reductions (sum, mean, std) via SIMD vectorization
- ✅ **50-100x faster** rolling operations (rolling mean, median, MAD)
- ✅ **Sub-microsecond latency** for small arrays
- ✅ **Scales to billions** of elements with OpenMP parallelization
- ✅ **Drop-in replacement** for NumPy/Pandas functions
- ✅ **Zero-copy** NumPy integration (no data copying overhead)

### Alternatives & Comparison

| Library | Rolling Median (100K elements) | SIMD | OpenMP | Python API |
|---------|-------------------------------|------|--------|------------|
| **Pandas** | ~850 ms | ❌ | ❌ | ✅ |
| **NumPy** | ~750 ms | ✅ | ❌ | ✅ |
| **Numba** | ~120 ms | ⚠️ | ⚠️ | ✅ (JIT) |
| **HPCSeries** | **~7 ms** | ✅ | ✅ | ✅ |

**Speedup**: **100-120x faster** than Pandas for rolling operations!

---

## Key Features

### Core Operations
- **Reductions**: `sum`, `mean`, `var`, `std`, `min`, `max` (SIMD-accelerated)
- **Robust Statistics**: `median`, `MAD`, `quantile`, `robust_zscore`
- **Rolling Operations**: Fast sliding windows (mean, median, std, MAD, z-score)
- **Anomaly Detection**: Statistical and robust outlier detection
- **Axis Operations**: Per-column/per-row statistics for 2D arrays
- **Masked Operations**: Handle missing data efficiently

### Performance Features
- **SIMD Vectorization**: Automatic AVX2/AVX-512/SSE2 dispatch
- **OpenMP Parallelization**: Multi-threaded for large datasets
- **Adaptive Auto-Tuning**: One-time calibration for optimal performance
- **Cache Optimization**: Block processing for L1/L2/L3 cache efficiency
- **Zero-Copy Design**: Direct NumPy array access (no memory overhead)

### Developer Experience
- **NumPy-Compatible API**: Familiar function signatures
- **12 Jupyter Notebooks**: Real-world examples and tutorials
- **CLI Tools**: `hpcs cpuinfo` for hardware detection
- **Type Safety**: Automatic dtype conversion
- **Comprehensive Docs**: Sphinx documentation with API reference

---

## Performance Highlights

### Real-World Benchmarks (AMD Ryzen 7, AVX2)

| Operation | Array Size | NumPy/Pandas | HPCSeries | Speedup |
|-----------|-----------|-------------|-----------|---------|
| `sum` | 1M | 0.45 ms | 0.12 ms | **3.8x** |
| `mean` | 1M | 0.48 ms | 0.13 ms | **3.7x** |
| `std` | 1M | 1.20 ms | 0.35 ms | **3.4x** |
| `rolling_mean` | 100K (w=50) | 45 ms | 0.8 ms | **56x** |
| `rolling_median` | 100K (w=50) | 850 ms | 7.2 ms | **118x** |

### Scaling to Large Data

| Records | Operation | Time | Throughput |
|---------|-----------|------|------------|
| 10M | `sum` | 7.8 ms | 10.3 GB/s |
| 100M | `rolling_mean` | 85 ms | 9.4 GB/s |
| 1B | `mean` (8 threads) | 1.2 s | 6.7 GB/s |

**Target Use Cases**: 10M-1B records, time-series analysis, sensor data, financial analytics.

---

## Installation

### System Requirements

**Minimum**:
- Python 3.8 or higher
- NumPy >= 1.20
- CPU with SSE2 support (Intel/AMD x86-64, released after 2003)

**Recommended**:
- CPU with AVX2 or AVX-512 support for maximum performance
- 8+ CPU cores for parallel operations
- Linux, macOS, or Windows (WSL2 recommended on Windows)

**Compilers** (for building from source):
- GCC 7+ or Clang 10+ (C compiler)
- gfortran 7+ (Fortran compiler)
- CMake 3.18+

---

### Option 1: Install via pip (Recommended)

```bash
pip install hpcs
```

Verify installation:

```python
import hpcs
print(hpcs.__version__)  # Should print: 0.7.0

# Check SIMD capabilities
print(hpcs.simd_info())
# {'isa': 'AVX2', 'width_bytes': 32, 'width_doubles': 4}
```

---

### Option 2: Install via Docker

**For development, or testing:**

```bash
# Clone repository
git clone https://github.com/your-org/HPCSeriesCore.git
cd HPCSeriesCore

# Build and run Docker container
docker compose -f docker-compose.python.yml up -d

# Enter container
docker compose -f docker-compose.python.yml exec hpcs-python bash

# HPCSeries is pre-installed and ready to use
python3 -c "import hpcs; print(hpcs.__version__)"
```

**Docker includes**:
- Pre-built HPCSeries library
- Python bindings
- All dependencies (NumPy, Jupyter, Matplotlib, etc.)
- Build tools (GCC, gfortran, CMake)

See [Docker Guide](docs/DOCKER_QUICKSTART.md) for details.

---

### Option 3: Build from Source

**Prerequisites**:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install gcc gfortran cmake python3-dev

# macOS (via Homebrew)
brew install gcc cmake

# Windows (via MSYS2)
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-cmake
```

**Build and Install**:

```bash
# Clone repository
git clone https://github.com/your-org/HPCSeriesCore.git
cd HPCSeriesCore

# Build C/Fortran library
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_C_COMPILER=gcc \
         -DCMAKE_Fortran_COMPILER=gfortran
make -j$(nproc)
cd ..

# Build and install Python bindings
pip install -e .
```

**Verify**:

```bash
python3 -c "import hpcs; hpcs.calibrate(); hpcs.save_calibration_config()"
```

See [Build Guide](docs/BUILD_AND_TEST.md) for detailed instructions.

---

## Quick Start

### Basic Operations

```python
import hpcs
import numpy as np

# Create sample data
x = np.random.randn(1_000_000)

# SIMD-accelerated reductions (2-5x faster than NumPy)
hpcs.sum(x)      # 500.123
hpcs.mean(x)     # 0.00050
hpcs.std(x)      # 1.0002
hpcs.min(x)      # -4.8234
hpcs.max(x)      # 4.7123
```

### Rolling Operations

```python
# 50-100x faster than Pandas!
data = np.random.randn(100_000)

rolling_mean = hpcs.rolling_mean(data, window=50)
rolling_median = hpcs.rolling_median(data, window=100)
rolling_std = hpcs.rolling_std(data, window=50)
```

### Anomaly Detection

```python
# Statistical anomaly detection
anomalies = hpcs.detect_anomalies(data, threshold=3.0)

# Robust detection (MAD-based, outlier-resistant)
robust_anomalies = hpcs.detect_anomalies_robust(data, threshold=3.0)
```

### Performance Calibration

```python
# One-time auto-tuning for your hardware (~30 seconds)
hpcs.calibrate()
hpcs.save_calibration_config()  # Saves to ~/.hpcs/config.json

# # Config loaded automatically in subsequent imports
import hpcs  
```

### Hardware Detection

```bash
# CLI tool for CPU information
hpcs cpuinfo
```

Example Output:
```
=== CPU Information ===

CPU Vendor:          AuthenticAMD
Physical Cores:      8
Logical Cores:       16
Optimal Threads:     8

Cache Hierarchy:
  L1:      32 KB
  L2:     256 KB
  L3:    4096 KB

SIMD Capabilities:
  Active ISA:          AVX2
  Vector width:        256-bit (4 doubles)
  AVX-512:             ✗
  AVX2:                ✓
  AVX:                 ✓
  SSE2:                ✓
```

---

## Documentation

### 📚 Main Documentation

- **[Full Documentation](https://hpcseries.readthedocs.io)** (Sphinx docs)
- **[API Reference](docs/source/api/index.rst)** - Complete function reference
- **[Installation Guide](docs/source/installation.rst)** - Detailed setup instructions
- **[Quick Start](docs/source/quickstart.rst)** - Hands-on tutorial
- **[User Guide](docs/source/user_guide/index.rst)** - Architecture, performance, migration

### Jupyter Notebooks

Located in [`notebooks/`](notebooks/) directory:

- **[00_getting_started.ipynb](notebooks/00_getting_started.ipynb)** - Introduction and basic usage
- **[01_rolling_mean_vs_median.ipynb](notebooks/01_rolling_mean_vs_median.ipynb)** - Rolling operations comparison
- **[02_robust_anomaly_climate.ipynb](notebooks/02_robust_anomaly_climate.ipynb)** - Climate data anomaly detection
- **[03_batched_iot_rolling.ipynb](notebooks/03_batched_iot_rolling.ipynb)** - IoT sensor processing
- **[04_axis_reductions_column_stats.ipynb](notebooks/04_axis_reductions_column_stats.ipynb)** - 2D array operations
- **[05_masked_missing_data.ipynb](notebooks/05_masked_missing_data.ipynb)** - Handling missing data
- **[06_performance_calibration.ipynb](notebooks/06_performance_calibration.ipynb)** - Auto-tuning guide
- **[07_c_optimized_operations.ipynb](notebooks/07_c_optimized_operations.ipynb)** - SIMD deep dive
- **[08_numpy_pandas_migration_guide.ipynb](notebooks/08_numpy_pandas_migration_guide.ipynb)** - Migration examples
- **[09_real_world_applications.ipynb](notebooks/09_real_world_applications.ipynb)** - Production use cases
- **[HPCSeries_Kaggle_StoreSales_v1.ipynb](notebooks/HPCSeries_Kaggle_StoreSales_v1.ipynb)** - Kaggle competition (baseline)
- **[HPCSeries_Kaggle_StoreSales_v2.ipynb](notebooks/HPCSeries_Kaggle_StoreSales_v2.ipynb)** - Kaggle competition (optimized)

See [Notebooks README](notebooks/README.md) for detailed descriptions.

### 🔧 Technical Documentation

- **[Architecture Guide](docs/source/user_guide/architecture.rst)** - System design and internals
- **[Performance Guide](docs/source/user_guide/performance.rst)** - Optimization and tuning
- **[Migration Guide](docs/source/user_guide/migration.rst)** - NumPy/Pandas → HPCSeries
- **[Calibration Guide](docs/CALIBRATION_GUIDE.md)** - Performance auto-tuning
- **[NUMA Affinity Guide](docs/NUMA_AFFINITY_GUIDE.md)** - Multi-socket optimization

### 🚀 Deployment

- **[AWS Deployment Guide](docs/AWS_DEPLOYMENT_GUIDE.md)** - Production deployment on AWS EC2 (CPU-optimized instances)
- **[Performance Methodology](docs/PERFORMANCE.md)** - Empirical benchmarks and thread scaling analysis

---

## Use Cases

### Financial Analytics
- High-frequency trading signal generation
- Risk metrics (rolling volatility, Sharpe ratio)
- Anomaly detection in transaction streams

### IoT & Sensor Data
- Real-time sensor stream processing
- Multi-sensor anomaly detection
- Predictive maintenance (rolling statistics)

### Scientific Computing
- Climate data analysis
- Large-scale simulations
- Time-series feature engineering

### Data Science
- Feature engineering for ML pipelines
- Exploratory data analysis (EDA)
- Data quality monitoring

---

## Project Structure

```
HPCSeriesCore/
├── src/                      # C/Fortran/C++ source code
│   ├── fortran/              # Fortran HPC kernels
│   ├── hpcs_*.c              # C orchestration layer
│   └── hpcs_*.cpp            # C++ optimized algorithms
├── include/                  # Public C headers
│   └── hpcs_core.h           # Main C API
├── python/                   # Python bindings
│   ├── hpcs/                 # Python package
│   │   ├── __init__.py       # Main API
│   │   ├── _core.pyx         # Cython bindings (core)
│   │   ├── _simd.pyx         # Cython bindings (SIMD)
│   │   └── cli.py            # CLI tools
│   └── setup.py              # Python build script
├── notebooks/                # Jupyter tutorial notebooks (12)
├── docs/                     # Sphinx documentation
│   ├── source/               # Documentation source files
│   └── build/                # Generated HTML docs
├── tests/                    # Test suites (C/Fortran/Python)
├── bench/                    # Performance benchmarks
├── examples/                 # C/Fortran usage examples
├── CMakeLists.txt            # CMake build configuration
├── pyproject.toml            # Python project metadata
├── docker-compose.python.yml # Docker development environment
└── README.md                 # This file
```

---

## Performance Tips

### 1. Run Calibration Once

```python
import hpcs

# First time setup (~30 seconds)
hpcs.calibrate()
hpcs.save_calibration_config()

# Subsequent runs: config loaded automatically
```

### 2. Use Contiguous Arrays

```python
# Ensure C-contiguous layout
if not x.flags['C_CONTIGUOUS']:
    x = np.ascontiguousarray(x)
```

### 3. Set OpenMP Threads

```bash
export OMP_NUM_THREADS=8  # Use physical core count
python your_script.py
```

### 4. Choose Right Function

```python
# For clean data: Use standard functions
hpcs.mean(data)
hpcs.zscore(data)

# For data with outliers: Use robust versions
hpcs.median(data)
hpcs.robust_zscore(data)
```

See [Performance Guide](docs/source/user_guide/performance.rst) for more tips.

---

## Version History

### v0.7.0 (Current - 2025-01-XX)
- ✅ Complete Python API with Cython bindings
- ✅ 12 comprehensive Jupyter notebook tutorials
- ✅ Sphinx documentation with Read the Docs theme
- ✅ CLI tools (`hpcs cpuinfo`)
- ✅ Performance calibration and auto-tuning
- ✅ SIMD vectorization (AVX2/AVX-512/SSE2)
- ✅ OpenMP parallelization
- ✅ Anomaly detection (statistical and robust)
- ✅ Axis and masked operations

### Previous Versions
- **v0.6.0** - SIMD dispatch and Fortran-C bridge
- **v0.5.0** - Calibration system and CPU detection
- **v0.4.0** - 2D operations and batched processing
- **v0.3.0** - Performance optimizations (parallel, fast rolling)
- **v0.2.0** - Robust statistics (median, MAD, quantiles)
- **v0.1.0** - Initial release (core kernels)

See [CHANGELOG](docs/source/changelog.rst) for detailed history.

---

## Contributing

We welcome contributions! See [Contributing Guide](docs/source/contributing.rst) for:

- Code style guidelines
- Development setup
- Testing requirements
- Pull request process

**Good first issues**:
- Adding examples to notebooks
- Improving documentation
- Adding test coverage
- Performance benchmarks

---

## Support

- ** Documentation**: https://hpcseries.readthedocs.io
- ** Bug Reports**: [GitHub Issues](https://github.com/your-org/HPCSeriesCore/issues)
- ** Discussions**: [GitHub Discussions](https://github.com/your-org/HPCSeriesCore/discussions)
- ** Email**: support@hpcseries.org

---

## License

[Specify your license - MIT, Apache 2.0, etc.]

---

## Citation

If you use HPCSeries Core in your research, please cite:

```bibtex
@software{hpcseries_core,
  title = {HPCSeries Core: High-Performance Statistical Computing Library},
  author = {HPCSeries Core Team},
  year = {2025},
  version = {0.7.0},
  url = {https://github.com/your-org/HPCSeriesCore}
}
```

---

## Acknowledgments

Built with:
- **Fortran** - HPC kernels and OpenMP parallelization
- **C/C++** - SIMD intrinsics and orchestration
- **Cython** - Zero-copy Python bindings
- **NumPy** - Array interface
- **CMake** - Build system
- **Sphinx** - Documentation

Special thanks to the open-source community for tools and libraries that made this possible.

---

**⭐ Star us on GitHub if HPCSeries Core helps accelerate your data analysis!**
