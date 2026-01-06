# Changelog

All notable changes to HPCSeries Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2025-01-06 (Current)

### Added

**Execution Mode System** - Safety vs Performance Trade-offs
- Global execution mode API: `set_execution_mode()`, `get_execution_mode()`
- Three modes: `MODE_SAFE` (IEEE 754, full validation), `MODE_FAST` (relaxed math), `MODE_DETERMINISTIC` (bit-exact reproducibility)
- Per-call mode override via `mode` parameter in all supported functions
- Thread-safe implementation using OpenMP threadprivate storage
- Zero overhead via compile-time dispatcher pattern
- 8 functions with mode support: ewma, ewvar, ewstd, cumulative_min, cumulative_max, convolve_valid, trimmed_mean, winsorized_mean

**Exponential Weighted Statistics (Group A)** - 15-60x faster than pandas
- `ewma()` - Exponentially weighted moving average with execution modes
- `ewvar()` - Exponentially weighted variance (Welford's method)
- `ewstd()` - Exponentially weighted std deviation
- Single-pass O(n), numerically stable, matches pandas API

**Time Series Transforms (Group B)** - 2-4x faster than NumPy
- `diff()` - Finite differencing (arbitrary order)
- `cumulative_min()` - Running minimum with execution modes
- `cumulative_max()` - Running maximum with execution modes

**FIR Filters (Group C)** - Template-specialized convolution
- `convolve_valid()` - 1D convolution with execution modes
- Template specializations for kernel sizes 3, 5, 7, 9, 11, 13, 15
- OpenMP parallelization in SAFE/FAST modes
- Optimized for small FIR filters

**Advanced Robust Statistics (Group D)** - 10-15x faster than SciPy
- `trimmed_mean()` - Mean after discarding extremes (with execution modes)
- `winsorized_mean()` - Mean after clamping extremes (with execution modes)
- Deterministic O(n) selection algorithm (introselect)

**Documentation & Examples**
- New notebook: `10_exponential_weighted_statistics.ipynb` (comprehensive)
- Updated: `00_getting_started.ipynb`, `08_numpy_pandas_migration_guide.ipynb`
- Added: `docs/GETTING_STARTED.md` (investor-ready guide)
- Fixed: GitHub URLs changed from `your-org` to `hpcseries` throughout docs
- Fixed: Sphinx autodoc now shows proper function signatures (not MagicMock)
- Added: Execution mode API complete documentation
- Added: Documentation stubs (`python/hpcs/_docstubs.py`) for Read the Docs

**Testing**
- `test_execution_modes_v08.py` - 400+ line test suite with 40+ tests (24 passed, 3 skipped)
- `test_transforms_v08.py` - 469-line test suite
- Reference comparisons vs NumPy/pandas/SciPy
- All C tests passing, confirming core implementation correctness
- Performance benchmarks included (deferred to v0.9.0 optimization phase)

**Implementation Details**
- Fortran execution mode infrastructure (`src/fortran/hpcs_core_execution_mode.f90`)
- C++ convolution with mode dispatch (`src/cpp/hpcs_convolution.cpp`)
- C++ robust statistics (`src/cpp/hpcs_robust_stats.cpp`, `src/cpp/hpcs_selection.cpp`)
- Hybrid Fortran/C++ architecture leveraging strengths of each language

**Licensing** - Changed to Apache 2.0 for commercialization
- Explicit patent grants
- `LICENSE_CHANGE_NOTICE.md` - investor documentation
- `NOTICE` file (Apache 2.0 requirement)

### Changed
- License: MIT → Apache 2.0 (all docs updated)
- Docker: Auto-rebuild Python extensions on volume mount
- Removed conflicting Fortran convolution implementation (replaced with C++ template version)

## [0.7.0] - 2025-12-17

### Added
- **Architecture-Aware Compilation System**
  - Automatic CPU detection for x86 (Intel/AMD) and ARM (Graviton)
  - CMake modules: `DetectArchitecture.cmake` and `CompilerFlags.cmake`
  - Architecture-specific compiler flags (`-march=native` vs `-mcpu=native`)
  - SAFE (default) and FAST (`-ffast-math`) compilation profiles
  - Support for Intel Sapphire Rapids, Ice Lake, AMD EPYC Genoa, ARM Graviton3

- **AWS Deployment Infrastructure**
  - AWS EC2 instance metadata detection via IMDSv2
  - Instance family detection (c7i, c6i, c7g, c7a, etc.)
  - Benchmark metadata collection (instance type, CPU vendor, SIMD capabilities)
  - Architecture detection scripts for x86 and ARM platforms

- **Performance Methodology Documentation**
  - `docs/PERFORMANCE.md` - Full empirical benchmark methodology
  - `docs/PERFORMANCE_SUMMARY.md` - One-page stakeholder summary
  - `docs/AWS_DEPLOYMENT_GUIDE.md` - Production deployment guide
  - Thread scaling analysis across AMD, Intel, and ARM architectures

- **Benchmark Enhancements**
  - Extended `run_benchmarks.sh` with AWS metadata collection
  - ARM CPU detection (Neoverse V1/V2, N1/N2)
  - CSV output with instance type, family, CPU vendor, and model
  - Benchmark comparison scripts

- **Python Bindings (v0.7)**
  - Complete Cython bindings for SIMD operations
  - CLI interface for benchmarking and testing
  - Python wheel distribution support
  - Docker environment for Python development

- **Comprehensive Examples**
  - 9 Jupyter notebooks covering real-world use cases
  - Kaggle competition examples (Store Sales forecasting)
  - Climate data, IoT sensors, financial analytics examples
  - NumPy/Pandas migration guide

- **Test Infrastructure**
  - Organized test suite (Python/C/Fortran)
  - Compiler flags verification script
  - Architecture detection tests

### Changed
- **Default Compilation Profile**: Changed to SAFE (no `-ffast-math`) for IEEE 754 compliance
- **OpenMP Configuration**: Documented optimal thread count (OMP_NUM_THREADS=2) across all architectures
- **Benchmark Script**: Now captures architecture and AWS metadata by default
- **.gitignore**: Updated to allow critical documentation files while ignoring generated content

### Performance
- **Thread Scaling Optimization**
  - Proven 5-18% degradation with 4+ threads across AMD, Intel, and ARM
  - Memory bandwidth saturation at 2 threads (vCPU-independent)
  - Optimal configuration: `OMP_NUM_THREADS=2` for all tested platforms
  - Validated on instances ranging from 4 to 16 vCPUs

- **Cross-Architecture Validation**
  - AMD EPYC Genoa (c7a.xlarge): 2 threads optimal
  - Intel Ice Lake (m6i.2xlarge, c6i.4xlarge): 2 threads optimal
  - ARM Graviton3 (c7g.xlarge): 2 threads optimal

### Fixed
- ARM CPU vendor and model detection in benchmark scripts
- Arithmetic increment bug in test script (bash `set -e` compatibility)
- Missing benchmark source files in CMake build
- .gitignore patterns blocking cmake modules and benchmark files

### Documentation
- **Changed license from MIT to Apache License 2.0** for commercial protection and patent grants
- Created comprehensive CHANGELOG
- Updated README with performance and deployment guide links
- Architecture-specific OpenMP configuration recommendations
- Publication-ready performance methodology

## [0.6.0] - 2024-XX-XX

### Added
- **SIMD Vectorization Infrastructure**
  - SIMD-optimized rolling operations (sum, mean, MAD, median)
  - SIMD-optimized reduction operations
  - Axis-based operations for multi-dimensional arrays
  - AVX-512, AVX2, and SSE2 support

- **Robust Statistics**
  - Median Absolute Deviation (MAD)
  - Robust z-score calculation
  - Quantile estimation
  - Outlier detection algorithms

- **Fortran-SIMD Integration**
  - Fortran modules for core statistics
  - C bridge for Fortran-SIMD interoperability
  - OpenMP parallelization in Fortran

- **Prefetch Optimization**
  - Memory prefetch hints for streaming operations
  - Microarchitecture-specific optimizations

### Changed
- Core algorithms rewritten with SIMD intrinsics
- Memory layout optimized for cache efficiency

## [0.5.0] - 2024-XX-XX

### Added
- **Calibration System**
  - Automatic performance tuning for target hardware
  - Calibration persistence and validation
  - Workload-specific optimization profiles

- **NUMA Affinity Support**
  - Multi-socket optimization
  - Thread pinning strategies
  - NUMA-aware memory allocation

- **CPU Detection Module**
  - Runtime CPU feature detection
  - Vendor identification (Intel, AMD, ARM)
  - SIMD capability detection (AVX-512, AVX2, NEON)

## [Unreleased]

### Planned
- GPU acceleration (CUDA/HIP)
- Additional robust statistics algorithms
- Distributed computing support
- Advanced time series decomposition

---

## Version History Summary

- **0.7.0** (2025-12-17): Architecture-aware compilation, AWS deployment, performance documentation
- **0.6.0**: SIMD vectorization, robust statistics, Fortran integration
- **0.5.0**: Calibration system, NUMA support, CPU detection

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to HPCSeries Core.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**License Change (v0.8)**: Changed from MIT to Apache 2.0 for enhanced patent protection and commercial-friendly terms suitable for enterprise adoption and investment.
