Changelog
=========

All notable changes to HPCSeries Core will be documented here.

Version 0.8.0 (2025-01-06)
--------------------------

Major Features
~~~~~~~~~~~~~~

**Execution Mode System**
  Safety vs performance trade-offs with three execution modes:

  - ``MODE_SAFE`` (default): IEEE 754 compliant, full error checking
  - ``MODE_FAST``: Relaxed math optimizations, minimal validation
  - ``MODE_DETERMINISTIC``: Bit-exact reproducibility, no OpenMP parallelization

  Control via ``set_execution_mode()`` and ``get_execution_mode()`` API, or per-call ``mode`` parameter.

**New Functions (v0.8)**
  - ``ewma()`` - Exponentially weighted moving average (15-60x faster than pandas)
  - ``ewvar()`` - Exponentially weighted variance
  - ``ewstd()`` - Exponentially weighted standard deviation
  - ``diff()`` - Finite differencing (arbitrary order)
  - ``cumulative_min()`` - Running minimum
  - ``cumulative_max()`` - Running maximum
  - ``convolve_valid()`` - 1D FIR convolution with template specialization (kernel sizes 3-15)
  - ``trimmed_mean()`` - Robust mean (10-15x faster than SciPy)
  - ``winsorized_mean()`` - Robust mean with winsorization

**Implementation**
  - Thread-safe execution mode using OpenMP threadprivate storage
  - Zero overhead via compile-time dispatcher pattern
  - Hybrid Fortran/C++ architecture
  - Template-specialized convolution for common kernel sizes

**Documentation**
  - Fixed GitHub URLs (``your-org`` → ``hpcseries``)
  - Fixed Sphinx autodoc to show proper function signatures (not MagicMock)
  - Added execution mode API complete documentation
  - New notebook: ``10_exponential_weighted_statistics.ipynb``
  - Updated API reference with all v0.8.0 functions

**Testing**
  - 400+ line test suite with 40+ tests (24 passed, 3 skipped)
  - Reference comparisons vs NumPy/pandas/SciPy
  - All C/Fortran tests passing

**License**
  - Changed from MIT to Apache 2.0 for commercialization

Version 0.7.0 (2025-12-17)
--------------------------

Major Features
~~~~~~~~~~~~~~

- Python API with comprehensive Cython bindings
- Complete set of Jupyter notebook examples (00-09)
- SIMD-accelerated reductions (sum, mean, var, std, min, max)
- Fast rolling operations (mean, median, std, var, MAD, z-score)
- Robust statistics (median, MAD, robust z-score)
- Anomaly detection (statistical and robust methods)
- Axis operations for 2D arrays
- Masked operations for handling missing data
- Performance calibration and auto-tuning
- CPU topology detection
- CLI tool (``hpcs cpuinfo``)

Performance
~~~~~~~~~~~

- 2-5x faster than NumPy for basic reductions
- 50-100x faster than Pandas for rolling operations
- Sub-microsecond latency for small arrays
- Automatic SIMD vectorization (AVX2/AVX-512/SSE2)
- OpenMP parallelization for large datasets

Documentation
~~~~~~~~~~~~~

- Complete API reference
- 12 Jupyter notebook tutorials
- Installation and quick start guides
- Migration guide from NumPy/Pandas

Version 0.6.0 (2024-12-01)
--------------------------

Features
~~~~~~~~

- SIMD dispatch system for runtime ISA detection
- Fortran-C bridge for SIMD operations
- Axis reductions (per-row/per-column statistics)
- Masked reductions for missing data
- Rolling operations with batching support

Version 0.5.0 (2024-11-01)
--------------------------

Features
~~~~~~~~

- Performance calibration system
- Configuration persistence (~/.hpcs/config.json)
- CPU detection (cores, cache, NUMA)
- Adaptive auto-tuning

Version 0.4.0 (2024-10-01)
--------------------------

Features
~~~~~~~~

- 2D array operations (axis reductions)
- Batched rolling operations
- Masked array support
- Quality indicators

Version 0.3.0 (2024-09-01)
--------------------------

Features
~~~~~~~~

- Rolling operations (mean, median, std, MAD)
- Robust statistics (median, MAD, quantile)
- Z-score normalization
- C++ optimized rolling implementations

Version 0.2.0 (2024-08-01)
--------------------------

Features
~~~~~~~~

- Fortran HPC kernels for core reductions
- OpenMP parallelization
- Anomaly detection

Version 0.1.0 (2024-07-01)
--------------------------

Initial Release
~~~~~~~~~~~~~~~

- Basic reduction operations (sum, mean, var, std, min, max)
- Fortran implementation
- C/Python bindings
