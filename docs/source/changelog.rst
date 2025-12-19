Changelog
=========

All notable changes to HPCSeries Core will be documented here.

Version 0.7.0 (2025-01-XX)
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

Version 0.6.0 (2024-12-XX)
--------------------------

Features
~~~~~~~~

- SIMD dispatch system for runtime ISA detection
- Fortran-C bridge for SIMD operations
- Axis reductions (per-row/per-column statistics)
- Masked reductions for missing data
- Rolling operations with batching support

Version 0.5.0 (2024-11-XX)
--------------------------

Features
~~~~~~~~

- Performance calibration system
- Configuration persistence (~/.hpcs/config.json)
- CPU detection (cores, cache, NUMA)
- Adaptive auto-tuning

Version 0.4.0 (2024-10-XX)
--------------------------

Features
~~~~~~~~

- 2D array operations (axis reductions)
- Batched rolling operations
- Masked array support
- Quality indicators

Version 0.3.0 (2024-09-XX)
--------------------------

Features
~~~~~~~~

- Rolling operations (mean, median, std, MAD)
- Robust statistics (median, MAD, quantile)
- Z-score normalization
- C++ optimized rolling implementations

Version 0.2.0 (2024-08-XX)
--------------------------

Features
~~~~~~~~

- Fortran HPC kernels for core reductions
- OpenMP parallelization
- Anomaly detection

Version 0.1.0 (2024-07-XX)
--------------------------

Initial Release
~~~~~~~~~~~~~~~

- Basic reduction operations (sum, mean, var, std, min, max)
- Fortran implementation
- C/Python bindings
