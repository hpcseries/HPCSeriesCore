HPCSeries Core Documentation
=============================

**Version:** 0.7.0

HPCSeries Core is a high-performance statistical computing library with SIMD vectorization,
OpenMP parallelization, and adaptive auto-tuning. Built with Fortran, C, and C++ for maximum
performance, with zero-copy Python bindings via Cython.

Key Features
------------

* **SIMD-Accelerated Operations**: Automatic vectorization using AVX2/AVX-512/SSE2
* **OpenMP Parallelization**: Multi-threaded operations for large datasets
* **Zero-Copy Integration**: Direct NumPy array access without data copying
* **Adaptive Auto-Tuning**: Automatic calibration for optimal performance
* **Robust Statistics**: MAD-based outlier detection and robust z-scores
* **Rolling Operations**: Fast sliding window computations
* **Anomaly Detection**: Statistical and robust anomaly detection methods

Performance
-----------

* 2-10x faster than NumPy for common operations
* 50-100x faster than Pandas for rolling operations
* Sub-microsecond latency for small arrays
* Scales to billions of elements

Quick Start
-----------

Installation:

.. code-block:: bash

   pip install hpcs

Basic usage:

.. code-block:: python

   import hpcs
   import numpy as np

   # Create sample data
   x = np.random.randn(1000000)

   # SIMD-accelerated reductions
   hpcs.sum(x)      # 2-5x faster than np.sum
   hpcs.mean(x)     # 2-5x faster than np.mean
   hpcs.std(x)      # 2-5x faster than np.std

   # Rolling operations
   hpcs.rolling_mean(x, window=50)    # 50-100x faster than pandas
   hpcs.rolling_median(x, window=100) # 100x faster than pandas

   # Robust statistics and anomaly detection
   hpcs.median(x)
   hpcs.mad(x)
   anomalies = hpcs.detect_anomalies(x, threshold=3.0)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   notebooks/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
