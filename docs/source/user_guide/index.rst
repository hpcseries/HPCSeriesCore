User Guide
==========

Comprehensive guides for using HPCSeries Core effectively.

.. toctree::
   :maxdepth: 2

   architecture
   performance
   migration

Architecture Overview
---------------------

HPCSeries Core is a hybrid Fortran/C/C++ library with Python bindings:

**Performance Stack**:

.. code-block:: text

   ┌─────────────────────────────────────┐
   │  Python API (Zero-Copy)             │  ← User Interface
   ├─────────────────────────────────────┤
   │  Cython Bindings                    │  ← NumPy Integration
   ├─────────────────────────────────────┤
   │  C/C++ Orchestration Layer          │  ← SIMD Dispatch
   ├─────────────────────────────────────┤
   │  Fortran HPC Kernels + C SIMD       │  ← Core Algorithms
   ├─────────────────────────────────────┤
   │  SIMD Instructions (AVX2/AVX-512)   │  ← Hardware Acceleration
   ├─────────────────────────────────────┤
   │  OpenMP Parallelization             │  ← Multi-threading
   └─────────────────────────────────────┘

Key Design Principles
~~~~~~~~~~~~~~~~~~~~~

1. **Zero-Copy**: Direct NumPy array access without data copying
2. **SIMD Vectorization**: Automatic use of AVX2/AVX-512 when available
3. **OpenMP Parallelization**: Multi-threaded for large datasets
4. **Adaptive Tuning**: Auto-calibration for optimal performance
5. **Robust Statistics**: MAD-based methods resistant to outliers

Contents
--------

- :doc:`architecture` - Detailed architecture and design decisions
- :doc:`performance` - Performance tuning and benchmarking
- :doc:`migration` - Migrating from NumPy/Pandas
