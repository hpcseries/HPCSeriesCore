Architecture
============

HPCSeries Core v0.7 — Logical Architecture
-------------------------------------------

Architectural Intent
--------------------

HPCSeries Core v0.7 is a **CPU-first, high-performance numeric kernel engine** designed to provide:

- Deterministic, cache-efficient time-series and array analytics
- Robust statistics resilient to outliers and missing data
- SIMD-vectorized and OpenMP-parallel execution paths
- Stable C ABI with Fortran and C++ implementations
- Python bindings for data-science workflows

The architecture is **layered**, **composable**, and **backend-agnostic** (GPU intentionally excluded from v1.x).

High-Level Layer Stack
----------------------

.. code-block:: text

   ┌─────────────────────────────────────────────┐
   │ Python API (v0.7)                            │
   │  - hpcs.sum, mean, std                      │
   │  - rolling_*                                │
   │  - robust_*                                 │
   │  - masked_*                                 │
   │  - axis_*                                   │
   └─────────────────────────────────────────────┘
                       │
   ┌─────────────────────────────────────────────┐
   │ Stable C ABI (hpcs_core.h)                  │
   │  - ISO_C_BINDING compliant                  │
   │  - status-out error handling                │
   │  - versioned, frozen for v1.x               │
   └─────────────────────────────────────────────┘
                       │
   ┌─────────────────────────────────────────────┐
   │ C++ High-Performance Extensions             │
   │  - Fast rolling median / MAD (O(n log w))   │
   │  - STL-based heaps / multisets              │
   │  - Used where asymptotics beat SIMD         │
   └─────────────────────────────────────────────┘
                       │
   ┌─────────────────────────────────────────────┐
   │ Fortran Numeric Kernel Core                 │
   │  - SIMD inner loops (AVX2)                  │
   │  - OpenMP parallel variants                 │
   │  - Masked, robust, axis-aware kernels       │
   └─────────────────────────────────────────────┘
                       │
   ┌─────────────────────────────────────────────┐
   │ Foundational Utilities & Constants          │
   │  - Status codes                             │
   │  - NaN/Inf handling                         │
   │  - Memory-safe helpers                     │
   └─────────────────────────────────────────────┘

Fortran Kernel Subsystems
--------------------------

Foundational Layer
~~~~~~~~~~~~~~~~~~

**Purpose**: Shared infrastructure used by all kernels.

**Modules**:

- ``hpcs_constants``

  - Status codes: SUCCESS, INVALID_ARGS, NUMERIC_FAILURE
  - Precision definitions

- ``hpcs_core_utils``

  - fill, copy, where
  - forward/backward fill
  - min–max normalization

This layer has **no dependencies** except the Fortran runtime.

1D Time-Series Kernels
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Fast, cache-friendly analytics on single vectors.

**Modules**:

- ``hpcs_core_1d``

  - rolling_sum, rolling_mean, rolling_variance, rolling_std
  - zscore

- ``hpcs_core_reductions``

  - reduce_sum / min / max / mean / variance / std

- ``hpcs_core_prefix``

  - prefix_sum (inclusive / exclusive)

**Characteristics**:

- Tight loops
- SIMD-vectorized
- Optional OpenMP variants
- Deterministic behavior

Robust Statistics Layer
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Outlier-resilient analytics.

**Modules**:

- ``hpcs_core_stats``

  - median (quickselect)
  - MAD
  - quantile

- ``hpcs_core_rolling``

  - rolling_median
  - rolling_mad

- ``hpcs_core_quality``

  - robust_zscore
  - winsorization
  - clipping

**Design principle**: Robust stats **build on** basic reductions but never mutate core primitives.

Masked Operations Layer
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Correct analytics in the presence of missing or invalid data.

**Modules**:

- ``hpcs_core_masked``

  - masked reductions
  - masked rolling means
  - masked robust stats

**Key behavior**:

- Explicit validity mask
- Propagates NaNs correctly
- Returns numeric failure when no valid data exists

2D & Axis-Aware Kernels
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Operate on matrices without materializing copies.

**Modules**:

- ``hpcs_core_batched``

  - column-wise independent 1D processing

- ``hpcs_core_axis``

  - axis-0 (column) reductions
  - axis-1 (row) reductions

**Design choice**:

- Axis loops outside, SIMD inside
- No implicit transposes
- Cache-aware traversal

Anomaly Detection Layer
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Structured anomaly detection built from primitives.

**Modules**:

- ``hpcs_core_anomaly``

  - detect_anomalies (mean/std)
  - detect_anomalies_robust (median/MAD)
  - rolling anomaly density
  - 2D rolling anomaly detection

**Important**: This layer contains **no novel math**—it is a *composition layer*.

Parallel & Vectorized Execution
--------------------------------

SIMD (v0.6+)
~~~~~~~~~~~~

- Explicit AVX2 inner loops
- 4 doubles per vector
- Used in:

  - reductions
  - rolling sums
  - z-scores
  - masked ops

OpenMP
~~~~~~

- Thread-level parallelism:

  - large reductions
  - rolling windows over long arrays

- Deterministic reduction variants preserved where required

.. note::

   **SIMD ≠ OpenMP**: SIMD accelerates **inside a core**, OpenMP scales **across cores**. Both are used where appropriate.

C++ Fast-Path Kernels
----------------------

**Why C++ exists in v0.7**: Some algorithms (e.g., rolling median) are **asymptotically better** with tree/heap structures.

**Examples**:

- ``hpcs_rolling_median_fast``
- ``hpcs_rolling_mad_fast``

These:

- Beat SIMD at large window sizes
- Are wrapped behind the same C ABI
- Are invisible to Python users

Stable C ABI Layer
------------------

**Design principles**:

- ``bind(C)`` everywhere
- No return values (status via pointer)
- ABI frozen for v1.x

This enables:

- Python bindings
- C++ wrappers
- Future Rust / Julia bindings
- Embedding in non-Python systems

Python Binding Layer (v0.7)
----------------------------

**What is exposed**:

- Core reductions
- Rolling statistics
- Robust statistics
- Masked & axis operations
- Fast rolling kernels

**What is NOT exposed**:

- Internal helpers
- Experimental tuning hooks
- Debug utilities

Python is **a consumer**, not the owner, of the architecture.

What HPCSeries Core IS (and Is Not)
------------------------------------

It IS
~~~~~

- A high-performance numeric kernel engine
- A foundation for domain-specific cores
- A CPU-optimized analytics backend
- A "numerical brick" for serious systems

It is NOT
~~~~~~~~~

- A BLAS replacement
- A NumPy clone
- A GPU framework (yet)
- A modeling / ML library

Architectural Stability
-----------------------

At v0.7:

- The **core architecture is stable**
- Future versions add **breadth, not depth**
- Domain engines sit **above**, not inside, the core
- GPU work (if any) becomes a **parallel backend**, not a rewrite

Design Decisions
----------------

Why Hybrid Fortran/C/C++?
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Fortran**: Best for array operations, excellent OpenMP support
- **C**: SIMD intrinsics, system-level control, portable C ABI
- **C++**: Modern features for complex rolling algorithms (heaps, multisets)
- **Python/Cython**: User-friendly API, NumPy integration

Why Runtime SIMD Dispatch?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Single binary works on all CPUs (SSE2 to AVX-512)
- Automatically uses best available instructions
- No need for multiple builds

Why Stable C ABI?
~~~~~~~~~~~~~~~~~

- Enables bindings to any language
- Prevents ABI breakage across versions
- Separates interface from implementation

Why No GPU in v1.x?
~~~~~~~~~~~~~~~~~~~

- CPU-first architecture is simpler and more deterministic
- GPU support would be a parallel backend, not a rewrite
- Focuses resources on CPU optimization first
- GPU may be added as backend in future versions

Zero-Copy Design
----------------

HPCSeries never copies NumPy array data:

- NumPy array in Python → pointer to data
- Cython receives same pointer
- C receives same pointer
- Fortran receives same pointer (via C bridge)
- **No memory copies!**

This is essential for performance with large arrays.

Thread Safety
-------------

HPCSeries is **thread-safe** for read operations:

- Multiple threads can call functions simultaneously
- No global mutable state
- OpenMP handles internal parallelization
- GIL released during C/Fortran execution

**Not thread-safe**:

- Calibration (should be run once at startup)
- Configuration saving/loading (serialize these operations)

See Also
--------

- :doc:`performance` - Performance optimization guide
- :doc:`migration` - Migrating from NumPy/Pandas
- API Reference: :doc:`../api/index`
