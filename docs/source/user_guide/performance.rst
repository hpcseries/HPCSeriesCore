Performance Guide
=================

This guide explains how to achieve optimal performance with HPCSeries Core.

Performance Characteristics
---------------------------

Speedup vs NumPy
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Operation
     - Array Size
     - Speedup
     - Notes
   * - ``sum/mean/std``
     - 1M elements
     - 2-5x
     - SIMD-accelerated
   * - ``rolling_mean``
     - 100K elements
     - 50-100x
     - vs Pandas rolling
   * - ``rolling_median``
     - 100K elements
     - 100-200x
     - vs Pandas rolling
   * - ``median``
     - 1M elements
     - 1.5-2x
     - Quickselect algorithm
   * - ``mad``
     - 1M elements
     - 2-3x
     - Two-pass algorithm

Latency Characteristics
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Array Size
     - Latency
     - Recommendation
   * - < 100
     - < 1 µs
     - Use HPCSeries (minimal overhead)
   * - 100 - 10K
     - 1-100 µs
     - HPCSeries excels here
   * - 10K - 1M
     - 0.1-10 ms
     - SIMD provides 2-5x benefit
   * - > 1M
     - > 10 ms
     - OpenMP parallelization kicks in

Calibration
-----------

What is Calibration?
~~~~~~~~~~~~~~~~~~~~

Calibration is a **one-time auto-tuning process** (~30 seconds) that determines optimal performance thresholds for your specific hardware. After calibration, HPCSeries saves a configuration file that is automatically loaded in future sessions.

**When to calibrate**:
  - First time setup on new hardware
  - After system upgrades (CPU, RAM)
  - If performance seems suboptimal

**Calibration functions**:
  - ``hpcs.calibrate()`` - Full calibration (~30 seconds)
  - ``hpcs.calibrate_quick()`` - Quick calibration (~5 seconds)
  - ``hpcs.save_calibration_config()`` - Save to ``~/.hpcs/config.json``
  - ``hpcs.load_calibration_config()`` - Manually load configuration

What Calibration Determines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration benchmarks your system to find optimal thresholds:

1. **SIMD Threshold** (typically 100-1000 elements)
   - Minimum array size where SIMD beats scalar operations
   - Depends on SIMD overhead and CPU clock speed

2. **OpenMP Threshold** (typically 10K-100K elements)
   - Minimum array size where multi-threading is beneficial
   - Depends on core count, thread overhead, cache size

3. **Block Sizes** for cache-friendly processing
   - L1 cache: ~4-8 KB blocks
   - L2 cache: ~64-256 KB blocks

4. **Rolling Window Optimizations**
   - Small windows: Direct computation
   - Large windows: Incremental updates

OpenMP Configuration
--------------------

Thread Count
~~~~~~~~~~~~

HPCSeries uses OpenMP for multi-threaded parallelization. Control thread count via the ``OMP_NUM_THREADS`` environment variable.

**Recommendations**:
  - **Desktop/Workstation**: Use physical core count (not logical/hyperthreaded)
  - **Server**: Test between 1x to 2x physical cores
  - **Single-threaded timing**: Set to 1 for consistent benchmarks

Thread Affinity
~~~~~~~~~~~~~~~~

Pin threads to cores for consistent performance using OpenMP affinity settings:

  - **Compact**: Fill one socket first (``OMP_PROC_BIND=close``)
  - **Spread**: Distribute across sockets (``OMP_PROC_BIND=spread``)
  - Set ``OMP_PLACES=cores`` for core-level binding

Scheduling
~~~~~~~~~~

Choose scheduling strategy based on workload:

  - **Dynamic** (default): Good for uneven workloads
  - **Static**: Better cache locality for uniform work

Array Layout Optimization
--------------------------

Contiguous Arrays
~~~~~~~~~~~~~~~~~

HPCSeries requires **C-contiguous arrays** (row-major layout) for optimal performance.

**Why contiguous arrays matter**:
  - SIMD instructions require aligned, contiguous memory
  - Prefetching works best with sequential access
  - Cache lines are filled efficiently

Check array layout with ``x.flags['C_CONTIGUOUS']``. Use ``np.ascontiguousarray(x)`` if needed.

Memory Alignment
~~~~~~~~~~~~~~~~

For best SIMD performance, arrays should be aligned to 32-byte boundaries (AVX2) or 64-byte boundaries (AVX-512). HPCSeries handles unaligned arrays but with reduced performance.

Data Types
~~~~~~~~~~

HPCSeries is optimized for ``float64`` (double precision). Other types are automatically converted with a small overhead:

  - **Best**: ``np.float64`` - no conversion
  - **Acceptable**: ``np.float32`` - converted to float64
  - **Avoid**: Integer types - conversion overhead

Benchmarking
------------

Accurate Timing Principles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For accurate benchmarks:

1. **Warm-up**: Run operation once before timing to load caches
2. **Multiple iterations**: Average over 100+ iterations
3. **Array size**: Use realistic data sizes (1K to 10M elements)
4. **Consistent environment**: Set ``OMP_NUM_THREADS`` to fixed value

**Tools**:
  - ``time.perf_counter()`` - High-resolution timing
  - ``timeit`` module - Automated benchmarking
  - ``cProfile`` - Detailed profiling

Performance Tips
----------------

1. **Minimize Python Overhead**
   - Use axis operations instead of Python loops
   - Batch operations when possible

2. **Batch Operations**
   - Stack multiple arrays and use ``axis_*`` functions
   - Reduces function call overhead

3. **Reuse Calibration**
   - Calibrate once, save configuration
   - Subsequent imports load automatically

4. **Choose Right Operation**
   - Standard operations (``mean``, ``std``) for clean data
   - Robust operations (``median``, ``mad``) for outlier-prone data

5. **Appropriate Window Sizes**
   - Small windows (< 100): Direct computation
   - Large windows: Consider downsampling or exponential moving average

Common Performance Issues
-------------------------

Slower Than Expected
~~~~~~~~~~~~~~~~~~~~

**Possible causes**:
  - Array not C-contiguous (check with ``x.flags['C_CONTIGUOUS']``)
  - Wrong dtype (convert to ``np.float64``)
  - Not calibrated (run ``hpcs.calibrate()``)
  - OpenMP not using all cores (set ``OMP_NUM_THREADS``)

High Latency for Small Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For arrays < 100 elements, function call overhead dominates. Consider using NumPy for very small arrays and HPCSeries for larger datasets.

Memory Bandwidth Bottleneck
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large arrays (> 100M elements), memory bandwidth limits speedup. Consider:
  - Processing in blocks
  - Fusing operations when possible
  - Using lower precision (future feature)

Performance Monitoring
----------------------

CPU Information
~~~~~~~~~~~~~~~

HPCSeries provides tools to inspect your hardware:

**Python API**:
  - ``hpcs.get_cpu_info()`` - Physical/logical cores, cache sizes
  - ``hpcs.simd_info()`` - Active SIMD ISA and vector width

**CLI Tool**:

.. code-block:: bash

   $ hpcs cpuinfo

Example output:

.. code-block:: text

   === CPU Information ===

   CPU Vendor:          GenuineIntel
   Physical Cores:      8
   Logical Cores:       16
   Optimal Threads:     8

   Cache Hierarchy:
     L1:      32 KB
     L2:     256 KB
     L3:   16384 KB

   SIMD Capabilities:
     Active ISA:          AVX2
     Vector width:        256-bit (4 doubles)
     AVX-512:             ✗
     AVX2:                ✓
     AVX:                 ✓
     SSE2:                ✓

Expected Performance
--------------------

Reference Benchmarks
~~~~~~~~~~~~~~~~~~~~

**AMD Ryzen 7 (8 cores, AVX2):**

.. code-block:: text

   Operation              Size        NumPy       HPCSeries   Speedup
   ----------------------------------------------------------------
   sum                    1M         0.45 ms      0.12 ms      3.8x
   mean                   1M         0.48 ms      0.13 ms      3.7x
   std                    1M         1.20 ms      0.35 ms      3.4x
   rolling_mean (w=50)    100K       45 ms        0.8 ms       56x
   rolling_median (w=50)  100K       850 ms       7.2 ms       118x
   median                 1M         12 ms        6.5 ms       1.8x

**Intel Xeon (16 cores, AVX-512):**

.. code-block:: text

   Operation              Size        NumPy       HPCSeries   Speedup
   ----------------------------------------------------------------
   sum                    1M         0.38 ms      0.08 ms      4.8x
   mean                   1M         0.41 ms      0.09 ms      4.6x
   std                    1M         1.05 ms      0.22 ms      4.8x
   rolling_mean (w=50)    100K       42 ms        0.6 ms       70x
   rolling_median (w=50)  100K       820 ms       6.8 ms       121x

Scaling Characteristics
~~~~~~~~~~~~~~~~~~~~~~~

**Array Size Scaling** (single-threaded):

.. code-block:: text

   Size      sum (ms)    Throughput (GB/s)
   ----------------------------------------
   1K        0.001       8.0
   10K       0.008       10.0
   100K      0.078       10.3
   1M        0.780       10.3
   10M       7.8         10.3

**Thread Scaling** (10M elements, AVX2):

.. code-block:: text

   Threads   sum (ms)    Speedup    Efficiency
   -------------------------------------------
   1         7.8         1.0x       100%
   2         4.1         1.9x       95%
   4         2.2         3.5x       88%
   8         1.3         6.0x       75%
   16        0.9         8.7x       54%

Advanced Optimization
---------------------

NUMA Awareness
~~~~~~~~~~~~~~

For multi-socket systems, consider NUMA (Non-Uniform Memory Access) topology:

  - Check topology with ``numactl --hardware``
  - Bind processes to specific NUMA nodes for consistent performance
  - See :doc:`../NUMA_AFFINITY_GUIDE` for detailed instructions

Huge Pages
~~~~~~~~~~

For very large arrays, Linux huge pages can improve performance:

  - Reduces TLB (Translation Lookaside Buffer) misses
  - Beneficial for arrays > 1GB
  - Requires system configuration

Custom Builds
~~~~~~~~~~~~~

For maximum performance, build with CPU-specific optimizations:

  - Use ``-march=native`` flag during compilation
  - Enables CPU-specific SIMD instructions
  - Optimizes cache sizes and instruction scheduling
  - See :doc:`../BUILD_AND_TEST` for build instructions

Troubleshooting Performance
----------------------------

Debug Mode Check
~~~~~~~~~~~~~~~~

Ensure HPCSeries was built in Release mode (not Debug):
  - Debug builds are 5-10x slower
  - Check build configuration in CMake

SIMD Verification
~~~~~~~~~~~~~~~~~

Verify SIMD is active:
  - Use ``hpcs.simd_info()`` to check active ISA
  - Should report AVX2, AVX-512, or at minimum SSE2
  - If reporting scalar operations, SIMD may not be enabled

Thread Utilization
~~~~~~~~~~~~~~~~~~~

Monitor CPU usage during operations:
  - Use ``htop`` or task manager
  - All cores should be active for large arrays (> 100K elements)
  - If not, check ``OMP_NUM_THREADS`` environment variable

See Also
--------

- :doc:`architecture` - System architecture and design
- :doc:`migration` - Migrating from NumPy/Pandas
- API Reference: :doc:`../api/index`
