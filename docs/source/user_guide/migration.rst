Migration Guide
===============

This guide helps you migrate existing NumPy and Pandas code to HPCSeries Core for better performance.

From NumPy
----------

Basic Reductions
~~~~~~~~~~~~~~~~

**NumPy → HPCSeries** (2-5x faster, drop-in replacement):

.. code-block:: python

   # NumPy
   np.sum(x)
   np.mean(x)
   np.var(x)
   np.std(x)
   np.min(x)
   np.max(x)

   # HPCSeries (same signature)
   hpcs.sum(x)
   hpcs.mean(x)
   hpcs.var(x)
   hpcs.std(x)
   hpcs.min(x)
   hpcs.max(x)

Quantiles & Median
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # NumPy
   np.median(x)
   np.percentile(x, 25)

   # HPCSeries
   hpcs.median(x)
   hpcs.quantile(x, 0.25)  # Note: 0-1 scale, not 0-100

Z-Score Normalization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # NumPy (multi-pass)
   z_scores = (x - x.mean()) / x.std()

   # HPCSeries (single-pass, faster)
   z_scores = hpcs.zscore(x)

Axis Operations
~~~~~~~~~~~~~~~

.. code-block:: python

   # NumPy
   col_means = np.mean(data, axis=0)
   col_medians = np.median(data, axis=0)

   # HPCSeries (faster)
   col_means = hpcs.axis_mean(data)
   col_medians = hpcs.axis_median(data)

Masked Arrays
~~~~~~~~~~~~~

.. code-block:: python

   # NumPy masked arrays
   import numpy.ma as ma
   masked = ma.masked_invalid(x)
   mean_np = ma.mean(masked)

   # HPCSeries
   mask = ~np.isnan(x)
   mask_int = mask.astype(np.int32)
   mean_hpcs = hpcs.mean_masked(x, mask_int)

From Pandas
-----------

Rolling Operations
~~~~~~~~~~~~~~~~~~

**Pandas → HPCSeries** (50-100x faster):

.. code-block:: python

   # Pandas (slow)
   df['value'].rolling(window=50).mean()
   df['value'].rolling(window=50).std()
   df['value'].rolling(window=100).median()

   # HPCSeries (50-100x faster)
   hpcs.rolling_mean(data, window=50)
   hpcs.rolling_std(data, window=50)
   hpcs.rolling_median(data, window=100)

**Important differences**:

1. **Return type**: Pandas returns ``Series`` (with index), HPCSeries returns ``ndarray``
2. **Window alignment**: Both use right-aligned windows by default
3. **min_periods**: HPCSeries always uses full window (returns NaN for first ``window-1`` elements)

GroupBy Operations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Pandas groupby
   df.groupby('group')['value'].mean()

   # HPCSeries: Use axis operations on reshaped array
   data_2d = df['value'].values.reshape(n_groups, group_size)
   group_means = hpcs.axis_mean(data_2d)

Hybrid Pandas + HPCSeries
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best practice**: Use Pandas for data manipulation, HPCSeries for heavy computation:

.. code-block:: python

   # Use Pandas for I/O and data prep
   df = pd.read_csv('data.csv')
   df = df.dropna()

   # Use HPCSeries for computation
   df['rolling_mean'] = hpcs.rolling_mean(df['price'].values, window=50)
   df['anomaly'] = hpcs.detect_anomalies_robust(df['price'].values)

   # Use Pandas for output
   df.to_csv('results.csv')

When to Use Each
~~~~~~~~~~~~~~~~

**Use Pandas when you need**:
  - Time-aware operations (resample, date offsets)
  - GroupBy with string keys
  - Join/merge operations
  - DataFrame manipulation (pivot, melt)
  - Label alignment

**Use HPCSeries when you need**:
  - Maximum performance for numerical operations
  - Large-scale rolling operations (> 10K elements)
  - Real-time/streaming computation
  - Low-latency requirements (< 1ms)
  - Memory-efficient computation

API Compatibility Matrix
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Operation
     - NumPy/Pandas
     - HPCSeries
     - Notes
   * - ``sum``
     - ``np.sum(x)``
     - ``hpcs.sum(x)``
     - ✓ Drop-in
   * - ``mean``
     - ``np.mean(x)``
     - ``hpcs.mean(x)``
     - ✓ Drop-in
   * - ``std``
     - ``np.std(x)``
     - ``hpcs.std(x)``
     - ✓ Drop-in
   * - ``var``
     - ``np.var(x)``
     - ``hpcs.var(x)``
     - ✓ Drop-in
   * - ``min``
     - ``np.min(x)``
     - ``hpcs.min(x)``
     - ✓ Drop-in
   * - ``max``
     - ``np.max(x)``
     - ``hpcs.max(x)``
     - ✓ Drop-in
   * - ``median``
     - ``np.median(x)``
     - ``hpcs.median(x)``
     - ✓ Drop-in
   * - ``quantile``
     - ``np.percentile(x, q)``
     - ``hpcs.quantile(x, q/100)``
     - ⚠ 0-1 scale
   * - ``rolling mean``
     - ``df.rolling(w).mean()``
     - ``hpcs.rolling_mean(x, w)``
     - ⚠ Returns array
   * - ``rolling std``
     - ``df.rolling(w).std()``
     - ``hpcs.rolling_std(x, w)``
     - ⚠ Returns array
   * - ``rolling median``
     - ``df.rolling(w).median()``
     - ``hpcs.rolling_median(x, w)``
     - ⚠ Returns array
   * - ``axis mean``
     - ``np.mean(x, axis=0)``
     - ``hpcs.axis_mean(x)``
     - ✓ axis=0 default
   * - ``masked mean``
     - ``ma.mean(masked)``
     - ``hpcs.mean_masked(x, mask)``
     - ⚠ Different API

Common Migration Patterns
--------------------------

Pattern 1: Time Series Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import hpcs

   # Load time series
   df = pd.read_csv('stock_prices.csv', parse_dates=['date'])
   prices = df['close'].values

   # Compute features with HPCSeries (fast)
   df['sma_20'] = hpcs.rolling_mean(prices, window=20)
   df['sma_50'] = hpcs.rolling_mean(prices, window=50)
   df['volatility'] = hpcs.rolling_std(prices, window=20)
   df['anomaly'] = hpcs.detect_anomalies_robust(prices, threshold=3.0)

Pattern 2: Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import hpcs

   df = pd.read_csv('data.csv')
   values = df['value'].values

   # Multiple rolling features (fast)
   df['rm_10'] = hpcs.rolling_mean(values, window=10)
   df['rm_50'] = hpcs.rolling_mean(values, window=50)
   df['rs_10'] = hpcs.rolling_std(values, window=10)
   df['rmed_20'] = hpcs.rolling_median(values, window=20)
   df['zscore'] = hpcs.zscore(values)
   df['robust_zscore'] = hpcs.robust_zscore(values)

Pattern 3: Multi-Sensor Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import hpcs

   # Load multi-sensor data
   df = pd.read_csv('sensors.csv')
   sensor_cols = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
   data_2d = df[sensor_cols].values  # Shape: (n_samples, 4)

   # Per-sensor statistics (vectorized)
   sensor_means = hpcs.axis_mean(data_2d)
   sensor_medians = hpcs.axis_median(data_2d)
   sensor_mad = hpcs.axis_mad(data_2d)

   # Detect anomalies per sensor
   anomalies_2d = hpcs.anomaly_robust_axis(data_2d, threshold=3.0)

Migration Checklist
-------------------

When migrating NumPy/Pandas code:

1. **Check array dtype**
   - HPCSeries uses ``float64``
   - Convert if needed: ``x.astype(np.float64)``

2. **Ensure C-contiguous layout**
   - Check: ``x.flags['C_CONTIGUOUS']``
   - Fix: ``np.ascontiguousarray(x)``

3. **Extract values from Pandas**
   - Use ``.values`` to get NumPy array from Series/DataFrame
   - Example: ``df['column'].values``

4. **Handle return types**
   - HPCSeries returns NumPy arrays, not Pandas Series
   - Assign back to DataFrame if needed: ``df['new_col'] = result``

5. **Adjust quantile scale**
   - NumPy: 0-100 scale (``np.percentile(x, 25)``)
   - HPCSeries: 0-1 scale (``hpcs.quantile(x, 0.25)``)

6. **Handle NaN differences**
   - Pandas ``min_periods`` not supported
   - Use masked operations for missing data

Key Differences Summary
-----------------------

Return Types
~~~~~~~~~~~~

- **Pandas**: Returns ``Series`` or ``DataFrame`` (preserves index/labels)
- **HPCSeries**: Returns ``ndarray`` (raw arrays for performance)

NaN Handling
~~~~~~~~~~~~

- **Pandas**: Configurable with ``min_periods`` parameter
- **HPCSeries**: Always uses full window; first ``(window-1)`` elements are NaN
- For missing data: Use ``*_masked()`` functions

Quantile Scale
~~~~~~~~~~~~~~

- **NumPy**: ``percentile(x, 25)`` uses 0-100 scale
- **HPCSeries**: ``quantile(x, 0.25)`` uses 0-1 scale

Window Alignment
~~~~~~~~~~~~~~~~

- **Pandas**: Supports ``center=True/False``
- **HPCSeries**: Right-aligned by default (same as Pandas default)
- For centered windows: Manually shift results

Performance Expectations
------------------------

Typical speedups when migrating:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Operation
     - Array Size
     - Typical Speedup
   * - Basic reductions (sum, mean, std)
     - 1M elements
     - 2-5x
   * - Rolling mean/std
     - 100K elements
     - 50-100x
   * - Rolling median
     - 100K elements
     - 100-200x
   * - Axis operations
     - 1000×100
     - 3-8x
   * - Robust statistics (median, MAD)
     - 1M elements
     - 1.5-3x

**Best speedups**: Rolling operations on large datasets (> 10K elements)

Migration Strategy
------------------

Incremental Approach
~~~~~~~~~~~~~~~~~~~~

You don't need to replace everything at once:

1. **Profile**: Identify performance bottlenecks using ``cProfile`` or timing
2. **Replace hot paths**: Migrate the slowest operations first
3. **Validate**: Compare results between Pandas and HPCSeries (``np.allclose()``)
4. **Benchmark**: Measure speedup on realistic data sizes

Gradual Migration
~~~~~~~~~~~~~~~~~

Combine Pandas and HPCSeries strengths:

.. code-block:: python

   import pandas as pd
   import hpcs

   # Keep Pandas for what it does best
   df = pd.read_csv('data.csv')
   df = df.dropna()

   # Use HPCSeries for performance-critical operations
   df['rolling_mean'] = hpcs.rolling_mean(df['value'].values, window=50)

   # Keep Pandas for output and time-based operations
   df_hourly = df.resample('1H').last()
   df_hourly.to_csv('output.csv')

See Also
--------

- **Examples**: :doc:`../notebooks/index` - Notebook 08 has detailed migration examples
- **API Reference**: :doc:`../api/index` - Complete function signatures
- **Performance Guide**: :doc:`performance` - Optimization tips
