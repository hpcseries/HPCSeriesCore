Quick Start Guide
=================

This guide will get you up and running with HPCSeries Core in minutes.

Basic Usage
-----------

Import and Simple Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hpcs
   import numpy as np

   # Create sample data
   x = np.random.randn(1000000)

   # Basic reductions (2-5x faster than NumPy)
   hpcs.sum(x)      # 500.123
   hpcs.mean(x)     # 0.00050
   hpcs.std(x)      # 1.0002
   hpcs.min(x)      # -4.8234
   hpcs.max(x)      # 4.7123

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time

   # NumPy
   start = time.perf_counter()
   result_np = np.mean(x)
   time_np = time.perf_counter() - start

   # HPCSeries (SIMD-accelerated)
   start = time.perf_counter()
   result_hpcs = hpcs.mean(x)
   time_hpcs = time.perf_counter() - start

   print(f"NumPy:     {time_np*1000:.2f} ms")
   print(f"HPCSeries: {time_hpcs*1000:.2f} ms")
   print(f"Speedup:   {time_np/time_hpcs:.1f}x")

Rolling Operations
------------------

Fast Sliding Window Computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HPCSeries provides rolling operations that are 50-100x faster than Pandas:

.. code-block:: python

   import hpcs
   import numpy as np

   # Time series data
   data = np.random.randn(100000)

   # Rolling mean (50x faster than pandas.rolling)
   rolling_avg = hpcs.rolling_mean(data, window=50)

   # Rolling median (100x faster than pandas.rolling)
   rolling_med = hpcs.rolling_median(data, window=50)

   # Rolling standard deviation
   rolling_std = hpcs.rolling_std(data, window=50)

   # Rolling z-score
   rolling_z = hpcs.rolling_zscore(data, window=100)

Anomaly Detection
-----------------

Statistical Anomaly Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create data with anomalies
   normal_data = np.random.randn(1000)
   anomalies = np.array([10, -10, 15])  # Outliers
   data = np.concatenate([normal_data, anomalies])

   # Detect anomalies (z-score > 3.0)
   is_anomaly = hpcs.detect_anomalies(data, threshold=3.0)

   # Find anomaly indices
   anomaly_indices = np.where(is_anomaly)[0]
   print(f"Found {len(anomaly_indices)} anomalies")

Robust Anomaly Detection
~~~~~~~~~~~~~~~~~~~~~~~~~

For data with outliers, use MAD-based robust detection:

.. code-block:: python

   # Robust anomaly detection (uses MAD instead of std)
   is_anomaly_robust = hpcs.detect_anomalies_robust(data, threshold=3.0)

   # Compute robust z-scores
   robust_z = hpcs.robust_zscore(data)

Robust Statistics
-----------------

Median and MAD
~~~~~~~~~~~~~~

.. code-block:: python

   # Median (more robust than mean)
   med = hpcs.median(data)

   # Median Absolute Deviation
   mad = hpcs.mad(data)

   # Quantiles
   q25 = hpcs.quantile(data, 0.25)
   q75 = hpcs.quantile(data, 0.75)

2D Array Operations
-------------------

Axis Reductions
~~~~~~~~~~~~~~~

Compute statistics along array axes (like NumPy's axis parameter):

.. code-block:: python

   # 2D array: 1000 samples × 10 features
   data_2d = np.random.randn(1000, 10)

   # Per-column mean (axis=0)
   col_means = hpcs.axis_mean(data_2d)  # shape: (10,)

   # Per-column median
   col_medians = hpcs.axis_median(data_2d)

   # Per-column MAD
   col_mad = hpcs.axis_mad(data_2d)

Masked Operations
-----------------

Handling Missing Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Data with some invalid values
   data = np.random.randn(1000)
   data[::10] = np.nan  # Every 10th value is NaN

   # Create mask (1 = valid, 0 = skip)
   mask = ~np.isnan(data)
   mask = mask.astype(np.int32)

   # Compute statistics ignoring NaN values
   mean_masked = hpcs.mean_masked(data, mask)
   median_masked = hpcs.median_masked(data, mask)
   var_masked = hpcs.var_masked(data, mask)

SIMD Information
----------------

Query CPU Capabilities
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get SIMD information
   simd = hpcs.simd_info()
   print(simd)
   # {'isa': 'AVX2', 'width_bytes': 32, 'width_doubles': 4}

   # Get SIMD width
   width = hpcs.get_simd_width()  # 4 doubles with AVX2

   # Get detailed CPU info
   cpu_info = hpcs.get_cpu_info()
   print(f"Physical cores: {cpu_info['physical_cores']}")
   print(f"Logical cores:  {cpu_info['logical_cores']}")

Performance Optimization
------------------------

Calibration
~~~~~~~~~~~

Run calibration to optimize performance for your hardware:

.. code-block:: python

   # Full calibration (~30 seconds)
   hpcs.calibrate()

   # Save configuration for future sessions
   hpcs.save_calibration_config()
   # Saves to ~/.hpcs/config.json

   # Load existing configuration
   hpcs.load_calibration_config()

Best Practices
~~~~~~~~~~~~~~

1. **Use contiguous arrays**: Ensure NumPy arrays are C-contiguous

   .. code-block:: python

      # Check if contiguous
      if not x.flags['C_CONTIGUOUS']:
          x = np.ascontiguousarray(x)

2. **Set OpenMP threads**: Control parallelization

   .. code-block:: python

      import os
      os.environ['OMP_NUM_THREADS'] = '8'  # Use 8 threads

3. **Calibrate once**: Run calibration and save config

   .. code-block:: python

      hpcs.calibrate()
      hpcs.save_calibration_config()

4. **Use appropriate data types**: HPCSeries works with ``float64`` (double precision)

   .. code-block:: python

      # Convert to float64 if needed
      x = x.astype(np.float64)

Next Steps
----------

- **Examples**: See :doc:`notebooks/index` for comprehensive tutorials
- **API Reference**: Full function documentation at :doc:`api/index`
- **User Guide**: Detailed explanations in :doc:`user_guide/index`

Common Patterns
---------------

Time Series Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hpcs
   import numpy as np

   # Load time series
   prices = np.random.randn(10000).cumsum()

   # Smooth with rolling mean
   smoothed = hpcs.rolling_mean(prices, window=20)

   # Detect volatility spikes
   volatility = hpcs.rolling_std(prices, window=20)
   high_vol = volatility > np.quantile(volatility, 0.95)

   # Detect price anomalies
   anomalies = hpcs.detect_anomalies_robust(prices, threshold=3.0)

Sensor Data Processing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # IoT sensor data with gaps
   sensor_data = np.random.randn(100000)
   sensor_data[1000:1500] = np.nan  # Sensor offline period

   # Create mask for valid data
   valid = ~np.isnan(sensor_data)
   mask = valid.astype(np.int32)

   # Compute rolling statistics ignoring gaps
   rolling_avg_masked = hpcs.rolling_mean_masked(sensor_data, mask, window=100)

Financial Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Stock returns
   returns = np.random.randn(5000) * 0.01

   # Compute rolling Sharpe ratio
   rolling_mean = hpcs.rolling_mean(returns, window=252)
   rolling_std = hpcs.rolling_std(returns, window=252)
   sharpe = rolling_mean / rolling_std * np.sqrt(252)

   # Detect extreme events
   extreme = hpcs.detect_anomalies(returns, threshold=4.0)
