API Reference
=============

This page contains the complete API reference for HPCSeries Core, automatically generated from docstrings.

Core Module
-----------

.. automodule:: hpcs
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Basic Reductions
~~~~~~~~~~~~~~~~

SIMD-accelerated reduction operations.

.. autosummary::
   :toctree: generated/

   hpcs.sum
   hpcs.mean
   hpcs.var
   hpcs.std
   hpcs.min
   hpcs.max

Robust Statistics
~~~~~~~~~~~~~~~~~

Robust statistical operations using Median Absolute Deviation (MAD).

.. autosummary::
   :toctree: generated/

   hpcs.median
   hpcs.mad
   hpcs.quantile

Execution Mode API (v0.8)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Control execution mode for safety/performance trade-offs.

.. autosummary::
   :toctree: generated/

   hpcs.set_execution_mode
   hpcs.get_execution_mode
   hpcs.MODE_SAFE
   hpcs.MODE_FAST
   hpcs.MODE_DETERMINISTIC

Exponential Weighted Statistics (v0.8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Streaming statistics with exponential decay weighting (15-60x faster than pandas).

.. autosummary::
   :toctree: generated/

   hpcs.ewma
   hpcs.ewvar
   hpcs.ewstd

Time Series Transforms (v0.8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Differencing, cumulative operations, and convolution.

.. autosummary::
   :toctree: generated/

   hpcs.diff
   hpcs.cumulative_min
   hpcs.cumulative_max
   hpcs.convolve_valid

Advanced Robust Statistics (v0.8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outlier-resistant descriptive statistics (10-15x faster than SciPy).

.. autosummary::
   :toctree: generated/

   hpcs.trimmed_mean
   hpcs.winsorized_mean

Transforms & Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Data transformation and normalization functions.

.. autosummary::
   :toctree: generated/

   hpcs.zscore
   hpcs.robust_zscore
   hpcs.normalize_minmax
   hpcs.clip

Anomaly Detection
~~~~~~~~~~~~~~~~~

Statistical and robust anomaly detection methods.

.. autosummary::
   :toctree: generated/

   hpcs.detect_anomalies
   hpcs.detect_anomalies_robust

Rolling Operations
~~~~~~~~~~~~~~~~~~

Fast sliding window computations (50-100x faster than Pandas).

.. autosummary::
   :toctree: generated/

   hpcs.rolling_sum
   hpcs.rolling_mean
   hpcs.rolling_std
   hpcs.rolling_var
   hpcs.rolling_median
   hpcs.rolling_mad
   hpcs.rolling_zscore
   hpcs.rolling_robust_zscore

Axis Operations
~~~~~~~~~~~~~~~

2D array operations along specified axes.

.. autosummary::
   :toctree: generated/

   hpcs.axis_sum
   hpcs.axis_mean
   hpcs.axis_median
   hpcs.axis_mad
   hpcs.axis_min
   hpcs.axis_max
   hpcs.anomaly_axis
   hpcs.anomaly_robust_axis

Masked Operations
~~~~~~~~~~~~~~~~~

Operations on arrays with missing data (masked values).

.. autosummary::
   :toctree: generated/

   hpcs.sum_masked
   hpcs.mean_masked
   hpcs.var_masked
   hpcs.median_masked
   hpcs.mad_masked
   hpcs.rolling_mean_masked
   hpcs.rolling_mean_batched

SIMD & CPU Information
~~~~~~~~~~~~~~~~~~~~~~

Query SIMD capabilities and CPU topology.

.. autosummary::
   :toctree: generated/

   hpcs.simd_info
   hpcs.get_simd_width
   hpcs.get_cpu_info

Calibration
~~~~~~~~~~~

Performance calibration and configuration management.

.. autosummary::
   :toctree: generated/

   hpcs.calibrate
   hpcs.save_calibration_config
   hpcs.load_calibration_config

Detailed API
------------

Sum
^^^

.. autofunction:: hpcs.sum

Mean
^^^^

.. autofunction:: hpcs.mean

Standard Deviation
^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.std

Variance
^^^^^^^^

.. autofunction:: hpcs.var

Minimum
^^^^^^^

.. autofunction:: hpcs.min

Maximum
^^^^^^^

.. autofunction:: hpcs.max

Median
^^^^^^

.. autofunction:: hpcs.median

MAD (Median Absolute Deviation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.mad

Quantile
^^^^^^^^

.. autofunction:: hpcs.quantile

Z-Score
^^^^^^^

.. autofunction:: hpcs.zscore

Robust Z-Score
^^^^^^^^^^^^^^

.. autofunction:: hpcs.robust_zscore

Min-Max Normalization
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.normalize_minmax

Clip
^^^^

.. autofunction:: hpcs.clip

Detect Anomalies
^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.detect_anomalies

Detect Anomalies (Robust)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.detect_anomalies_robust

Rolling Sum
^^^^^^^^^^^

.. autofunction:: hpcs.rolling_sum

Rolling Mean
^^^^^^^^^^^^

.. autofunction:: hpcs.rolling_mean

Rolling Standard Deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.rolling_std

Rolling Variance
^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.rolling_var

Rolling Median
^^^^^^^^^^^^^^

.. autofunction:: hpcs.rolling_median

Rolling MAD
^^^^^^^^^^^

.. autofunction:: hpcs.rolling_mad

Rolling Z-Score
^^^^^^^^^^^^^^^

.. autofunction:: hpcs.rolling_zscore

Rolling Robust Z-Score
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.rolling_robust_zscore

Axis Sum
^^^^^^^^

.. autofunction:: hpcs.axis_sum

Axis Mean
^^^^^^^^^

.. autofunction:: hpcs.axis_mean

Axis Median
^^^^^^^^^^^

.. autofunction:: hpcs.axis_median

Axis MAD
^^^^^^^^

.. autofunction:: hpcs.axis_mad

Axis Min
^^^^^^^^

.. autofunction:: hpcs.axis_min

Axis Max
^^^^^^^^

.. autofunction:: hpcs.axis_max

Anomaly Detection (Axis)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.anomaly_axis

Robust Anomaly Detection (Axis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.anomaly_robust_axis

Sum (Masked)
^^^^^^^^^^^^

.. autofunction:: hpcs.sum_masked

Mean (Masked)
^^^^^^^^^^^^^

.. autofunction:: hpcs.mean_masked

Variance (Masked)
^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.var_masked

Median (Masked)
^^^^^^^^^^^^^^^

.. autofunction:: hpcs.median_masked

MAD (Masked)
^^^^^^^^^^^^

.. autofunction:: hpcs.mad_masked

Rolling Mean (Masked)
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.rolling_mean_masked

Rolling Mean (Batched)
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.rolling_mean_batched

SIMD Info
^^^^^^^^^

.. autofunction:: hpcs.simd_info

Get SIMD Width
^^^^^^^^^^^^^^

.. autofunction:: hpcs.get_simd_width

Get CPU Info
^^^^^^^^^^^^

.. autofunction:: hpcs.get_cpu_info

Calibrate
^^^^^^^^^

.. autofunction:: hpcs.calibrate

Save Calibration Config
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.save_calibration_config

Load Calibration Config
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.load_calibration_config

Set Execution Mode
^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.set_execution_mode

Get Execution Mode
^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.get_execution_mode

MODE_SAFE
^^^^^^^^^

Safe execution mode (IEEE 754 compliant, full error checking).

.. data:: hpcs.MODE_SAFE
   :type: int
   :value: 0

   Use SAFE mode for maximum numerical stability and error detection.

MODE_FAST
^^^^^^^^^

Fast execution mode (relaxed math optimizations, minimal validation).

.. data:: hpcs.MODE_FAST
   :type: int
   :value: 1

   Use FAST mode for performance-critical code where data is pre-validated.

MODE_DETERMINISTIC
^^^^^^^^^^^^^^^^^^

Deterministic execution mode (bit-exact reproducibility, no OpenMP).

.. data:: hpcs.MODE_DETERMINISTIC
   :type: int
   :value: 2

   Use DETERMINISTIC mode when exact reproducibility is required across runs.

Exponential Weighted Moving Average
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.ewma

Exponential Weighted Variance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.ewvar

Exponential Weighted Standard Deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.ewstd

Differencing
^^^^^^^^^^^^

.. autofunction:: hpcs.diff

Cumulative Minimum
^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.cumulative_min

Cumulative Maximum
^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.cumulative_max

Convolution (Valid Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: hpcs.convolve_valid

Trimmed Mean
^^^^^^^^^^^^

.. autofunction:: hpcs.trimmed_mean

Winsorized Mean
^^^^^^^^^^^^^^^

.. autofunction:: hpcs.winsorized_mean
