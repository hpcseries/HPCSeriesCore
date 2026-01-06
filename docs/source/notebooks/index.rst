Examples & Tutorials
====================

HPCSeries Core includes **12 comprehensive Jupyter notebooks** demonstrating real-world use cases.

📓 **All notebooks are located in**: ``notebooks/`` directory in the repository.

Getting Started
---------------

To run these notebooks locally:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/hpcseries/HPCSeriesCore.git
   cd HPCSeriesCore

   # Install HPCSeries with examples dependencies
   pip install -e ".[examples]"

   # Launch Jupyter
   jupyter notebook notebooks/

Alternatively, install just the examples dependencies:

.. code-block:: bash

   pip install jupyter matplotlib pandas seaborn scikit-learn

Notebook Catalog
----------------

All notebooks showcase **real or realistic datasets** and demonstrate practical use cases including:
- Time-series analytics (rolling operations, reductions, transformations)
- Robust statistics (median, MAD, outlier-resistant methods)
- Anomaly detection (z-score and robust methods)
- Multi-series processing (batched operations)
- Missing data handling (masked operations)

00. Getting Started
~~~~~~~~~~~~~~~~~~~

**Notebook**: `00_getting_started.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/00_getting_started.ipynb>`_

**Quick introduction to HPCSeries Core**

Learn:
  - Basic statistical operations (sum, mean, std, etc.)
  - Rolling window operations
  - Auto-tuning calibration
  - CLI commands (``hpcs cpuinfo``)
  - Performance comparison with NumPy

**Key Functions**: ``hpcs.calibrate()``, ``hpcs.rolling_mean()``, ``hpcs.zscore()``

**Runtime**: ~5 minutes

01. Rolling Mean vs Rolling Median
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `01_rolling_mean_vs_median.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/01_rolling_mean_vs_median.ipynb>`_

**Dataset**: Daily temperature readings from a weather station (1 year)

Learn:
  - How rolling mean and rolling median differ
  - Why rolling median is more robust to sensor spikes/outliers
  - Performance comparison (50-100x faster than Pandas)

**Key Functions**: ``hpcs.rolling_mean()``, ``hpcs.rolling_median()``, ``hpcs.rolling_std()``

**Performance**: 50-100x faster than ``pandas.rolling()``

02. Robust Anomaly Detection (Climate Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `02_robust_anomaly_climate.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/02_robust_anomaly_climate.ipynb>`_

**Dataset**: Historical climate data with extreme weather events

Learn:
  - Detect anomalies using z-scores
  - Robust detection using MAD (Median Absolute Deviation)
  - Why robust methods are better for real-world data

**Key Functions**: ``hpcs.detect_anomalies()``, ``hpcs.detect_anomalies_robust()``, ``hpcs.mad()``

**Use Case**: Climate monitoring, weather stations, environmental sensors

03. Batched IoT Rolling Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `03_batched_iot_rolling.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/03_batched_iot_rolling.ipynb>`_

**Dataset**: Multiple IoT sensor streams with data gaps

Learn:
  - Process multiple time-series simultaneously
  - Handle sensor offline periods
  - Batched rolling operations for efficiency

**Key Functions**: ``hpcs.rolling_mean_batched()``, ``hpcs.rolling_mean_masked()``

**Use Case**: IoT sensor networks, industrial monitoring, smart buildings

04. Axis Reductions (Column Statistics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `04_axis_reductions_column_stats.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/04_axis_reductions_column_stats.ipynb>`_

**Dataset**: Multi-feature sensor array (1000 samples × 10 sensors)

Learn:
  - Compute per-column statistics efficiently
  - Process 2D arrays (samples × features)
  - Alternative to ``pandas.DataFrame.agg()``

**Key Functions**: ``hpcs.axis_mean()``, ``hpcs.axis_median()``, ``hpcs.axis_mad()``

**Use Case**: Feature engineering, multi-sensor analysis, data preprocessing

05. Masked Operations (Missing Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `05_masked_missing_data.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/05_masked_missing_data.ipynb>`_

**Dataset**: Sensor data with missing values (NaN gaps)

Learn:
  - Handle missing data without imputation
  - Compute statistics ignoring invalid values
  - Masked rolling operations

**Key Functions**: ``hpcs.mean_masked()``, ``hpcs.median_masked()``, ``hpcs.rolling_mean_masked()``

**Use Case**: Real-world sensor data, incomplete datasets, telemetry with dropouts

06. Performance Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `06_performance_calibration.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/06_performance_calibration.ipynb>`_

**Auto-tuning HPCSeries for your hardware**

Learn:
  - How calibration works
  - Benchmarking before/after calibration
  - Understanding SIMD and OpenMP thresholds
  - Configuration persistence

**Key Functions**: ``hpcs.calibrate()``, ``hpcs.save_calibration_config()``, ``hpcs.simd_info()``

**Use Case**: First-time setup, performance optimization

07. C-Optimized Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `07_c_optimized_operations.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/07_c_optimized_operations.ipynb>`_

**Deep dive into SIMD acceleration**

Learn:
  - How SIMD vectorization works
  - AVX2 vs AVX-512 vs SSE2 performance
  - When SIMD provides benefits
  - Cache-friendly algorithms

**Key Functions**: ``hpcs.get_cpu_info()``, ``hpcs.simd_info()``

**Use Case**: Understanding performance characteristics, optimization

08. NumPy/Pandas Migration Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `08_numpy_pandas_migration_guide.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/08_numpy_pandas_migration_guide.ipynb>`_

**Step-by-step migration examples**

Learn:
  - Drop-in replacements for NumPy functions
  - Converting Pandas rolling operations
  - Performance comparisons
  - When to use NumPy/Pandas vs HPCSeries

**Use Case**: Migrating existing code, evaluating HPCSeries

09. Real-World Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: `09_real_world_applications.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/09_real_world_applications.ipynb>`_

**Complete examples across multiple domains**

Learn:
  - Financial data analysis (stock prices, Sharpe ratios)
  - Sensor stream processing (real-time monitoring)
  - Scientific computing (large-scale simulations)

**Use Case**: End-to-end workflows, production deployments

Case Studies
------------

Kaggle Competition: Store Sales Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebooks**:
  - `HPCSeries_Kaggle_StoreSales_v1.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/HPCSeries_Kaggle_StoreSales_v1.ipynb>`_ - Baseline approach
  - `HPCSeries_Kaggle_StoreSales_v2.ipynb <https://github.com/hpcseries/HPCSeriesCore/blob/main/notebooks/HPCSeries_Kaggle_StoreSales_v2.ipynb>`_ - Optimized with HPCSeries

**Competition**: Kaggle Store Sales - Time Series Forecasting

Learn:
  - Feature engineering with rolling statistics
  - Lag features and time-based aggregations
  - Performance optimization for Kaggle kernels
  - HPCSeries vs Pandas for competition work

**Results**: ~10x faster feature engineering

Running in the Cloud
--------------------

Google Colab
~~~~~~~~~~~~

Upload notebooks to Colab and install HPCSeries:

.. code-block:: python

   !pip install hpcs

   import hpcs
   print(hpcs.__version__)

Binder
~~~~~~

Launch an interactive environment (if configured):

.. code-block:: bash

   # Add binder configuration to your repo
   # Create environment.yml with dependencies

Contributing Notebooks
----------------------

We welcome notebook contributions! See :doc:`../contributing` for guidelines.

**Good notebook topics**:
  - Domain-specific applications (finance, biology, physics)
  - Integration with other libraries (Dask, PyTorch, etc.)
  - Performance comparisons and benchmarks
  - Advanced use cases and optimizations

Support
-------

- **Questions?** Open an issue on `GitHub <https://github.com/hpcseries/HPCSeriesCore/issues>`_
- **Want to share your notebook?** Submit a pull request!
- **Found a bug in a notebook?** Please report it

Download All Notebooks
-----------------------

All notebooks are included in the repository:

.. code-block:: bash

   git clone https://github.com/hpcseries/HPCSeriesCore.git
   cd HPCSeriesCore/notebooks

Or download as ZIP from GitHub.
