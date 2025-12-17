# HPCSeries Core — Example Notebooks

This directory contains **12 Jupyter notebooks** demonstrating the capabilities of HPCSeries Core v0.7.

## Overview

These notebooks showcase:
- **Time-series analytics** — rolling operations, reductions, transformations
- **Robust statistics** — median, MAD, outlier-resistant methods
- **Anomaly detection** — z-score and robust z-score based detection
- **Multi-series processing** — batched operations for many time-series at once
- **Missing data handling** — masked operations for real-world messy data
- **Performance optimization** — calibration and benchmarking
- **Migration guides** — NumPy/Pandas to HPCSeries
- **Real-world applications** — Industry-specific use cases

---

## Notebooks

### 0. Getting Started ✅
**File**: [`00_getting_started.ipynb`](00_getting_started.ipynb)

**What you'll learn**:
- Quick introduction to HPCSeries Core
- Basic statistical operations
- Rolling window operations
- Auto-tuning calibration
- CLI commands
- Performance comparison with NumPy

**Key HPCSeries features**:
- Core API overview
- `hpcs.calibrate()` for auto-tuning
- Command-line interface
- Basic benchmarking

**Runtime**: ~5 minutes

---

### 1. Rolling Mean vs Rolling Median ✅
**File**: [`01_rolling_mean_vs_median.ipynb`](01_rolling_mean_vs_median.ipynb)

**Dataset**: Daily temperature readings from a weather station (1 year)

**What you'll learn**:
- How rolling mean and rolling median differ
- Why rolling median is more robust to sensor spikes/outliers
- Performance comparison between the two methods (50-100x faster than Pandas)

**Key HPCSeries functions**:
- `hpcs.rolling_mean()`
- `hpcs.rolling_median()`
- `hpcs.rolling_std()`

---

### 2. Robust Anomaly Detection on Climate Data ✅
**File**: [`02_robust_anomaly_climate.ipynb`](02_robust_anomaly_climate.ipynb)

**Dataset**: PM2.5 air pollution time-series with occasional sensor errors

**What you'll learn**:
- How to detect anomalies using z-score methods
- Why robust z-score (median + MAD) is better for noisy sensors
- How to visualize and interpret anomaly detection results

**Key HPCSeries functions**:
- `hpcs.detect_anomalies()`
- `hpcs.detect_anomalies_robust()`
- `hpcs.rolling_robust_zscore()`
- `hpcs.mad()`

---

### 3. Batched Rolling Analytics for IoT Sensors ✅
**File**: [`03_batched_iot_rolling.ipynb`](03_batched_iot_rolling.ipynb)

**Dataset**: Multi-sensor IoT data (100 sensors, 10k time steps)

**What you'll learn**:
- How to process many time-series at once
- How `rolling_mean_batched()` handles 2D arrays
- Performance benefits of batched processing
- Handling sensor gaps and missing data

**Key HPCSeries functions**:
- `hpcs.rolling_mean_batched()`
- `hpcs.rolling_mean_masked()`
- `hpcs.axis_mean()`
- `hpcs.anomaly_axis()`

---

### 4. Axis Reductions — Column-Wise Statistics ✅
**File**: [`04_axis_reductions_column_stats.ipynb`](04_axis_reductions_column_stats.ipynb)

**Dataset**: Financial market data (multiple stocks over time)

**What you'll learn**:
- How to compute column-wise statistics efficiently
- How axis operations work with Fortran-order arrays
- How to aggregate multi-series data for reporting

**Key HPCSeries functions**:
- `hpcs.axis_sum()`
- `hpcs.axis_mean()`
- `hpcs.axis_median()`
- `hpcs.axis_mad()`
- `hpcs.axis_min()` / `hpcs.axis_max()`

---

### 5. Masked Analytics — Missing Data Workflows ✅
**File**: [`05_masked_missing_data.ipynb`](05_masked_missing_data.ipynb)

**Dataset**: Environmental sensor data with missing values

**What you'll learn**:
- How to handle missing data with masked operations
- How to compute statistics only on valid data points
- How to combine masks with rolling operations
- Best practices for real-world messy data

**Key HPCSeries functions**:
- `hpcs.sum_masked()`
- `hpcs.mean_masked()`
- `hpcs.var_masked()`
- `hpcs.median_masked()`
- `hpcs.mad_masked()`
- `hpcs.rolling_mean_masked()`

---

### 6. Performance Optimization and Calibration ✅
**File**: [`06_performance_calibration.ipynb`](06_performance_calibration.ipynb)

**What you'll learn**:
- Auto-tuning calibration system
- Understanding configuration parameters
- Performance benchmarking methodology
- Scaling analysis across different array sizes
- Operation-specific tuning

**Key HPCSeries features**:
- `hpcs.calibrate()` - Full calibration (~30 seconds)
- `hpcs.save_calibration_config()` - Save configuration
- `hpcs.load_calibration_config()` - Load configuration
- `hpcs.simd_info()` - Check SIMD capabilities
- Performance visualization and analysis

---

### 7. C-Optimized Operations ✅
**File**: [`07_c_optimized_operations.ipynb`](07_c_optimized_operations.ipynb)

**What you'll learn**:
- C-accelerated rolling z-score computation
- Robust z-score using MAD
- Performance comparison: C vs Python
- Memory efficiency improvements
- Real-world anomaly detection use case
- Implementation details and optimizations

**Key HPCSeries features**:
- `hpcs.rolling_zscore()` - Single-pass C-optimized
- `hpcs.rolling_robust_zscore()` - MAD-based robust normalization
- SIMD vectorization benefits
- Zero-copy NumPy integration
- 2-3x speedup demonstrations

---

### 8. NumPy/Pandas Migration Guide ✅
**File**: [`08_numpy_pandas_migration_guide.ipynb`](08_numpy_pandas_migration_guide.ipynb)

**What you'll learn**:
- Complete NumPy/Pandas → HPCSeries API mapping
- Performance comparison across all operations
- Memory efficiency analysis (2-3x reduction)
- When to use HPCSeries (scaling analysis)
- Real-world IoT sensor analytics migration
- Pandas DataFrame integration patterns
- Step-by-step migration checklist

**Key topics**:
- Side-by-side code examples
- Performance scaling (1K to 10M elements)
- Memory usage comparison
- Robust statistics advantages
- Cost-benefit analysis
- Complete API reference table

**Why migrate?**
- 5-10x faster on large arrays
- 2-3x less memory usage
- Built-in robust statistics
- Drop-in replacement for NumPy/Pandas

---

### 9. Real-World Applications ✅
**File**: [`09_real_world_applications.ipynb`](09_real_world_applications.ipynb)

**What you'll learn**:
- End-to-end industry workflows
- Environmental monitoring with regulatory compliance
- IoT predictive maintenance (HVAC systems)
- Financial portfolio risk management
- Automated alert and report generation
- Production-ready decision-making logic

**Industries covered**:
- Environmental monitoring (air quality compliance)
- Smart buildings (predictive HVAC maintenance)
- Financial services (portfolio risk/VaR analysis)
- Manufacturing (quality control)
- Healthcare (patient monitoring)

**Key features**:
- Complete workflows: data → analysis → action
- Automated compliance reporting
- Predictive maintenance algorithms
- Risk scoring and health metrics
- Cost-benefit analyses
- Real-time performance at scale (200+ M values/sec)

**Practical value**:
- Ready-to-deploy code templates
- Industry-specific use cases
- ROI calculations (cost savings)
- Regulatory compliance examples

---

## Case Studies

### Kaggle Store Sales Forecasting Competition ✅

**Files**:
- [`HPCSeries_Kaggle_StoreSales_v1.ipynb`](HPCSeries_Kaggle_StoreSales_v1.ipynb) - Baseline approach
- [`HPCSeries_Kaggle_StoreSales_v2.ipynb`](HPCSeries_Kaggle_StoreSales_v2.ipynb) - Optimized with HPCSeries

**Competition**: Kaggle Store Sales - Time Series Forecasting

**What you'll learn**:
- Feature engineering with rolling statistics
- Lag features and time-based aggregations
- Performance optimization for Kaggle kernels
- HPCSeries vs Pandas for competition work

**Results**:
- ~10x faster feature engineering
- Memory-efficient rolling operations
- Improved model training speed

---

## Running the Notebooks

### Prerequisites

Install HPCSeries Core v0.7:

```bash
# From repository root
pip install -e .
```

Or install with all notebook dependencies:

```bash
pip install -e ".[examples]"
```

This installs:
- `jupyter` - Jupyter notebook server
- `matplotlib` - Plotting library
- `pandas` - DataFrame operations
- `seaborn` - Statistical visualizations
- `scikit-learn` - Machine learning utilities

### Launch Jupyter

```bash
cd notebooks
jupyter notebook
```

Open any `.ipynb` file in your browser.

### Run Individual Notebooks

```bash
cd notebooks
jupyter notebook 00_getting_started.ipynb
```

---

## Data Directory

All sample datasets are stored in `notebooks/data/`:

| File | Size | Description | Used In |
|------|------|-------------|---------|
| `climate_daily_temp.csv` | 13 KB | Daily temperature readings (1 year, with sensor spikes) | Notebook 01 |
| `climate_pm25_timeseries.csv` | 15 KB | PM2.5 air pollution data with sensor errors | Notebook 02 |
| `iot_sensors_multiseries.csv` | 7 KB | Multi-sensor IoT dataset (100 sensors) | Notebook 03 |
| `financial_market_data.csv` | 3.5 KB | Stock prices for multiple tickers | Notebook 04 |
| `environmental_sensors_missing.csv` | 4.3 KB | Environmental data with missing values | Notebook 05 |
| `kaggle/` | Directory | Kaggle Store Sales competition data | Kaggle notebooks |

**All datasets are included in the repository** - no external downloads required!

---

## Performance Notes

All notebooks include **performance benchmarks** to demonstrate:
- SIMD acceleration for reductions and rolling operations
- OpenMP parallelization for large datasets
- Comparison with NumPy/Pandas baseline

### Expected Speedups (vs NumPy/Pandas)

| Operation | Speedup | Notes |
|-----------|---------|-------|
| `sum`, `mean`, `std` | 2-5x | SIMD vectorization |
| `rolling_mean` | 50-100x | vs Pandas rolling |
| `rolling_median` | 100-200x | vs Pandas rolling |
| `axis_sum` | 3-8x | SIMD + Fortran-order optimization |
| `median` | 1.5-2x | Quickselect algorithm |
| `mad` | 2-3x | Two-pass algorithm |

### Scaling to Large Data

Performance improves with array size due to:
- SIMD efficiency (process 4-8 doubles per cycle)
- OpenMP parallelization (auto-engages for large arrays)
- Cache-friendly algorithms

**Sweet spot**: 10K - 100M elements per array.

---

## Contributing

Have a use case you'd like to see demonstrated? We welcome contributions!

**To contribute a notebook**:

1. Create a realistic dataset (CSV format, < 10 MB)
2. Write a clear use case description
3. Demonstrate HPCSeries functions solving a real problem
4. Include performance comparisons
5. Add visualizations
6. Submit a PR

**Good notebook topics**:
- Domain-specific applications (finance, biology, physics, etc.)
- Integration with other libraries (Dask, PyTorch, TensorFlow)
- Advanced performance optimization techniques
- Production deployment patterns
- Edge cases and troubleshooting

See [Contributing Guide](../docs/source/contributing.rst) for more details.

---

## Tips for Best Results

### 1. Run Calibration First

```python
import hpcs
hpcs.calibrate()
hpcs.save_calibration_config()
```

This optimizes HPCSeries for your specific hardware (~30 seconds, one-time).

### 2. Use Appropriate Array Sizes

For meaningful benchmarks:
- Small: 1K - 10K elements (test overhead)
- Medium: 10K - 1M elements (SIMD benefits visible)
- Large: 1M - 100M elements (OpenMP parallelization)

### 3. Compare Fairly

When comparing with NumPy/Pandas:
- Use same data types (`float64`)
- Use same array layout (C-contiguous)
- Warm up caches before timing
- Average over multiple runs

### 4. Visualize Results

All notebooks include visualization examples:
- Time-series plots
- Anomaly detection overlays
- Performance scaling charts
- Memory usage comparisons

---

## Support

**Questions about notebooks?**
- Open an issue: [GitHub Issues](https://github.com/your-org/HPCSeriesCore/issues)
- Check docs: [Documentation](../docs/source/index.rst)

**Found a bug in a notebook?**
- Please report it with the notebook name and cell number

**Want to share your notebook?**
- Submit a PR with your `.ipynb` file and any data files

---

## License

All notebooks and sample data are provided under the MIT License.

See `../LICENSE` for details.

---

