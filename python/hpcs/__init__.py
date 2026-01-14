"""
HPCSeries Core v0.9 - Python Bindings
======================================

High-performance statistical computing library with SIMD vectorization,
OpenMP parallelization, adaptive auto-tuning, and composable pipelines.

Examples
--------
>>> import hpcs
>>> import numpy as np
>>> x = np.random.randn(1000000)

# Basic reductions (SIMD-accelerated)
>>> hpcs.sum(x)
>>> hpcs.mean(x)
>>> hpcs.std(x)

# Rolling operations (fast C++ implementation)
>>> hpcs.rolling_mean(x, window=50)
>>> hpcs.rolling_median(x, window=100)

# Robust statistics (MAD-based outlier detection)
>>> hpcs.median(x)
>>> hpcs.mad(x)
>>> hpcs.robust_zscore(x)

# Anomaly detection
>>> anomalies = hpcs.detect_anomalies(x, threshold=3.0)

# v0.8.0: Composable pipelines
>>> pipe = hpcs.pipeline(mode='fast')
>>> pipe.diff(order=1).ewma(alpha=0.2).robust_zscore()
>>> result = pipe.execute(x)
>>> print(pipe.summary())
"""

__version__ = "0.8.0"
__author__ = "HPCSeries Core Team"

# Import core reduction functions from Cython extension
from hpcs._core import (
    # Basic reductions
    sum,
    mean,
    var,
    std,
    min,
    max,


    group_sum,
    group_mean,
    group_var,

    # Robust statistics
    median,
    mad,
    quantile,

    # Transforms & normalization
    zscore,
    robust_zscore,
    normalize_minmax,
    clip,

    # Extended transforms & descriptive statistics
    ewma,
    ewvar,
    ewstd,
    diff,
    cumulative_min,
    cumulative_max,
    convolve_valid,
    trimmed_mean,
    winsorized_mean,

    # Execution mode API
    set_execution_mode,
    get_execution_mode,
    MODE_SAFE,
    MODE_FAST,
    MODE_DETERMINISTIC,

    # Anomaly detection
    detect_anomalies,
    detect_anomalies_robust,

    # Rolling operations
    rolling_sum,
    rolling_mean,
    rolling_std,
    rolling_var,
    rolling_median,
    rolling_mad,
    rolling_zscore,
    rolling_robust_zscore,

    # 2D Axis operations
    axis_sum,
    axis_mean,
    axis_median,
    axis_mad,
    axis_min,
    axis_max,

    # Anomaly detection - axis operations
    anomaly_axis,
    anomaly_robust_axis,

    # Batched/Masked rolling operations
    rolling_mean_batched,
    rolling_mean_masked,

    # Masked operations
    sum_masked,
    mean_masked,
    var_masked,
    median_masked,
    mad_masked,
)

# Import SIMD-specific functions
from hpcs._simd import (
    simd_info,
    get_simd_width,
    get_cpu_info,
)

# Import calibration functions
from hpcs._core import (
    calibrate,
    save_calibration_config,
    load_calibration_config,
)

# Import pipeline, workspace, and feature discovery (v0.8.0)
from hpcs._core import (
    # Pipeline API
    pipeline,
    workspace,

    # Feature discovery
    build_features,
    last_error,

    # Feature bitmask constants
    FEAT_OPENMP,
    FEAT_SIMD_AVX2,
    FEAT_SIMD_AVX512,
    FEAT_SIMD_NEON,
    FEAT_FAST_MATH,
    FEAT_GPU_OFFLOAD,
    FEAT_CALIBRATED,
)

# Public API
__all__ = [
    # Version
    "__version__",

    # Reductions
    "sum",
    "mean",
    "var",
    "std",
    "min",
    "max",

    # Grouped reductions
    "group_sum",
    "group_mean",
    "group_var",

    # Robust stats
    "median",
    "mad",
    "quantile",

    # Transforms & normalization
    "zscore",
    "robust_zscore",
    "normalize_minmax",
    "clip",

    # Extended transforms & descriptive statistics
    "ewma",
    "ewvar",
    "ewstd",
    "diff",
    "cumulative_min",
    "cumulative_max",
    "convolve_valid",
    "trimmed_mean",
    "winsorized_mean",

    # Execution mode API
    "set_execution_mode",
    "get_execution_mode",
    "MODE_SAFE",
    "MODE_FAST",
    "MODE_DETERMINISTIC",

    # Anomaly detection
    "detect_anomalies",
    "detect_anomalies_robust",

    # Rolling operations
    "rolling_sum",
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "rolling_median",
    "rolling_mad",
    "rolling_zscore",
    "rolling_robust_zscore",

    # 2D Axis operations
    "axis_sum",
    "axis_mean",
    "axis_median",
    "axis_mad",
    "axis_min",
    "axis_max",

    # Anomaly detection - axis operations
    "anomaly_axis",
    "anomaly_robust_axis",

    # Batched/Masked rolling operations
    "rolling_mean_batched",
    "rolling_mean_masked",

    # Masked operations
    "sum_masked",
    "mean_masked",
    "var_masked",
    "median_masked",
    "mad_masked",

    # SIMD info
    "simd_info",
    "get_simd_width",
    "get_cpu_info",

    # Calibration
    "calibrate",
    "save_calibration_config",
    "load_calibration_config",

    # Pipeline API (v0.8.0)
    "pipeline",
    "workspace",

    # Feature discovery (v0.8.0)
    "build_features",
    "last_error",
    "FEAT_OPENMP",
    "FEAT_SIMD_AVX2",
    "FEAT_SIMD_AVX512",
    "FEAT_SIMD_NEON",
    "FEAT_FAST_MATH",
    "FEAT_GPU_OFFLOAD",
    "FEAT_CALIBRATED",
]
