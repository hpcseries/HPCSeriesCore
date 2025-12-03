"""
HPCSeries Core v0.7 - Python Bindings
======================================

High-performance statistical computing library with SIMD vectorization,
OpenMP parallelization, and adaptive auto-tuning.

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
"""

__version__ = "0.7.0"
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

    # Robust statistics
    median,
    mad,

    # Rolling operations
    rolling_mean,
    rolling_std,
    rolling_var,
    rolling_median,
    rolling_mad,
)

# Import SIMD-specific functions
from hpcs._simd import (
    simd_info,
    get_simd_width,
    get_cpu_info,
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

    # Robust stats
    "median",
    "mad",

    # Rolling operations
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "rolling_median",
    "rolling_mad",

    # SIMD info
    "simd_info",
    "get_simd_width",
    "get_cpu_info",
]
