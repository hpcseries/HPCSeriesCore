# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
HPCSeries Core v0.7 - Core Reductions & Rolling Operations
===========================================================

Cython bindings to C/Fortran HPC kernels with zero-copy NumPy integration.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t

# Initialize NumPy C API
cnp.import_array()

# ==============================================================================
# C Function Declarations (from libhpcs_core)
# ==============================================================================

cdef extern from "hpcs_core.h":
    # Initialization
    void hpcs_simd_reductions_init()
    void hpcs_rolling_simd_init()
    void hpcs_zscore_simd_init()

    # Status codes
    int HPCS_SUCCESS
    int HPCS_ERR_INVALID_ARGS

    # Reduction functions (Fortran interface with v0.8.0 execution mode support)
    void hpcs_reduce_sum(const double *x, int n, double *out, int mode, int *status)
    void hpcs_reduce_mean(const double *x, int n, double *out, int mode, int *status)
    void hpcs_reduce_variance(const double *x, int n, double *out, int mode, int *status)
    void hpcs_reduce_std(const double *x, int n, double *out, int mode, int *status)
    void hpcs_reduce_min(const double *x, int n, double *out, int mode, int *status)
    void hpcs_reduce_max(const double *x, int n, double *out, int mode, int *status)

    # Grouped reductions (v0.8.0 with execution mode support)
    void hpcs_group_reduce_sum(const double *x, int n, const int *group_ids, int n_groups, double *y, int mode, int *status)
    void hpcs_group_reduce_mean(const double *x, int n, const int *group_ids, int n_groups, double *y, int mode, int *status)
    void hpcs_group_reduce_variance(const double *x, int n, const int *group_ids, int n_groups, double *y, int mode, int *status)

    # SIMD-accelerated reductions (v0.6)
    void hpcs_reduce_sum_simd(const double *x, int n, double *out, int *status)
    void hpcs_reduce_mean_simd(const double *x, int n, double *out, int *status)
    void hpcs_reduce_variance_simd(const double *x, int n, double *out, int *status)
    void hpcs_reduce_min_simd(const double *x, int n, double *out, int *status)
    void hpcs_reduce_max_simd(const double *x, int n, double *out, int *status)
    void hpcs_reduce_std_simd(const double *x, int n, double *out, int *status)

    # Robust statistics (v0.3)
    void hpcs_median(const double *x, int n, double *out, int *status)
    void hpcs_mad(const double *x, int n, double *out, int *status)
    void hpcs_quantile(const double *x, int n, double q, double *out, int *status)

    # Transform operations
    void hpcs_zscore(const double *x, int n, double *y, int *status)
    void hpcs_robust_zscore(const double *x, int n, double *y, int *status)
    void hpcs_normalize_minmax(const double *x, int n, double *y, int *status)
    void hpcs_clip(double *x, int n, double min_val, double max_val, int *status)

    # Anomaly detection
    void hpcs_detect_anomalies(const double *x, int n, double threshold, int *anomaly, int *status)
    void hpcs_detect_anomalies_robust(const double *x, int n, double threshold, int *anomaly, int *status)

    # Rolling operations (v0.3)
    void hpcs_rolling_sum(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_mean(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_std(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_variance(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_median(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_mad(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_zscore(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_robust_zscore(const double *x, int n, int window, double *y, int *status)

    # Batched/Axis operations (v0.4)
    void hpcs_reduce_sum_axis1(const double *x, int n, int m, double *out, int *status)
    void hpcs_reduce_mean_axis1(const double *x, int n, int m, double *out, int *status)
    void hpcs_median_axis1(const double *x, int n, int m, double *out, int *status)
    void hpcs_mad_axis1(const double *x, int n, int m, double *out, int *status)
    void hpcs_reduce_min_axis0_simd(const double *x, int n, int m, double *out, int *status)
    void hpcs_reduce_max_axis0_simd(const double *x, int n, int m, double *out, int *status)

    # Masked operations (v0.4)
    void hpcs_reduce_sum_masked(const double *x, const int *mask, int n, double *out, int *status)
    void hpcs_reduce_mean_masked(const double *x, const int *mask, int n, double *out, int *status)
    void hpcs_reduce_variance_masked(const double *x, const int *mask, int n, double *out, int *status)
    void hpcs_median_masked(const double *x, const int *mask, int n, double *out, int *status)
    void hpcs_mad_masked(const double *x, const int *mask, int n, double *out, int *status)

    # Calibration functions (v0.5)
    void hpcs_calibrate(int *status)
    void hpcs_calibrate_quick(int *status)
    void hpcs_save_config(const char *path, int *status)
    void hpcs_load_config(const char *path, int *status)

    # v0.8.0 Transform & Robust Statistics
    # Exponential weighted statistics
    void hpcs_ewma(const double *x, int n, double alpha, double *y, int mode, int *status)
    void hpcs_ewvar(const double *x, int n, double alpha, double *v_out, int mode, int *status)
    void hpcs_ewstd(const double *x, int n, double alpha, double *y, int mode, int *status)

    # Differencing & cumulative transforms
    void hpcs_diff(const double *x, int n, int order, double *y, int *status)
    void hpcs_cumulative_min(const double *x, int n, double *y, int mode, int *status)
    void hpcs_cumulative_max(const double *x, int n, double *y, int mode, int *status)

    # FIR convolution
    void hpcs_convolve_valid(const double *x, int n, const double *k, int m, double *y, int mode, int *status)

    # Robust descriptive statistics
    void hpcs_trimmed_mean(const double *x, int n, double trim_frac, double *result, int mode, int *status)
    void hpcs_winsorized_mean(const double *x, int n, double win_frac, double *result, int mode, int *status)

    # Execution mode API (v0.8.0)
    int HPCS_MODE_SAFE
    int HPCS_MODE_FAST
    int HPCS_MODE_DETERMINISTIC
    int HPCS_MODE_USE_GLOBAL
    void hpcs_set_execution_mode(int mode, int *status)
    void hpcs_get_execution_mode(int *mode, int *status)

# ==============================================================================
# Helper Functions
# ==============================================================================

cdef inline cnp.ndarray[cnp.float64_t, ndim=1] ensure_c_contiguous(object x):
    """Convert input to C-contiguous float64 NumPy array."""
    if not isinstance(x, cnp.ndarray):
        x = np.asarray(x, dtype=np.float64)
    elif x.dtype != np.float64:
        x = x.astype(np.float64)

    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)

    return x

cdef inline void check_status(int status, str func_name):
    """Check HPCS status code and raise exception if error."""
    if status != HPCS_SUCCESS:
        if status == HPCS_ERR_INVALID_ARGS:
            raise ValueError(f"{func_name}: Invalid arguments")
        else:
            raise RuntimeError(f"{func_name}: Error code {status}")

# ==============================================================================
# Module Initialization
# ==============================================================================

# Initialize SIMD dispatch system when module is imported
hpcs_simd_reductions_init()
hpcs_rolling_simd_init()
hpcs_zscore_simd_init()

# ==============================================================================
# Execution Mode API (v0.8.0)
# ==============================================================================

# Python-friendly mode names
MODE_SAFE = 'safe'
MODE_FAST = 'fast'
MODE_DETERMINISTIC = 'deterministic'

def set_execution_mode(mode):
    """
    Set the global execution mode for HPCS operations.

    Parameters
    ----------
    mode : str
        Execution mode: 'safe', 'fast', or 'deterministic'
        - 'safe': Full NaN detection and validation (default)
        - 'fast': Skip checks for maximum performance (1.2-2x faster)
        - 'deterministic': Full validation, disable SIMD/threading for reproducibility

    Raises
    ------
    ValueError
        If mode is not one of: 'safe', 'fast', 'deterministic'

    Examples
    --------
    >>> import hpcs
    >>> hpcs.set_execution_mode('fast')  # Skip NaN checks globally
    >>> result = hpcs.ewma(data, alpha=0.5)  # Uses FAST mode
    """
    cdef int c_mode
    cdef int status

    # Parse mode string to C constant
    if mode == 'safe':
        c_mode = HPCS_MODE_SAFE
    elif mode == 'fast':
        c_mode = HPCS_MODE_FAST
    elif mode == 'deterministic':
        c_mode = HPCS_MODE_DETERMINISTIC
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'safe', 'fast', or 'deterministic'")

    # Call C API
    hpcs_set_execution_mode(c_mode, &status)
    check_status(status, "set_execution_mode")

def get_execution_mode():
    """
    Get the current global execution mode.

    Returns
    -------
    str
        Current mode: 'safe', 'fast', or 'deterministic'

    Examples
    --------
    >>> import hpcs
    >>> hpcs.get_execution_mode()
    'safe'
    >>> hpcs.set_execution_mode('fast')
    >>> hpcs.get_execution_mode()
    'fast'
    """
    cdef int c_mode
    cdef int status

    hpcs_get_execution_mode(&c_mode, &status)
    check_status(status, "get_execution_mode")

    # Convert C constant to string
    if c_mode == HPCS_MODE_SAFE:
        return 'safe'
    elif c_mode == HPCS_MODE_FAST:
        return 'fast'
    elif c_mode == HPCS_MODE_DETERMINISTIC:
        return 'deterministic'
    else:
        return f'unknown({c_mode})'

cdef inline int _parse_mode(object mode) except -999:
    """
    Internal helper to parse mode parameter (None or string) to C constant.
    Returns HPCS_MODE_USE_GLOBAL if mode is None.
    """
    if mode is None:
        return HPCS_MODE_USE_GLOBAL
    elif mode == 'safe':
        return HPCS_MODE_SAFE
    elif mode == 'fast':
        return HPCS_MODE_FAST
    elif mode == 'deterministic':
        return HPCS_MODE_DETERMINISTIC
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be None, 'safe', 'fast', or 'deterministic'")

# ==============================================================================
# Python API - Reductions
# ==============================================================================

def sum(x, mode=None):
    """
    Sum of array elements (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : float
        Sum of all elements

    Examples
    --------
    >>> import hpcs
    >>> hpcs.sum([1, 2, 3, 4, 5])
    15.0
    >>> hpcs.sum([1, 2, 3, 4, 5], mode='fast')
    15.0
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_reduce_sum(&arr[0], n, &result, c_mode, &status)
    check_status(status, "sum")

    return result

def mean(x, mode=None):
    """
    Mean of array elements (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : float
        Arithmetic mean

    Examples
    --------
    >>> import hpcs
    >>> hpcs.mean([1, 2, 3, 4, 5])
    3.0
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_reduce_mean(&arr[0], n, &result, c_mode, &status)
    check_status(status, "mean")

    return result

def var(x, mode=None):
    """
    Variance of array elements (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : float
        Population variance

    Examples
    --------
    >>> import hpcs
    >>> hpcs.var([1, 2, 3, 4, 5])
    2.0
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_reduce_variance(&arr[0], n, &result, c_mode, &status)
    check_status(status, "var")

    return result

def std(x, mode=None):
    """
    Standard deviation of array elements (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : float
        Population standard deviation

    Examples
    --------
    >>> import hpcs
    >>> hpcs.std([1, 2, 3, 4, 5])
    1.4142135623730951
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_reduce_std(&arr[0], n, &result, c_mode, &status)
    check_status(status, "std")

    return result

def min(x, mode=None):
    """
    Minimum of array elements (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : float
        Minimum value
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_reduce_min(&arr[0], n, &result, c_mode, &status)
    check_status(status, "min")

    return result

def max(x, mode=None):
    """
    Maximum of array elements (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : float
        Maximum value
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_reduce_max(&arr[0], n, &result, c_mode, &status)
    check_status(status, "max")

    return result

def group_sum(x, group_ids, n_groups, mode=None):
    """
    Grouped sum (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    group_ids : array_like (int)
        Group IDs (0-indexed, values in [0, n_groups-1])
    n_groups : int
        Number of groups
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : ndarray
        Sum for each group (length n_groups)
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] gids = np.ascontiguousarray(group_ids, dtype=np.int32)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n_groups, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_group_reduce_sum(&arr[0], n, <int*>&gids[0], n_groups, &result[0], c_mode, &status)
    check_status(status, "group_sum")
    return result

def group_mean(x, group_ids, n_groups, mode=None):
    """
    Grouped mean (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    group_ids : array_like (int)
        Group IDs (0-indexed, values in [0, n_groups-1])
    n_groups : int
        Number of groups
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : ndarray
        Mean for each group (length n_groups). Empty groups return NaN.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] gids = np.ascontiguousarray(group_ids, dtype=np.int32)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n_groups, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_group_reduce_mean(&arr[0], n, <int*>&gids[0], n_groups, &result[0], c_mode, &status)
    check_status(status, "group_mean")
    return result

def group_var(x, group_ids, n_groups, mode=None):
    """
    Grouped variance (v0.8.0 with execution mode support).

    Parameters
    ----------
    x : array_like
        Input array
    group_ids : array_like (int)
        Group IDs (0-indexed, values in [0, n_groups-1])
    n_groups : int
        Number of groups
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : ndarray
        Population variance for each group (length n_groups). Empty groups return NaN.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] gids = np.ascontiguousarray(group_ids, dtype=np.int32)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n_groups, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_group_reduce_variance(&arr[0], n, <int*>&gids[0], n_groups, &result[0], c_mode, &status)
    check_status(status, "group_var")
    return result

# ==============================================================================
# Python API - Robust Statistics
# ==============================================================================

def median(x):
    """
    Median of array elements.

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : float
        Median value

    Examples
    --------
    >>> import hpcs
    >>> hpcs.median([1, 2, 3, 4, 5])
    3.0
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_median(&arr[0], n, &result, &status)
    check_status(status, "median")

    return result

def mad(x):
    """
    Median Absolute Deviation (MAD) - robust scale estimator.

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : float
        MAD value

    Examples
    --------
    >>> import hpcs
    >>> hpcs.mad([1, 2, 3, 4, 5])
    1.4826
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_mad(&arr[0], n, &result, &status)
    check_status(status, "mad")

    return result

def quantile(x, double q):
    """
    Compute q-th quantile (0 <= q <= 1).

    Uses Type 7 interpolation (linear between order statistics).

    Parameters
    ----------
    x : array_like
        Input array
    q : float
        Quantile to compute (0.0 to 1.0)

    Returns
    -------
    result : float
        Quantile value

    Examples
    --------
    >>> import hpcs
    >>> hpcs.quantile([1, 2, 3, 4, 5], 0.5)  # median
    3.0
    >>> hpcs.quantile([1, 2, 3, 4, 5], 0.25)  # 25th percentile
    2.0
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_quantile(&arr[0], n, q, &result, &status)
    check_status(status, "quantile")

    return result

# ==============================================================================
# Python API - Transforms & Normalization
# ==============================================================================

def zscore(x):
    """
    Z-score normalization: (x - mean) / std.

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : ndarray
        Z-score normalized array

    Examples
    --------
    >>> import hpcs
    >>> hpcs.zscore([1, 2, 3, 4, 5])
    array([-1.414, -0.707,  0.   ,  0.707,  1.414])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_zscore(&arr[0], n, &result[0], &status)
    check_status(status, "zscore")

    return result

def robust_zscore(x):
    """
    Robust z-score using median and MAD: (x - median) / (MAD * 1.4826).

    More resistant to outliers than standard z-score.

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : ndarray
        Robust z-score normalized array

    Examples
    --------
    >>> import hpcs
    >>> hpcs.robust_zscore([1, 2, 3, 4, 100])  # Outlier doesn't affect much
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_robust_zscore(&arr[0], n, &result[0], &status)
    check_status(status, "robust_zscore")

    return result

def normalize_minmax(x):
    """
    Min-max normalization to [0, 1] range.

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : ndarray
        Normalized array in [0, 1]

    Examples
    --------
    >>> import hpcs
    >>> hpcs.normalize_minmax([1, 2, 3, 4, 5])
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_normalize_minmax(&arr[0], n, &result[0], &status)
    check_status(status, "normalize_minmax")

    return result

def clip(x, double lo, double hi):
    """
    Clip (clamp) values to [lo, hi] range (in-place).

    Parameters
    ----------
    x : array_like
        Input array
    lo : float
        Lower bound
    hi : float
        Upper bound

    Returns
    -------
    result : ndarray
        Clipped array

    Examples
    --------
    >>> import hpcs
    >>> hpcs.clip([1, 2, 3, 4, 5], 2, 4)
    array([2., 2., 3., 4., 4.])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    # Make a copy since clip modifies in-place
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = arr.copy()
    cdef int status

    hpcs_clip(&result[0], n, lo, hi, &status)
    check_status(status, "clip")

    return result

# ==============================================================================
# Python API - Anomaly Detection
# ==============================================================================

def detect_anomalies(x, double threshold=3.0):
    """
    Detect anomalies using z-score threshold.

    Parameters
    ----------
    x : array_like
        Input array
    threshold : float, optional
        Z-score threshold (default: 3.0)

    Returns
    -------
    result : ndarray of bool
        True for anomalies, False for normal values

    Examples
    --------
    >>> import hpcs
    >>> hpcs.detect_anomalies([1, 2, 3, 4, 100], threshold=3.0)
    array([False, False, False, False, True])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[int, ndim=1] anomaly = np.empty(n, dtype=np.int32)
    cdef int status

    hpcs_detect_anomalies(&arr[0], n, threshold, <int*>&anomaly[0], &status)
    check_status(status, "detect_anomalies")

    return anomaly.astype(bool)

def detect_anomalies_robust(x, double threshold=3.0):
    """
    Detect anomalies using robust MAD-based z-score.

    More resistant to outliers than standard z-score method.

    Parameters
    ----------
    x : array_like
        Input array
    threshold : float, optional
        Robust z-score threshold (default: 3.0)

    Returns
    -------
    result : ndarray of bool
        True for anomalies, False for normal values

    Examples
    --------
    >>> import hpcs
    >>> hpcs.detect_anomalies_robust([1, 2, 3, 4, 100], threshold=3.0)
    array([False, False, False, False, True])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[int, ndim=1] anomaly = np.empty(n, dtype=np.int32)
    cdef int status

    hpcs_detect_anomalies_robust(&arr[0], n, threshold, <int*>&anomaly[0], &status)
    check_status(status, "detect_anomalies_robust")

    return anomaly.astype(bool)

# ==============================================================================
# Python API - Rolling Operations
# ==============================================================================

def rolling_sum(x, int window):
    """
    Rolling sum with specified window size.

    Parameters
    ----------
    x : array_like
        Input array
    window : int
        Window size

    Returns
    -------
    result : ndarray
        Rolling sums (same length as input)

    Examples
    --------
    >>> import hpcs
    >>> hpcs.rolling_sum([1, 2, 3, 4, 5], window=3)
    array([nan, nan, 6., 9., 12.])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_rolling_sum(&arr[0], n, window, &result[0], &status)
    check_status(status, "rolling_sum")

    return result

def rolling_mean(x, int window):
    """
    Rolling mean with specified window size.

    Parameters
    ----------
    x : array_like
        Input array
    window : int
        Window size

    Returns
    -------
    result : ndarray
        Rolling means (same length as input)

    Examples
    --------
    >>> import hpcs
    >>> hpcs.rolling_mean([1, 2, 3, 4, 5], window=3)
    array([nan, nan, 2., 3., 4.])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_rolling_mean(&arr[0], n, window, &result[0], &status)
    check_status(status, "rolling_mean")

    return result

def rolling_std(x, int window):
    """Rolling standard deviation with specified window size."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_rolling_std(&arr[0], n, window, &result[0], &status)
    check_status(status, "rolling_std")

    return result

def rolling_var(x, int window):
    """Rolling variance with specified window size."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_rolling_variance(&arr[0], n, window, &result[0], &status)
    check_status(status, "rolling_var")

    return result

def rolling_median(x, int window):
    """Rolling median with specified window size (fast C++ heap implementation)."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_rolling_median(&arr[0], n, window, &result[0], &status)
    check_status(status, "rolling_median")

    return result

def rolling_mad(x, int window):
    """Rolling MAD with specified window size (fast C++ heap implementation)."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_rolling_mad(&arr[0], n, window, &result[0], &status)
    check_status(status, "rolling_mad")

    return result

# ==============================================================================
# Python API - 2D Axis Operations (Tier B)
# ==============================================================================

def axis_sum(x, int axis=1):
    """
    Sum along specified axis of 2D array.

    Parameters
    ----------
    x : array_like (2D)
        Input 2D array
    axis : int, optional
        Axis along which to sum (default: 1, reduces columns)

    Returns
    -------
    result : ndarray (1D)
        Sums along the specified axis

    Examples
    --------
    >>> import hpcs
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> hpcs.axis_sum(x, axis=1)  # Sum across columns for each row
    array([6., 15.])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr = np.asarray(x, dtype=np.float64, order='F')
    cdef int n = arr.shape[0]
    cdef int m = arr.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result
    cdef int status

    if axis == 1:
        result = np.empty(n, dtype=np.float64)
        hpcs_reduce_sum_axis1(&arr[0,0], n, m, &result[0], &status)
    else:
        raise ValueError("Only axis=1 currently supported")

    check_status(status, "axis_sum")
    return result

def axis_mean(x, int axis=1):
    """Mean along specified axis of 2D array."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr = np.asarray(x, dtype=np.float64, order='F')
    cdef int n = arr.shape[0]
    cdef int m = arr.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result
    cdef int status

    if axis == 1:
        result = np.empty(n, dtype=np.float64)
        hpcs_reduce_mean_axis1(&arr[0,0], n, m, &result[0], &status)
    else:
        raise ValueError("Only axis=1 currently supported")

    check_status(status, "axis_mean")
    return result

def axis_median(x, int axis=1):
    """Median along specified axis of 2D array."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr = np.asarray(x, dtype=np.float64, order='F')
    cdef int n = arr.shape[0]
    cdef int m = arr.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result
    cdef int status

    if axis == 1:
        result = np.empty(n, dtype=np.float64)
        hpcs_median_axis1(&arr[0,0], n, m, &result[0], &status)
    else:
        raise ValueError("Only axis=1 currently supported")

    check_status(status, "axis_median")
    return result

def axis_mad(x, int axis=1):
    """MAD along specified axis of 2D array."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr = np.asarray(x, dtype=np.float64, order='F')
    cdef int n = arr.shape[0]
    cdef int m = arr.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result
    cdef int status

    if axis == 1:
        result = np.empty(n, dtype=np.float64)
        hpcs_mad_axis1(&arr[0,0], n, m, &result[0], &status)
    else:
        raise ValueError("Only axis=1 currently supported")

    check_status(status, "axis_mad")
    return result

# ==============================================================================
# Python API - Masked Operations (Tier B)
# ==============================================================================

def sum_masked(x, mask):
    """
    Sum of array elements where mask is True/non-zero.

    Parameters
    ----------
    x : array_like
        Input array
    mask : array_like (bool or int)
        Mask array (True/non-zero = valid, False/0 = masked)

    Returns
    -------
    result : float
        Sum of unmasked elements
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[int, ndim=1] mask_arr = np.asarray(mask, dtype=np.int32)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_reduce_sum_masked(&arr[0], <int*>&mask_arr[0], n, &result, &status)
    check_status(status, "sum_masked")

    return result

def mean_masked(x, mask):
    """Mean of array elements where mask is True/non-zero."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[int, ndim=1] mask_arr = np.asarray(mask, dtype=np.int32)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_reduce_mean_masked(&arr[0], <int*>&mask_arr[0], n, &result, &status)
    check_status(status, "mean_masked")

    return result

def var_masked(x, mask):
    """Variance of array elements where mask is True/non-zero."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[int, ndim=1] mask_arr = np.asarray(mask, dtype=np.int32)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_reduce_variance_masked(&arr[0], <int*>&mask_arr[0], n, &result, &status)
    check_status(status, "var_masked")

    return result

def median_masked(x, mask):
    """Median of array elements where mask is True/non-zero."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[int, ndim=1] mask_arr = np.asarray(mask, dtype=np.int32)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_median_masked(&arr[0], <int*>&mask_arr[0], n, &result, &status)
    check_status(status, "median_masked")

    return result

def mad_masked(x, mask):
    """MAD of array elements where mask is True/non-zero."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[int, ndim=1] mask_arr = np.asarray(mask, dtype=np.int32)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_mad_masked(&arr[0], <int*>&mask_arr[0], n, &result, &status)
    check_status(status, "mad_masked")

    return result

# ==============================================================================
# Python API - Additional Functions (Python-level implementations)
# ==============================================================================

def rolling_zscore(x, int window):
    """
    Rolling z-score normalization within a moving window (C-accelerated).

    Computes (x[i] - rolling_mean) / rolling_std for each position.
    More efficient than separate rolling_mean + rolling_std calls.

    Parameters
    ----------
    x : array_like
        Input array
    window : int
        Window size

    Returns
    -------
    result : ndarray
        Rolling z-scores
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_rolling_zscore(&arr[0], n, window, &result[0], &status)
    check_status(status, "rolling_zscore")

    return result

def rolling_robust_zscore(x, int window):
    """
    Rolling robust z-score using median and MAD within a moving window (C-accelerated).

    More resistant to outliers than rolling_zscore.
    More efficient than separate rolling_median + rolling_mad calls.

    Parameters
    ----------
    x : array_like
        Input array
    window : int
        Window size

    Returns
    -------
    result : ndarray
        Rolling robust z-scores
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_rolling_robust_zscore(&arr[0], n, window, &result[0], &status)
    check_status(status, "rolling_robust_zscore")

    return result

def axis_min(x, int axis=1):
    """
    Minimum along specified axis of 2D array.

    Parameters
    ----------
    x : array_like (2D)
        Input 2D array
    axis : int, optional
        Axis along which to compute (default: 1)

    Returns
    -------
    result : ndarray (1D)
        Minimum values along the specified axis
    """
    arr = np.asarray(x, dtype=np.float64, order='F')

    if axis == 1:
        # Min across columns for each row (use NumPy to avoid name conflict)
        result = np.min(arr, axis=1)
        return result
    else:
        raise ValueError("Only axis=1 currently supported")

def axis_max(x, int axis=1):
    """
    Maximum along specified axis of 2D array.

    Parameters
    ----------
    x : array_like (2D)
        Input 2D array
    axis : int, optional
        Axis along which to compute (default: 1)

    Returns
    -------
    result : ndarray (1D)
        Maximum values along the specified axis
    """
    arr = np.asarray(x, dtype=np.float64, order='F')

    if axis == 1:
        # Max across columns for each row (use NumPy to avoid name conflict)
        result = np.max(arr, axis=1)
        return result
    else:
        raise ValueError("Only axis=1 currently supported")

def anomaly_axis(x, int axis=1, double threshold=3.0):
    """
    Anomaly detection along specified axis using z-score method.

    Parameters
    ----------
    x : array_like (2D)
        Input 2D array
    axis : int, optional
        Axis along which to detect anomalies (default: 1)
    threshold : float, optional
        Z-score threshold for anomaly detection (default: 3.0)

    Returns
    -------
    result : ndarray (2D, bool)
        Boolean array marking anomalies (True = anomaly)
    """
    arr = np.asarray(x, dtype=np.float64)

    if axis == 1:
        # Detect anomalies per row
        n, m = arr.shape
        result = np.zeros((n, m), dtype=bool)
        for i in range(n):
            anomalies = detect_anomalies(arr[i, :], threshold)
            result[i, :] = anomalies
        return result
    else:
        raise ValueError("Only axis=1 currently supported")

def anomaly_robust_axis(x, int axis=1, double threshold=3.5):
    """
    Robust anomaly detection along specified axis using MAD-based method.

    Parameters
    ----------
    x : array_like (2D)
        Input 2D array
    axis : int, optional
        Axis along which to detect anomalies (default: 1)
    threshold : float, optional
        Robust z-score threshold (default: 3.5)

    Returns
    -------
    result : ndarray (2D, bool)
        Boolean array marking anomalies (True = anomaly)
    """
    arr = np.asarray(x, dtype=np.float64)

    if axis == 1:
        # Detect anomalies per row using robust method
        n, m = arr.shape
        result = np.zeros((n, m), dtype=bool)
        for i in range(n):
            anomalies = detect_anomalies_robust(arr[i, :], threshold)
            result[i, :] = anomalies
        return result
    else:
        raise ValueError("Only axis=1 currently supported")

def rolling_mean_batched(x, int window, int axis=0):
    """
    Rolling mean on 2D array along specified axis.

    Parameters
    ----------
    x : array_like (2D)
        Input 2D array
    window : int
        Window size
    axis : int, optional
        Axis along which to apply rolling operation (default: 0)

    Returns
    -------
    result : ndarray (2D)
        Rolling means
    """
    arr = np.asarray(x, dtype=np.float64)

    if axis == 0:
        # Apply rolling mean to each column
        n, m = arr.shape
        result = np.empty((n, m), dtype=np.float64)
        for j in range(m):
            result[:, j] = rolling_mean(arr[:, j], window)
        return result
    else:
        # Apply rolling mean to each row
        n, m = arr.shape
        result = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            result[i, :] = rolling_mean(arr[i, :], window)
        return result

def rolling_mean_masked(x, int window, mask):
    """
    Rolling mean with mask (skip invalid values).

    Parameters
    ----------
    x : array_like
        Input array
    window : int
        Window size
    mask : array_like (bool or int)
        Validity mask (True/1 = valid, False/0 = invalid)

    Returns
    -------
    result : ndarray
        Rolling means computed only on valid data
    """
    arr = np.asarray(x, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    cdef int n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    # Simple implementation: compute rolling mean on valid data
    cdef int i, start
    for i in range(n):
        # Compute start index (avoid using max() function which conflicts)
        start = i - window + 1
        if start < 0:
            start = 0

        window_data = arr[start:i+1]
        window_mask = mask_arr[start:i+1]

        # Extract valid data
        valid_data = window_data[window_mask]

        if len(valid_data) > 0:
            result[i] = mean_masked(window_data, window_mask.astype(np.int32))
        else:
            result[i] = np.nan

    return result

# ==============================================================================
# Calibration API (v0.5)
# ==============================================================================

def calibrate(quick=False):
    """
    Run auto-tuning calibration to find optimal parallelization thresholds.

    This benchmarks operations at various array sizes to determine when to
    switch from serial to parallel execution. Results are stored in memory
    and can be saved to a configuration file.

    Parameters
    ----------
    quick : bool, optional
        If True, run quick calibration (5-10 seconds, less accurate).
        If False, run full calibration (30-60 seconds, more accurate).
        Default: False

    Examples
    --------
    >>> import hpcs
    >>> hpcs.calibrate()  # Full calibration
    >>> hpcs.calibrate(quick=True)  # Quick calibration
    """
    cdef int status

    if quick:
        hpcs_calibrate_quick(&status)
    else:
        hpcs_calibrate(&status)

    check_status(status, "calibrate")

def save_calibration_config(path):
    """
    Save calibration configuration to file.

    Parameters
    ----------
    path : str
        Path to configuration file (typically ~/.hpcs/config.json)

    Examples
    --------
    >>> import hpcs
    >>> import os
    >>> hpcs.calibrate()
    >>> hpcs.save_calibration_config(os.path.expanduser("~/.hpcs/config.json"))
    """
    cdef int status
    cdef bytes path_bytes = path.encode('utf-8')

    hpcs_save_config(path_bytes, &status)
    check_status(status, "save_calibration_config")

def load_calibration_config(path):
    """
    Load calibration configuration from file.

    Parameters
    ----------
    path : str
        Path to configuration file

    Examples
    --------
    >>> import hpcs
    >>> import os
    >>> hpcs.load_calibration_config(os.path.expanduser("~/.hpcs/config.json"))
    """
    cdef int status
    cdef bytes path_bytes = path.encode('utf-8')

    hpcs_load_config(path_bytes, &status)
    check_status(status, "load_calibration_config")

# ==============================================================================
# v0.8.0 API - Transform & Robust Statistics
# ==============================================================================

def ewma(x, alpha, mode=None):
    """
    Exponentially weighted moving average.

    Parameters
    ----------
    x : array_like
        Input array
    alpha : float
        Smoothing factor ∈ (0, 1]. Higher alpha = more weight on recent values.
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.
        - 'safe': Full NaN detection (default)
        - 'fast': Skip NaN checks for ~1.5x speedup
        - 'deterministic': Disable SIMD for reproducibility

    Returns
    -------
    result : ndarray
        EWMA of input array

    Examples
    --------
    >>> import hpcs
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> hpcs.ewma(x, alpha=0.5)
    array([1. , 1.5, 2.25, 3.125, 4.0625])

    >>> # Use FAST mode for maximum performance (skip NaN checks)
    >>> hpcs.ewma(x, alpha=0.5, mode='fast')
    array([1. , 1.5, 2.25, 3.125, 4.0625])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_ewma(&arr[0], n, alpha, &result[0], c_mode, &status)
    check_status(status, "ewma")
    return result

def ewvar(x, alpha, mode=None):
    """
    Exponentially weighted variance.

    Parameters
    ----------
    x : array_like
        Input array
    alpha : float
        Smoothing factor ∈ (0, 1]
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : ndarray
        EW variance of input array

    Examples
    --------
    >>> import hpcs
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> hpcs.ewvar(x, alpha=0.3)
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_ewvar(&arr[0], n, alpha, &result[0], c_mode, &status)
    check_status(status, "ewvar")
    return result

def ewstd(x, alpha, mode=None):
    """
    Exponentially weighted standard deviation.

    Parameters
    ----------
    x : array_like
        Input array
    alpha : float
        Smoothing factor ∈ (0, 1]
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : ndarray
        EW standard deviation of input array

    Examples
    --------
    >>> import hpcs
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> hpcs.ewstd(x, alpha=0.4)
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_ewstd(&arr[0], n, alpha, &result[0], c_mode, &status)
    check_status(status, "ewstd")
    return result

def diff(x, order=1):
    """
    Finite differencing (discrete derivative).

    Parameters
    ----------
    x : array_like
        Input array
    order : int, optional
        Lag order (default: 1 for first difference)

    Returns
    -------
    result : ndarray
        Differenced array. First `order` elements are NaN.

    Examples
    --------
    >>> import hpcs
    >>> x = [1.0, 3.0, 6.0, 10.0, 15.0]
    >>> hpcs.diff(x, order=1)  # First differences
    array([nan,  2.,  3.,  4.,  5.])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status

    hpcs_diff(&arr[0], n, order, &result[0], &status)
    check_status(status, "diff")
    return result

def cumulative_min(x, mode=None):
    """
    Cumulative minimum (running minimum).

    Parameters
    ----------
    x : array_like
        Input array
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : ndarray
        Cumulative minimum at each position

    Examples
    --------
    >>> import hpcs
    >>> x = [5.0, 3.0, 4.0, 1.0, 2.0]
    >>> hpcs.cumulative_min(x)
    array([5., 3., 3., 1., 1.])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_cumulative_min(&arr[0], n, &result[0], c_mode, &status)
    check_status(status, "cumulative_min")
    return result

def cumulative_max(x, mode=None):
    """
    Cumulative maximum (running maximum).

    Parameters
    ----------
    x : array_like
        Input array
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : ndarray
        Cumulative maximum at each position

    Examples
    --------
    >>> import hpcs
    >>> x = [1.0, 5.0, 3.0, 7.0, 2.0]
    >>> hpcs.cumulative_max(x)
    array([1., 5., 5., 7., 7.])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_cumulative_max(&arr[0], n, &result[0], c_mode, &status)
    check_status(status, "cumulative_max")
    return result

def convolve_valid(x, kernel, mode=None):
    """
    1D convolution (valid mode, no padding).
    Optimized for small kernels (size 3-15).

    Parameters
    ----------
    x : array_like
        Input signal
    kernel : array_like
        Filter kernel (weights)
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.
        - 'deterministic': Disables OpenMP parallelization for reproducibility

    Returns
    -------
    result : ndarray
        Convolved signal (length = len(x) - len(kernel) + 1)

    Examples
    --------
    >>> import hpcs
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> kernel = [0.25, 0.5, 0.25]  # Smoothing kernel
    >>> hpcs.convolve_valid(x, kernel)
    array([2., 3., 4.])
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] k = ensure_c_contiguous(kernel)
    cdef int n = arr.shape[0]
    cdef int m = k.shape[0]
    cdef int out_n = n - m + 1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(out_n, dtype=np.float64)
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_convolve_valid(&arr[0], n, &k[0], m, &result[0], c_mode, &status)
    check_status(status, "convolve_valid")
    return result

def trimmed_mean(x, trim_frac, mode=None):
    """
    Trimmed mean (discard extremes before averaging).
    Uses deterministic O(n) selection.

    Parameters
    ----------
    x : array_like
        Input array
    trim_frac : float
        Fraction to trim from each tail ∈ [0, 0.5)
        E.g., 0.1 = remove bottom 10% and top 10%
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : float
        Trimmed mean

    Examples
    --------
    >>> import hpcs
    >>> x = [1.0, 2.0, 3.0, 4.0, 100.0]  # outlier
    >>> hpcs.trimmed_mean(x, trim_frac=0.2)  # Remove 1 from each end
    3.0
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_trimmed_mean(&arr[0], n, trim_frac, &result, c_mode, &status)
    check_status(status, "trimmed_mean")
    return result

def winsorized_mean(x, win_frac, mode=None):
    """
    Winsorized mean (clamp extremes before averaging).
    Uses deterministic O(n) selection.

    Parameters
    ----------
    x : array_like
        Input array
    win_frac : float
        Fraction to winsorize from each tail ∈ [0, 0.5)
        E.g., 0.1 = clamp bottom 10% and top 10%
    mode : str, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None (default), uses global mode setting.

    Returns
    -------
    result : float
        Winsorized mean

    Examples
    --------
    >>> import hpcs
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    >>> hpcs.winsorized_mean(x, win_frac=0.1)  # Clamp 1 value each end
    5.5
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status
    cdef int c_mode = _parse_mode(mode)

    hpcs_winsorized_mean(&arr[0], n, win_frac, &result, c_mode, &status)
    check_status(status, "winsorized_mean")
    return result
