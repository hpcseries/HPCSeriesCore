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
    # Status codes
    int HPCS_SUCCESS
    int HPCS_ERR_INVALID_ARGS

    # Reduction functions (Fortran interface)
    void hpcs_reduce_sum(const double *x, int n, double *out, int *status)
    void hpcs_reduce_mean(const double *x, int n, double *out, int *status)
    void hpcs_reduce_variance(const double *x, int n, double *out, int *status)
    void hpcs_reduce_min(const double *x, int n, double *out, int *status)
    void hpcs_reduce_max(const double *x, int n, double *out, int *status)

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

    # Rolling operations (v0.3)
    void hpcs_rolling_mean(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_std(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_var(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_median(const double *x, int n, int window, double *y, int *status)
    void hpcs_rolling_mad(const double *x, int n, int window, double *y, int *status)

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
# Python API - Reductions
# ==============================================================================

def sum(x):
    """
    Sum of array elements (SIMD-accelerated).

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : float
        Sum of all elements

    Examples
    --------
    >>> import hpcs
    >>> hpcs.sum([1, 2, 3, 4, 5])
    15.0
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_reduce_sum_simd(&arr[0], n, &result, &status)
    check_status(status, "sum")

    return result

def mean(x):
    """
    Mean of array elements (SIMD-accelerated).

    Parameters
    ----------
    x : array_like
        Input array

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

    hpcs_reduce_mean_simd(&arr[0], n, &result, &status)
    check_status(status, "mean")

    return result

def var(x):
    """
    Variance of array elements (SIMD-accelerated).

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : float
        Sample variance

    Examples
    --------
    >>> import hpcs
    >>> hpcs.var([1, 2, 3, 4, 5])
    2.5
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_reduce_variance_simd(&arr[0], n, &result, &status)
    check_status(status, "var")

    return result

def std(x):
    """
    Standard deviation of array elements (SIMD-accelerated).

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : float
        Sample standard deviation

    Examples
    --------
    >>> import hpcs
    >>> hpcs.std([1, 2, 3, 4, 5])
    1.5811388300841898
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_reduce_std_simd(&arr[0], n, &result, &status)
    check_status(status, "std")

    return result

def min(x):
    """
    Minimum of array elements (SIMD-accelerated).

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : float
        Minimum value
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_reduce_min_simd(&arr[0], n, &result, &status)
    check_status(status, "min")

    return result

def max(x):
    """
    Maximum of array elements (SIMD-accelerated).

    Parameters
    ----------
    x : array_like
        Input array

    Returns
    -------
    result : float
        Maximum value
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = ensure_c_contiguous(x)
    cdef int n = arr.shape[0]
    cdef double result
    cdef int status

    hpcs_reduce_max_simd(&arr[0], n, &result, &status)
    check_status(status, "max")

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

# ==============================================================================
# Python API - Rolling Operations
# ==============================================================================

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

    hpcs_rolling_var(&arr[0], n, window, &result[0], &status)
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
