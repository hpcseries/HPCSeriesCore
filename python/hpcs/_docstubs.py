"""
Documentation stubs for HPCSeries Core API.

This module provides function signatures and docstrings for Sphinx documentation
when the compiled Cython extensions are not available (e.g., on Read the Docs).
These are only used for documentation generation, not at runtime.
"""

def set_execution_mode(mode):
    """
    Set the global execution mode for all HPCSeries operations.

    The execution mode controls the trade-off between safety, performance,
    and reproducibility. Individual function calls can override the global
    mode using the `mode` parameter.

    Parameters
    ----------
    mode : int or str
        Execution mode to set. Can be:
        - 'safe' or MODE_SAFE (0): IEEE 754 compliant, full error checking
        - 'fast' or MODE_FAST (1): Relaxed math, minimal validation
        - 'deterministic' or MODE_DETERMINISTIC (2): Bit-exact reproducibility

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If mode is not one of the valid values.

    Examples
    --------
    >>> import hpcs
    >>> hpcs.set_execution_mode('safe')  # Maximum safety
    >>> hpcs.set_execution_mode('fast')  # Maximum performance
    >>> hpcs.set_execution_mode('deterministic')  # Exact reproducibility

    See Also
    --------
    get_execution_mode : Get the current global execution mode
    MODE_SAFE, MODE_FAST, MODE_DETERMINISTIC : Execution mode constants

    Notes
    -----
    The execution mode affects:

    - **SAFE mode** (default):
      - Full IEEE 754 compliance (NaN/Inf handling)
      - Input validation and error checking
      - OpenMP parallelization enabled
      - Best for: Production code, untrusted data

    - **FAST mode**:
      - Relaxed floating-point math (faster but less precise)
      - Minimal input validation
      - OpenMP parallelization enabled
      - Best for: Performance-critical inner loops with validated data

    - **DETERMINISTIC mode**:
      - IEEE 754 compliance
      - Full error checking
      - **No OpenMP** (single-threaded for reproducibility)
      - Best for: Unit tests, scientific reproducibility

    Notes
    -----
    **Thread Safety**: The global mode is stored in thread-local storage
    (OpenMP threadprivate). Each thread has its own independent mode setting.
    """
    pass


def get_execution_mode():
    """
    Get the current global execution mode.

    Returns
    -------
    mode : int
        Current execution mode:
        - 0 = MODE_SAFE (default)
        - 1 = MODE_FAST
        - 2 = MODE_DETERMINISTIC

    Examples
    --------
    >>> import hpcs
    >>> hpcs.get_execution_mode()
    0  # MODE_SAFE (default)
    >>> hpcs.set_execution_mode('fast')
    >>> hpcs.get_execution_mode()
    1  # MODE_FAST

    See Also
    --------
    set_execution_mode : Set the global execution mode
    """
    pass


def ewma(x, alpha, mode=None):
    """
    Exponential weighted moving average (EWMA).

    Computes the exponentially weighted moving average with decay factor alpha.
    15-60x faster than pandas.ewm().mean().

    Parameters
    ----------
    x : array_like
        Input array (1D)
    alpha : float
        Decay factor (0 < alpha <= 1). Higher values give more weight to recent data.
    mode : str or int, optional
        Execution mode: 'safe', 'fast', or 'deterministic'.
        If None, uses global mode (default: 'safe').

    Returns
    -------
    result : ndarray
        EWMA values (same shape as input)

    Examples
    --------
    >>> import hpcs
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> hpcs.ewma(x, alpha=0.5)
    array([1.  , 1.5 , 2.25, 3.12, 4.06])

    >>> # Fast mode for performance-critical code
    >>> hpcs.ewma(x, alpha=0.3, mode='fast')

    See Also
    --------
    ewvar : Exponential weighted variance
    ewstd : Exponential weighted standard deviation
    """
    pass


def ewvar(x, alpha, mode=None):
    """
    Exponential weighted variance (EWVAR).

    Computes the exponentially weighted variance using Welford's online algorithm.

    Parameters
    ----------
    x : array_like
        Input array (1D)
    alpha : float
        Decay factor (0 < alpha <= 1)
    mode : str or int, optional
        Execution mode ('safe', 'fast', or 'deterministic')

    Returns
    -------
    result : ndarray
        EWVAR values (same shape as input)

    See Also
    --------
    ewma : Exponential weighted mean
    ewstd : Exponential weighted standard deviation
    """
    pass


def ewstd(x, alpha, mode=None):
    """
    Exponential weighted standard deviation (EWSTD).

    Computes the exponentially weighted standard deviation (sqrt of EWVAR).

    Parameters
    ----------
    x : array_like
        Input array (1D)
    alpha : float
        Decay factor (0 < alpha <= 1)
    mode : str or int, optional
        Execution mode ('safe', 'fast', or 'deterministic')

    Returns
    -------
    result : ndarray
        EWSTD values (same shape as input)

    See Also
    --------
    ewma : Exponential weighted mean
    ewvar : Exponential weighted variance
    """
    pass


def cumulative_min(x, mode=None):
    """
    Cumulative minimum (running minimum).

    Computes the minimum value seen so far at each position.

    Parameters
    ----------
    x : array_like
        Input array (1D)
    mode : str or int, optional
        Execution mode ('safe', 'fast', or 'deterministic')

    Returns
    -------
    result : ndarray
        Cumulative minimum (same shape as input)

    Examples
    --------
    >>> import hpcs
    >>> x = np.array([5.0, 3.0, 7.0, 1.0, 4.0])
    >>> hpcs.cumulative_min(x)
    array([5., 3., 3., 1., 1.])

    See Also
    --------
    cumulative_max : Cumulative maximum
    """
    pass


def cumulative_max(x, mode=None):
    """
    Cumulative maximum (running maximum).

    Computes the maximum value seen so far at each position.

    Parameters
    ----------
    x : array_like
        Input array (1D)
    mode : str or int, optional
        Execution mode ('safe', 'fast', or 'deterministic')

    Returns
    -------
    result : ndarray
        Cumulative maximum (same shape as input)

    See Also
    --------
    cumulative_min : Cumulative minimum
    """
    pass


def convolve_valid(x, kernel, mode=None):
    """
    1D convolution (valid mode, no padding).

    Optimized for small kernels (size 3-15) with template specializations.
    Uses OpenMP parallelization in SAFE/FAST modes.

    Parameters
    ----------
    x : array_like
        Input signal (1D)
    kernel : array_like
        Filter kernel/weights (1D)
    mode : str or int, optional
        Execution mode ('safe', 'fast', or 'deterministic')

    Returns
    -------
    result : ndarray
        Convolved signal, length = len(x) - len(kernel) + 1

    Examples
    --------
    >>> import hpcs
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> kernel = np.array([0.25, 0.5, 0.25])  # Smoothing
    >>> hpcs.convolve_valid(x, kernel)
    array([2., 3., 4.])

    Notes
    -----
    - Uses template specializations for kernel sizes 3, 5, 7, 9, 11, 13, 15
    - Deterministic mode disables OpenMP for bit-exact reproducibility
    """
    pass


def trimmed_mean(x, trim_frac, mode=None):
    """
    Trimmed mean (discard extremes before averaging).

    Removes the specified fraction of smallest and largest values,
    then computes the mean of the remaining values. 10-15x faster than SciPy.

    Parameters
    ----------
    x : array_like
        Input array (1D)
    trim_frac : float
        Fraction to trim from each end (0 <= trim_frac < 0.5)
    mode : str or int, optional
        Execution mode ('safe', 'fast', or 'deterministic')

    Returns
    -------
    result : float
        Trimmed mean value

    Examples
    --------
    >>> import hpcs
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # 100 is outlier
    >>> hpcs.trimmed_mean(x, trim_frac=0.2)
    3.0  # Excludes 1 and 100

    See Also
    --------
    winsorized_mean : Winsorized mean (clip instead of trim)
    """
    pass


def winsorized_mean(x, win_frac, mode=None):
    """
    Winsorized mean (clip extremes before averaging).

    Replaces the specified fraction of smallest and largest values
    with the adjacent non-extreme values, then computes the mean.

    Parameters
    ----------
    x : array_like
        Input array (1D)
    win_frac : float
        Fraction to winsorize from each end (0 <= win_frac < 0.5)
    mode : str or int, optional
        Execution mode ('safe', 'fast', or 'deterministic')

    Returns
    -------
    result : float
        Winsorized mean value

    Examples
    --------
    >>> import hpcs
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # 100 is outlier
    >>> hpcs.winsorized_mean(x, win_frac=0.2)
    4.6  # Replaces 100 with 4

    See Also
    --------
    trimmed_mean : Trimmed mean (remove instead of clip)
    """
    pass


# Execution mode constants
MODE_SAFE = 0
MODE_FAST = 1
MODE_DETERMINISTIC = 2
