"""
Python benchmark for HPCSeries kernels
====================================

Because the execution environment used for this project may not
include a C/C++ compiler, this script implements simple rolling
operations in a vectorised manner using NumPy and measures their
execution time. It serves as a stand-in for the C++ benchmark found
in :file:`bench_core.cpp`. The algorithms reproduce the behaviour of
their C counterparts: a running sum over a sliding window for
``rolling_sum``, an average that divides by either the window size or
the number of elements seen so far for ``rolling_mean``, and a full
reduction for ``reduce_sum``.

To run the benchmark execute:

.. code-block:: sh

    python bench/python_benchmark.py

The output is printed in CSV format with columns ``n``, ``kernel`` and
``elapsed_seconds``.
"""

import numpy as np
from time import perf_counter


def rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Compute a rolling sum over a one‑dimensional array.

    The first ``window`` elements accumulate a prefix sum. Once the
    index exceeds ``window`` the oldest element is subtracted each
    iteration. Returns an array of the same length as ``x``.
    """
    cumsum = np.cumsum(x)
    out = cumsum.copy()
    # For positions >= window subtract the prefix sum that lies window
    # elements behind to maintain a sliding window of length ``window``.
    out[window:] = cumsum[window:] - cumsum[:-window]
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Compute a rolling mean over a one‑dimensional array.

    Uses :func:`rolling_sum` internally and divides by the number of
    elements contributing to the sum: ``i+1`` for the first ``window``
    positions and ``window`` thereafter.
    """
    sums = rolling_sum(x, window)
    n = x.shape[0]
    denom = np.minimum(np.arange(n) + 1, window)
    return sums / denom


def reduce_sum(x: np.ndarray) -> float:
    """Return the sum of all elements in the array."""
    return float(x.sum())


def benchmark():
    sizes = [100_000, 1_000_000, 10_000_000]
    window = 100
    rng = np.random.default_rng(42)
    print("n,kernel,elapsed_seconds")
    for n in sizes:
        x = rng.random(n, dtype=np.float64)
        # rolling_sum
        start = perf_counter()
        _ = rolling_sum(x, window)
        elapsed = perf_counter() - start
        print(f"{n},rolling_sum,{elapsed:.6f}")
        
        # rolling_mean
        start = perf_counter()
        _ = rolling_mean(x, window)
        elapsed = perf_counter() - start
        print(f"{n},rolling_mean,{elapsed:.6f}")
        
        # reduce_sum
        start = perf_counter()
        _ = reduce_sum(x)
        elapsed = perf_counter() - start
        print(f"{n},reduce_sum,{elapsed:.6f}")


if __name__ == "__main__":
    benchmark()