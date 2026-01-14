#!/usr/bin/env python3
"""
Phase 2 Execution Modes Performance Benchmark (v0.8.0)
======================================================

Benchmark the 6 Phase 2 robust statistics functions across execution modes:
  - SAFE: Full NaN detection, validation (baseline)
  - FAST: No validation, no NaN checks (target: 1.3-1.6x faster than SAFE)
  - DETERMINISTIC: Single-threaded, bit-exact (expected: 1.5-2.5x slower than SAFE)

Functions tested:
  - median, mad, quantile
  - rolling_median, rolling_mad, rolling_robust_zscore

Also compares against NumPy/SciPy as reference.
"""

import numpy as np
from scipy import stats
import time
import sys

sys.path.insert(0, '/mnt/c/Users/Samkelo/OneDrive - iqlab/Clients/HPCSeriesCore/python')
import hpcs

# ============================================================================
# Benchmark Infrastructure
# ============================================================================

def benchmark(func, *args, n_runs=10, warmup=2, **kwargs):
    """Time a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Actual timing
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times), result


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_mode_comparison(name, safe_time, fast_time, det_time, size):
    """Print mode comparison."""
    fast_speedup = safe_time / fast_time
    det_slowdown = det_time / safe_time

    # Color codes
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    reset = "\033[0m"

    # Fast should be 1.3-1.6x faster (green if in range)
    fast_color = green if 1.2 < fast_speedup < 2.0 else yellow

    # Deterministic should be 1.5-2.5x slower (acceptable range)
    det_color = green if 1.0 < det_slowdown < 3.0 else red

    print(f"\n{name} (n={size:,})")
    print(f"  SAFE:          {safe_time*1000:8.3f} ms")
    print(f"  FAST:          {fast_time*1000:8.3f} ms  ({fast_color}{fast_speedup:4.2f}x faster{reset})")
    print(f"  DETERMINISTIC: {det_time*1000:8.3f} ms  ({det_color}{det_slowdown:4.2f}x slower{reset})")


def print_numpy_comparison(name, hpcs_time, numpy_time, size):
    """Print HPCS vs NumPy comparison."""
    speedup = numpy_time / hpcs_time
    color = "\033[92m" if speedup > 1.0 else "\033[91m"
    reset = "\033[0m"

    print(f"\n{name} vs NumPy (n={size:,})")
    print(f"  HPCS (SAFE):  {hpcs_time*1000:8.3f} ms")
    print(f"  NumPy:        {numpy_time*1000:8.3f} ms")
    print(f"  Speedup:      {color}{speedup:8.2f}x{reset}")


# ============================================================================
# Basic Robust Statistics Benchmarks
# ============================================================================

def benchmark_median_modes(size):
    """Benchmark median across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.median, data, mode='safe')
    fast_time, _ = benchmark(hpcs.median, data, mode='fast')
    det_time, _ = benchmark(hpcs.median, data, mode='deterministic')
    numpy_time, _ = benchmark(np.median, data)

    print_mode_comparison("median", safe_time, fast_time, det_time, size)
    print_numpy_comparison("median", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_mad_modes(size):
    """Benchmark MAD across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.mad, data, mode='safe')
    fast_time, _ = benchmark(hpcs.mad, data, mode='fast')
    det_time, _ = benchmark(hpcs.mad, data, mode='deterministic')

    # NumPy equivalent: median(abs(x - median(x)))
    def numpy_mad(x):
        return np.median(np.abs(x - np.median(x)))

    numpy_time, _ = benchmark(numpy_mad, data)

    print_mode_comparison("mad", safe_time, fast_time, det_time, size)
    print_numpy_comparison("mad", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_quantile_modes(size):
    """Benchmark quantile across all execution modes."""
    data = np.random.randn(size)
    q = 0.75

    safe_time, _ = benchmark(hpcs.quantile, data, q, mode='safe')
    fast_time, _ = benchmark(hpcs.quantile, data, q, mode='fast')
    det_time, _ = benchmark(hpcs.quantile, data, q, mode='deterministic')
    numpy_time, _ = benchmark(np.quantile, data, q)

    print_mode_comparison("quantile (q=0.75)", safe_time, fast_time, det_time, size)
    print_numpy_comparison("quantile", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


# ============================================================================
# Rolling Robust Statistics Benchmarks
# ============================================================================

def benchmark_rolling_median_modes(size, window=50):
    """Benchmark rolling_median across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.rolling_median, data, window, mode='safe')
    fast_time, _ = benchmark(hpcs.rolling_median, data, window, mode='fast')
    det_time, _ = benchmark(hpcs.rolling_median, data, window, mode='deterministic')

    # NumPy rolling median (simple implementation for comparison)
    def numpy_rolling_median(x, w):
        result = np.empty(len(x))
        result[:w-1] = np.nan
        for i in range(w-1, len(x)):
            result[i] = np.median(x[i-w+1:i+1])
        return result

    numpy_time, _ = benchmark(numpy_rolling_median, data, window)

    print_mode_comparison(f"rolling_median (window={window})", safe_time, fast_time, det_time, size)
    print_numpy_comparison("rolling_median", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_rolling_mad_modes(size, window=50):
    """Benchmark rolling_mad across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.rolling_mad, data, window, mode='safe')
    fast_time, _ = benchmark(hpcs.rolling_mad, data, window, mode='fast')
    det_time, _ = benchmark(hpcs.rolling_mad, data, window, mode='deterministic')

    print_mode_comparison(f"rolling_mad (window={window})", safe_time, fast_time, det_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time}


def benchmark_rolling_robust_zscore_modes(size, window=50):
    """Benchmark rolling_robust_zscore across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.rolling_robust_zscore, data, window, mode='safe')
    fast_time, _ = benchmark(hpcs.rolling_robust_zscore, data, window, mode='fast')
    det_time, _ = benchmark(hpcs.rolling_robust_zscore, data, window, mode='deterministic')

    print_mode_comparison(f"rolling_robust_zscore (window={window})", safe_time, fast_time, det_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time}


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def run_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 80)
    print("  Phase 2 Execution Modes Performance Benchmark (v0.8.0)")
    print("=" * 80)
    print("\nTargets:")
    print("  - FAST mode: 1.3-1.6x faster than SAFE")
    print("  - DETERMINISTIC mode: 1.5-2.5x slower than SAFE (acceptable for reproducibility)")
    print()

    sizes = [10_000, 100_000, 1_000_000]

    for size in sizes:
        print_header(f"Array Size: n={size:,}")

        # Basic robust statistics
        benchmark_median_modes(size)
        benchmark_mad_modes(size)
        benchmark_quantile_modes(size)

        # Rolling robust statistics (smaller window for large arrays)
        window = min(100, size // 100)
        benchmark_rolling_median_modes(size, window)
        benchmark_rolling_mad_modes(size, window)
        benchmark_rolling_robust_zscore_modes(size, window)

    print("\n" + "=" * 80)
    print("  Benchmark Complete!")
    print("=" * 80)
    print("\nInterpretation:")
    print("  - Green: Within expected performance range")
    print("  - Yellow: Slightly outside expected range (may need investigation)")
    print("  - Red: Significantly outside expected range (needs optimization)")
    print()


if __name__ == '__main__':
    run_benchmarks()
