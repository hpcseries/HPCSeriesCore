#!/usr/bin/env python3
"""
HPCSeries v0.8.0 Performance Benchmark
=======================================

Compare Fortran/C++ implementations against pandas/NumPy/SciPy to measure
actual speedups and justify the implementation effort.

Key Questions:
1. Are we faster than pandas for EWMA/EWVAR/EWSTD?
2. Are we faster than NumPy for diff/cumulative ops?
3. Are we faster than SciPy for robust statistics?
4. What's the speedup as a function of array size?
"""

import numpy as np
import pandas as pd
from scipy import stats
import time
import sys

sys.path.insert(0, '/mnt/c/Users/Samkelo/OneDrive - iqlab/Clients/HPCSeriesCore')
import hpcs

# ============================================================================
# Benchmark Infrastructure
# ============================================================================

def benchmark(func, *args, n_runs=5, warmup=1, **kwargs):
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


def print_comparison(name, hpcs_time, ref_time, ref_name, size):
    """Print benchmark comparison."""
    speedup = ref_time / hpcs_time
    color = "\033[92m" if speedup > 1.0 else "\033[91m"  # Green if faster, red if slower
    reset = "\033[0m"

    print(f"\n{name} (n={size:,})")
    print(f"  HPCS:      {hpcs_time*1000:8.3f} ms")
    print(f"  {ref_name:9}: {ref_time*1000:8.3f} ms")
    print(f"  Speedup:   {color}{speedup:8.2f}x{reset}")


# ============================================================================
# Benchmarks by Category
# ============================================================================

def benchmark_ewma(size):
    """Benchmark EWMA against pandas."""
    data = np.random.randn(size)
    alpha = 0.3

    hpcs_time, hpcs_result = benchmark(hpcs.ewma, data, alpha=alpha)
    pandas_time, pandas_result = benchmark(
        lambda d, a: pd.Series(d).ewm(alpha=a, adjust=False).mean().values,
        data, alpha
    )

    print_comparison("EWMA", hpcs_time, pandas_time, "pandas", size)
    return hpcs_time, pandas_time


def benchmark_ewvar(size):
    """Benchmark EWVAR against pandas."""
    data = np.random.randn(size)
    alpha = 0.3

    hpcs_time, hpcs_result = benchmark(hpcs.ewvar, data, alpha=alpha)
    pandas_time, pandas_result = benchmark(
        lambda d, a: pd.Series(d).ewm(alpha=a, adjust=False).var(bias=True).values,
        data, alpha
    )

    print_comparison("EWVAR", hpcs_time, pandas_time, "pandas", size)
    return hpcs_time, pandas_time


def benchmark_ewstd(size):
    """Benchmark EWSTD against pandas."""
    data = np.random.randn(size)
    alpha = 0.25

    hpcs_time, hpcs_result = benchmark(hpcs.ewstd, data, alpha=alpha)
    pandas_time, pandas_result = benchmark(
        lambda d, a: pd.Series(d).ewm(alpha=a, adjust=False).std(bias=True).values,
        data, alpha
    )

    print_comparison("EWSTD", hpcs_time, pandas_time, "pandas", size)
    return hpcs_time, pandas_time


def benchmark_diff(size):
    """Benchmark differencing against NumPy."""
    data = np.random.randn(size)
    order = 1

    hpcs_time, hpcs_result = benchmark(hpcs.diff, data, order=order)
    numpy_time, numpy_result = benchmark(np.diff, data, n=order)

    print_comparison("diff (order=1)", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_cumulative_min(size):
    """Benchmark cumulative min against NumPy."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.cumulative_min, data)
    numpy_time, numpy_result = benchmark(np.minimum.accumulate, data)

    print_comparison("cumulative_min", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_cumulative_max(size):
    """Benchmark cumulative max against NumPy."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.cumulative_max, data)
    numpy_time, numpy_result = benchmark(np.maximum.accumulate, data)

    print_comparison("cumulative_max", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_convolve(size):
    """Benchmark convolution against NumPy."""
    data = np.random.randn(size)
    kernel = np.array([1.0, 2.0, 1.0]) / 4.0  # Size 3 kernel

    hpcs_time, hpcs_result = benchmark(hpcs.convolve_valid, data, kernel)
    numpy_time, numpy_result = benchmark(np.convolve, data, kernel, mode='valid')

    print_comparison("convolve_valid (k=3)", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_trimmed_mean(size):
    """Benchmark trimmed mean against SciPy."""
    data = np.random.randn(size)
    trim_frac = 0.2

    hpcs_time, hpcs_result = benchmark(hpcs.trimmed_mean, data, trim_frac=trim_frac)
    scipy_time, scipy_result = benchmark(stats.trim_mean, data, trim_frac)

    print_comparison("trimmed_mean (20%)", hpcs_time, scipy_time, "SciPy", size)
    return hpcs_time, scipy_time


def benchmark_winsorized_mean(size):
    """Benchmark winsorized mean against manual calculation."""
    data = np.random.randn(size)
    win_frac = 0.1

    def scipy_winsorized(d, frac):
        from scipy.stats.mstats import winsorize
        return np.mean(winsorize(d, limits=(frac, frac)))

    hpcs_time, hpcs_result = benchmark(hpcs.winsorized_mean, data, win_frac=win_frac)
    scipy_time, scipy_result = benchmark(scipy_winsorized, data, win_frac)

    print_comparison("winsorized_mean (10%)", hpcs_time, scipy_time, "SciPy", size)
    return hpcs_time, scipy_time


# ============================================================================
# Rolling Operations (Tier 1)
# ============================================================================

def benchmark_rolling_mean(size):
    """Benchmark rolling mean against pandas."""
    data = np.random.randn(size)
    window = 50

    hpcs_time, hpcs_result = benchmark(hpcs.rolling_mean, data, window=window)
    pandas_time, pandas_result = benchmark(
        lambda d, w: pd.Series(d).rolling(window=w).mean().values,
        data, window
    )

    print_comparison(f"rolling_mean (w={window})", hpcs_time, pandas_time, "pandas", size)
    return hpcs_time, pandas_time


def benchmark_rolling_std(size):
    """Benchmark rolling std against pandas."""
    data = np.random.randn(size)
    window = 50

    hpcs_time, hpcs_result = benchmark(hpcs.rolling_std, data, window=window)
    pandas_time, pandas_result = benchmark(
        lambda d, w: pd.Series(d).rolling(window=w).std().values,
        data, window
    )

    print_comparison(f"rolling_std (w={window})", hpcs_time, pandas_time, "pandas", size)
    return hpcs_time, pandas_time


def benchmark_rolling_var(size):
    """Benchmark rolling var against pandas."""
    data = np.random.randn(size)
    window = 50

    hpcs_time, hpcs_result = benchmark(hpcs.rolling_var, data, window=window)
    pandas_time, pandas_result = benchmark(
        lambda d, w: pd.Series(d).rolling(window=w).var().values,
        data, window
    )

    print_comparison(f"rolling_var (w={window})", hpcs_time, pandas_time, "pandas", size)
    return hpcs_time, pandas_time


# ============================================================================
# Tier 2: Core Reductions
# ============================================================================

def benchmark_sum(size):
    """Benchmark sum against NumPy."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.sum, data)
    numpy_time, numpy_result = benchmark(np.sum, data)

    print_comparison("sum", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_mean(size):
    """Benchmark mean against NumPy."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.mean, data)
    numpy_time, numpy_result = benchmark(np.mean, data)

    print_comparison("mean", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_var(size):
    """Benchmark variance against NumPy."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.var, data)
    numpy_time, numpy_result = benchmark(np.var, data)

    print_comparison("var", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_std(size):
    """Benchmark std against NumPy."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.std, data)
    numpy_time, numpy_result = benchmark(np.std, data)

    print_comparison("std", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_min(size):
    """Benchmark min against NumPy."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.min, data)
    numpy_time, numpy_result = benchmark(np.min, data)

    print_comparison("min", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_max(size):
    """Benchmark max against NumPy."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.max, data)
    numpy_time, numpy_result = benchmark(np.max, data)

    print_comparison("max", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


# ============================================================================
# Tier 3: Transforms
# ============================================================================

def benchmark_zscore(size):
    """Benchmark z-score against scipy.stats.zscore."""
    data = np.random.randn(size)

    hpcs_time, hpcs_result = benchmark(hpcs.zscore, data)
    scipy_time, scipy_result = benchmark(stats.zscore, data)

    print_comparison("zscore", hpcs_time, scipy_time, "SciPy", size)
    return hpcs_time, scipy_time


def benchmark_normalize_minmax(size):
    """Benchmark min-max normalization against manual implementation."""
    data = np.random.randn(size)

    def numpy_normalize(d):
        min_val = np.min(d)
        max_val = np.max(d)
        return (d - min_val) / (max_val - min_val)

    hpcs_time, hpcs_result = benchmark(hpcs.normalize_minmax, data)
    numpy_time, numpy_result = benchmark(numpy_normalize, data)

    print_comparison("normalize_minmax", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


def benchmark_clip(size):
    """Benchmark clip against NumPy."""
    data = np.random.randn(size)
    lower = -2.0
    upper = 2.0

    hpcs_time, hpcs_result = benchmark(hpcs.clip, data, lower, upper)
    numpy_time, numpy_result = benchmark(np.clip, data, lower, upper)

    print_comparison("clip", hpcs_time, numpy_time, "NumPy", size)
    return hpcs_time, numpy_time


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def run_benchmarks():
    """Run all benchmarks and generate report."""
    print_header("HPCSeries v0.8.0 Performance Benchmark")
    print("\nTesting speedup vs pandas/NumPy/SciPy across different array sizes")
    print("Green = faster, Red = slower")

    sizes = [1_000, 10_000, 100_000, 1_000_000]

    results = {
        # v0.8.0 functions
        'EWMA': [],
        'EWVAR': [],
        'EWSTD': [],
        'diff': [],
        'cumulative_min': [],
        'cumulative_max': [],
        'convolve': [],
        'trimmed_mean': [],
        'winsorized_mean': [],

        # Tier 1: Rolling operations
        'rolling_mean': [],
        'rolling_std': [],
        'rolling_var': [],

        # Tier 2: Core reductions
        'sum': [],
        'mean': [],
        'var': [],
        'std': [],
        'min': [],
        'max': [],

        # Tier 3: Transforms
        'zscore': [],
        'normalize_minmax': [],
        'clip': []
    }

    for size in sizes:
        print_header(f"Array Size: {size:,} elements")

        # v0.8.0 functions
        results['EWMA'].append(benchmark_ewma(size))
        results['EWVAR'].append(benchmark_ewvar(size))
        results['EWSTD'].append(benchmark_ewstd(size))
        results['diff'].append(benchmark_diff(size))
        results['cumulative_min'].append(benchmark_cumulative_min(size))
        results['cumulative_max'].append(benchmark_cumulative_max(size))
        results['convolve'].append(benchmark_convolve(size))
        results['trimmed_mean'].append(benchmark_trimmed_mean(size))
        results['winsorized_mean'].append(benchmark_winsorized_mean(size))

        # Tier 1: Rolling operations
        results['rolling_mean'].append(benchmark_rolling_mean(size))
        results['rolling_std'].append(benchmark_rolling_std(size))
        results['rolling_var'].append(benchmark_rolling_var(size))

        # Tier 2: Core reductions
        results['sum'].append(benchmark_sum(size))
        results['mean'].append(benchmark_mean(size))
        results['var'].append(benchmark_var(size))
        results['std'].append(benchmark_std(size))
        results['min'].append(benchmark_min(size))
        results['max'].append(benchmark_max(size))

        # Tier 3: Transforms
        results['zscore'].append(benchmark_zscore(size))
        results['normalize_minmax'].append(benchmark_normalize_minmax(size))
        results['clip'].append(benchmark_clip(size))

    # Summary
    print_header("Summary Report")

    print("\nAverage Speedup by Function (across all sizes):")
    print("-" * 60)

    for func_name, timings in results.items():
        speedups = [ref_t / hpcs_t for hpcs_t, ref_t in timings]
        avg_speedup = np.mean(speedups)
        color = "\033[92m" if avg_speedup > 1.0 else "\033[91m"
        reset = "\033[0m"
        print(f"  {func_name:20} {color}{avg_speedup:6.2f}x{reset}")

    print("\n" + "=" * 80)
    print("\nKey Insights:")
    print("-" * 60)

    # Calculate overall stats
    all_speedups = []
    for timings in results.values():
        all_speedups.extend([ref_t / hpcs_t for hpcs_t, ref_t in timings])

    overall_avg = np.mean(all_speedups)
    overall_median = np.median(all_speedups)
    overall_min = np.min(all_speedups)
    overall_max = np.max(all_speedups)

    print(f"\nOverall Performance:")
    print(f"  Average Speedup:  {overall_avg:.2f}x")
    print(f"  Median Speedup:   {overall_median:.2f}x")
    print(f"  Min Speedup:      {overall_min:.2f}x")
    print(f"  Max Speedup:      {overall_max:.2f}x")

    # Analysis
    faster_count = sum(1 for s in all_speedups if s > 1.0)
    total_count = len(all_speedups)

    print(f"\nWin Rate: {faster_count}/{total_count} ({100*faster_count/total_count:.0f}%) cases faster than reference")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 60)

    if overall_median > 2.0:
        print("✓ STRONG CASE: 2x+ median speedup justifies Fortran/C++ implementation")
    elif overall_median > 1.5:
        print("✓ MODERATE CASE: 1.5x+ median speedup provides value for large-scale workloads")
    elif overall_median > 1.0:
        print("? WEAK CASE: <1.5x speedup may not justify maintenance overhead")
    else:
        print("✗ NO CASE: Slower than reference - investigate optimizations or reconsider")

    print("\nKey Areas:")
    for func_name, timings in results.items():
        speedups = [ref_t / hpcs_t for hpcs_t, ref_t in timings]
        avg_speedup = np.mean(speedups)

        if avg_speedup < 1.0:
            print(f"  ⚠ {func_name}: SLOWER - needs optimization")
        elif avg_speedup > 2.0:
            print(f"  ✓ {func_name}: FAST - clear win")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_benchmarks()
