#!/usr/bin/env python3
"""
Phase 1 Execution Modes Performance Benchmark (v0.8.0)
=======================================================

Benchmark the 9 Phase 1 reduction functions across execution modes:
  - SAFE: Full NaN detection, validation (baseline)
  - FAST: No validation, no NaN checks (target: 1.3-1.6x faster than SAFE)
  - DETERMINISTIC: Single-threaded, bit-exact (expected: 1.5-2.5x slower than SAFE)

Functions tested:
  - sum, min, max, mean, var, std
  - group_sum, group_mean, group_var

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
# Basic Reduction Benchmarks
# ============================================================================

def benchmark_sum_modes(size):
    """Benchmark sum across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.sum, data, mode='safe')
    fast_time, _ = benchmark(hpcs.sum, data, mode='fast')
    det_time, _ = benchmark(hpcs.sum, data, mode='deterministic')
    numpy_time, _ = benchmark(np.sum, data)

    print_mode_comparison("sum", safe_time, fast_time, det_time, size)
    print_numpy_comparison("sum", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_mean_modes(size):
    """Benchmark mean across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.mean, data, mode='safe')
    fast_time, _ = benchmark(hpcs.mean, data, mode='fast')
    det_time, _ = benchmark(hpcs.mean, data, mode='deterministic')
    numpy_time, _ = benchmark(np.mean, data)

    print_mode_comparison("mean", safe_time, fast_time, det_time, size)
    print_numpy_comparison("mean", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_var_modes(size):
    """Benchmark variance across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.var, data, mode='safe')
    fast_time, _ = benchmark(hpcs.var, data, mode='fast')
    det_time, _ = benchmark(hpcs.var, data, mode='deterministic')
    numpy_time, _ = benchmark(np.var, data, ddof=0)  # Population variance

    print_mode_comparison("var", safe_time, fast_time, det_time, size)
    print_numpy_comparison("var", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_std_modes(size):
    """Benchmark std across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.std, data, mode='safe')
    fast_time, _ = benchmark(hpcs.std, data, mode='fast')
    det_time, _ = benchmark(hpcs.std, data, mode='deterministic')
    numpy_time, _ = benchmark(np.std, data, ddof=0)

    print_mode_comparison("std", safe_time, fast_time, det_time, size)
    print_numpy_comparison("std", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_min_modes(size):
    """Benchmark min across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.min, data, mode='safe')
    fast_time, _ = benchmark(hpcs.min, data, mode='fast')
    det_time, _ = benchmark(hpcs.min, data, mode='deterministic')
    numpy_time, _ = benchmark(np.min, data)

    print_mode_comparison("min", safe_time, fast_time, det_time, size)
    print_numpy_comparison("min", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_max_modes(size):
    """Benchmark max across all execution modes."""
    data = np.random.randn(size)

    safe_time, _ = benchmark(hpcs.max, data, mode='safe')
    fast_time, _ = benchmark(hpcs.max, data, mode='fast')
    det_time, _ = benchmark(hpcs.max, data, mode='deterministic')
    numpy_time, _ = benchmark(np.max, data)

    print_mode_comparison("max", safe_time, fast_time, det_time, size)
    print_numpy_comparison("max", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


# ============================================================================
# Grouped Reduction Benchmarks
# ============================================================================

def benchmark_group_sum_modes(size, n_groups=100):
    """Benchmark group_sum across all execution modes."""
    data = np.random.randn(size)
    group_ids = np.random.randint(0, n_groups, size=size, dtype=np.int32)

    safe_time, _ = benchmark(hpcs.group_sum, data, group_ids, n_groups, mode='safe')
    fast_time, _ = benchmark(hpcs.group_sum, data, group_ids, n_groups, mode='fast')
    det_time, _ = benchmark(hpcs.group_sum, data, group_ids, n_groups, mode='deterministic')

    # NumPy groupby equivalent (using np.bincount for comparison)
    def numpy_group_sum(data, groups, n_groups):
        return np.bincount(groups, weights=data, minlength=n_groups)

    numpy_time, _ = benchmark(numpy_group_sum, data, group_ids, n_groups)

    print_mode_comparison("group_sum", safe_time, fast_time, det_time, size)
    print_numpy_comparison("group_sum", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_group_mean_modes(size, n_groups=100):
    """Benchmark group_mean across all execution modes."""
    data = np.random.randn(size)
    group_ids = np.random.randint(0, n_groups, size=size, dtype=np.int32)

    safe_time, _ = benchmark(hpcs.group_mean, data, group_ids, n_groups, mode='safe')
    fast_time, _ = benchmark(hpcs.group_mean, data, group_ids, n_groups, mode='fast')
    det_time, _ = benchmark(hpcs.group_mean, data, group_ids, n_groups, mode='deterministic')

    # NumPy groupby mean
    def numpy_group_mean(data, groups, n_groups):
        sums = np.bincount(groups, weights=data, minlength=n_groups)
        counts = np.bincount(groups, minlength=n_groups)
        return np.divide(sums, counts, out=np.full(n_groups, np.nan), where=counts != 0)

    numpy_time, _ = benchmark(numpy_group_mean, data, group_ids, n_groups)

    print_mode_comparison("group_mean", safe_time, fast_time, det_time, size)
    print_numpy_comparison("group_mean", safe_time, numpy_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time, 'numpy': numpy_time}


def benchmark_group_var_modes(size, n_groups=100):
    """Benchmark group_var across all execution modes."""
    data = np.random.randn(size)
    group_ids = np.random.randint(0, n_groups, size=size, dtype=np.int32)

    safe_time, _ = benchmark(hpcs.group_var, data, group_ids, n_groups, mode='safe')
    fast_time, _ = benchmark(hpcs.group_var, data, group_ids, n_groups, mode='fast')
    det_time, _ = benchmark(hpcs.group_var, data, group_ids, n_groups, mode='deterministic')

    print_mode_comparison("group_var", safe_time, fast_time, det_time, size)

    return {'safe': safe_time, 'fast': fast_time, 'deterministic': det_time}


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def run_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 80)
    print("  Phase 1 Execution Modes Performance Benchmark (v0.8.0)")
    print("=" * 80)
    print("\nTargets:")
    print("  - FAST mode: 1.3-1.6x faster than SAFE")
    print("  - DETERMINISTIC mode: 1.5-2.5x slower than SAFE (acceptable for reproducibility)")
    print()

    sizes = [10_000, 100_000, 1_000_000]

    for size in sizes:
        print_header(f"Array Size: n={size:,}")

        # Basic reductions
        benchmark_sum_modes(size)
        benchmark_mean_modes(size)
        benchmark_var_modes(size)
        benchmark_std_modes(size)
        benchmark_min_modes(size)
        benchmark_max_modes(size)

        # Grouped reductions
        n_groups = min(1000, size // 100)
        benchmark_group_sum_modes(size, n_groups)
        benchmark_group_mean_modes(size, n_groups)
        benchmark_group_var_modes(size, n_groups)

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
