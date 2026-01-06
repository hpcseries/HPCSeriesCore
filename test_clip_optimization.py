#!/usr/bin/env python3
"""Quick test to verify clip optimization"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/c/Users/Samkelo/OneDrive - iqlab/Clients/HPCSeriesCore')
import hpcs
import time

# Test with NaN to ensure IEEE 754 semantics work
data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
result = hpcs.clip(data, 2.0, 4.0)
print(f"Clip with NaN: {result}")
print(f"NaN preserved: {np.isnan(result[2])}")

# Quick performance test
sizes = [1000, 10000, 100000, 1000000]
for n in sizes:
    data = np.random.randn(n)

    # Warm up
    for _ in range(3):
        _ = hpcs.clip(data.copy(), -2.0, 2.0)

    # Time it
    times = []
    for _ in range(5):
        d = data.copy()
        start = time.perf_counter()
        _ = hpcs.clip(d, -2.0, 2.0)
        times.append(time.perf_counter() - start)

    hpcs_time = np.median(times) * 1000

    # NumPy reference
    times = []
    for _ in range(5):
        d = data.copy()
        start = time.perf_counter()
        _ = np.clip(d, -2.0, 2.0)
        times.append(time.perf_counter() - start)

    numpy_time = np.median(times) * 1000
    speedup = numpy_time / hpcs_time

    print(f"n={n:7,}: HPCS={hpcs_time:6.3f}ms NumPy={numpy_time:6.3f}ms Speedup={speedup:.2f}x")
