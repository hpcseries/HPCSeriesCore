#!/usr/bin/env python3
"""
Compare before/after performance tuning results
"""

import csv
from pathlib import Path

def load_results(csv_file):
    """Load benchmark results from CSV."""
    data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['n']), int(row['m']), row['kernel'])
            data[key] = float(row['elapsed_seconds'])
    return data

def calculate_speedup(baseline, parallel):
    """Calculate speedup."""
    speedups = {}
    for key in baseline:
        if key in parallel:
            speedups[key] = baseline[key] / parallel[key]
    return speedups

# Load old (untuned) results
old_1t = load_results('benchmark_results_v04/results_1threads.csv')
old_8t = load_results('benchmark_results_v04/results_8threads.csv')
old_speedup = calculate_speedup(old_1t, old_8t)

# Load new (tuned) results
new_1t = load_results('build/benchmark_results_v04_tuned/results_1threads.csv')
new_8t = load_results('build/benchmark_results_v04_tuned/results_8threads.csv')
new_speedup = calculate_speedup(new_1t, new_8t)

print("=" * 80)
print("Performance Tuning Results Comparison")
print("=" * 80)
print()

# Find largest array size
max_n = max(k[0] for k in old_speedup.keys())
max_m = max(k[1] for k in old_speedup.keys())

# Get all kernels
kernels = sorted(set(k[2] for k in old_speedup.keys()))

print(f"Comparing performance on largest arrays ({max_n}x{max_m}):")
print()

improvements = []
regressions = []

for kernel in kernels:
    key = (max_n, max_m, kernel)
    if key in old_speedup and key in new_speedup:
        old_sp = old_speedup[key]
        new_sp = new_speedup[key]
        delta = new_sp - old_sp
        pct_change = (delta / old_sp) * 100 if old_sp > 0 else 0

        status = ""
        if abs(delta) < 0.05:
            status = "="
        elif delta > 0:
            status = "↑"
            improvements.append((kernel, old_sp, new_sp, delta, pct_change))
        else:
            status = "↓"
            regressions.append((kernel, old_sp, new_sp, delta, pct_change))

        print(f"{status} {kernel:35s}  "
              f"Before: {old_sp:5.2f}x  →  After: {new_sp:5.2f}x  "
              f"({delta:+.2f}x, {pct_change:+6.1f}%)")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print()

if improvements:
    print(f"✓ Improvements ({len(improvements)} kernels):")
    for kernel, old, new, delta, pct in sorted(improvements, key=lambda x: -x[3]):
        print(f"  {kernel:35s}: {old:5.2f}x → {new:5.2f}x  ({pct:+6.1f}%)")
    print()

if regressions:
    print(f"⚠ Regressions ({len(regressions)} kernels):")
    for kernel, old, new, delta, pct in sorted(regressions, key=lambda x: x[3]):
        print(f"  {kernel:35s}: {old:5.2f}x → {new:5.2f}x  ({pct:+6.1f}%)")
    print()

# Calculate average speedup before and after
avg_old = sum(old_speedup[k] for k in old_speedup if k[0] == max_n and k[1] == max_m) / \
          len([k for k in old_speedup if k[0] == max_n and k[1] == max_m])
avg_new = sum(new_speedup[k] for k in new_speedup if k[0] == max_n and k[1] == max_m) / \
          len([k for k in new_speedup if k[0] == max_n and k[1] == max_m])

print(f"Average speedup (8 threads, {max_n}x{max_m} arrays):")
print(f"  Before tuning: {avg_old:.2f}x")
print(f"  After tuning:  {avg_new:.2f}x")
print(f"  Improvement:   {avg_new - avg_old:+.2f}x ({((avg_new - avg_old) / avg_old) * 100:+.1f}%)")
print()
