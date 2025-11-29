#!/usr/bin/env python3
"""
Simple benchmark analysis without pandas dependency
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

def load_results(results_dir):
    """Load all benchmark CSV files."""
    data = []

    for csv_file in sorted(Path(results_dir).glob("results_*threads.csv")):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['n'] = int(row['n'])
                row['m'] = int(row['m'])
                row['threads'] = int(row['threads'])
                row['elapsed_seconds'] = float(row['elapsed_seconds'])
                data.append(row)

    return data

def calculate_speedup(data):
    """Calculate speedup relative to single-threaded baseline."""
    # Build baseline lookup
    baseline = {}
    for row in data:
        if row['threads'] == 1:
            key = (row['n'], row['m'], row['kernel'])
            baseline[key] = row['elapsed_seconds']

    # Add speedup to each row
    for row in data:
        key = (row['n'], row['m'], row['kernel'])
        if key in baseline:
            row['speedup'] = baseline[key] / row['elapsed_seconds']
            row['efficiency'] = (row['speedup'] / row['threads']) * 100
        else:
            row['speedup'] = 0
            row['efficiency'] = 0

    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_v04_simple.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    print(f"Loading results from: {results_dir}")
    data = load_results(results_dir)
    print(f"✓ Loaded {len(data)} benchmark results\n")

    data = calculate_speedup(data)

    # Find max threads
    max_threads = max(row['threads'] for row in data)

    # Calculate summary statistics
    max_speedup_row = max([r for r in data if r['threads'] == max_threads],
                          key=lambda x: x['speedup'])
    avg_speedup = sum(r['speedup'] for r in data if r['threads'] == max_threads) / \
                  len([r for r in data if r['threads'] == max_threads])

    print("="*70)
    print("HPCSeries Core v0.4 Benchmark Results")
    print("="*70)
    print()
    print(f"Max threads tested: {max_threads}")
    print(f"Average speedup ({max_threads} threads): {avg_speedup:.2f}x")
    print(f"Best speedup: {max_speedup_row['speedup']:.2f}x ({max_speedup_row['kernel']}, "
          f"n={max_speedup_row['n']}, m={max_speedup_row['m']})")
    print()

    # Group by kernel and show speedup for max threads
    print("="*70)
    print("Speedup Summary (for largest arrays)")
    print("="*70)
    print()

    kernels = sorted(set(row['kernel'] for row in data))
    max_n = max(row['n'] for row in data)
    max_m = max(row['m'] for row in data)

    for kernel in kernels:
        kernel_data = [r for r in data
                       if r['kernel'] == kernel and r['n'] == max_n and r['m'] == max_m]

        if kernel_data:
            print(f"{kernel}:")
            for row in sorted(kernel_data, key=lambda x: x['threads']):
                print(f"  {row['threads']:2d} threads: {row['elapsed_seconds']:8.6f}s  "
                      f"→ {row['speedup']:5.2f}x speedup  "
                      f"({row['efficiency']:5.1f}% efficiency)")
            print()

    # Performance categories
    print("="*70)
    print("Performance Categories")
    print("="*70)
    print()

    max_thread_data = [r for r in data if r['threads'] == max_threads]

    excellent = [r for r in max_thread_data if r['speedup'] >= max_threads * 0.7]
    good = [r for r in max_thread_data
            if max_threads * 0.5 <= r['speedup'] < max_threads * 0.7]
    moderate = [r for r in max_thread_data if r['speedup'] < max_threads * 0.5]

    print(f"Excellent Scaling (≥{max_threads*0.7:.1f}x speedup):")
    if excellent:
        for kernel in sorted(set(r['kernel'] for r in excellent)):
            avg = sum(r['speedup'] for r in excellent if r['kernel'] == kernel) / \
                  len([r for r in excellent if r['kernel'] == kernel])
            print(f"  - {kernel}: {avg:.2f}x")
    else:
        print("  - None")
    print()

    print(f"Good Scaling ({max_threads*0.5:.1f}x - {max_threads*0.7:.1f}x speedup):")
    if good:
        for kernel in sorted(set(r['kernel'] for r in good)):
            avg = sum(r['speedup'] for r in good if r['kernel'] == kernel) / \
                  len([r for r in good if r['kernel'] == kernel])
            print(f"  - {kernel}: {avg:.2f}x")
    else:
        print("  - None")
    print()

    print(f"Moderate Scaling (<{max_threads*0.5:.1f}x speedup):")
    if moderate:
        for kernel in sorted(set(r['kernel'] for r in moderate)):
            avg = sum(r['speedup'] for r in moderate if r['kernel'] == kernel) / \
                  len([r for r in moderate if r['kernel'] == kernel])
            print(f"  - {kernel}: {avg:.2f}x")
    else:
        print("  - None")
    print()

    print("="*70)
    print(f"✓ Benchmark analysis complete!")
    print(f"✓ Results saved in: {results_dir}/")
    print("="*70)

if __name__ == "__main__":
    main()
