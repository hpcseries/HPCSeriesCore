#!/usr/bin/env python3
"""
HPCSeries Core v0.4 Benchmark Analysis

Analyzes benchmark results from run_v04_benchmarks.sh and generates:
1. Speedup comparison table
2. Performance summary report
3. CSV with speedup metrics

Usage:
    python3 analyze_v04_benchmarks.py <results_dir>
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(results_dir):
    """Load all benchmark CSV files from the results directory."""
    data = []

    for csv_file in sorted(Path(results_dir).glob("results_*threads.csv")):
        df = pd.read_csv(csv_file)
        data.append(df)

    if not data:
        raise ValueError(f"No results found in {results_dir}")

    return pd.concat(data, ignore_index=True)


def calculate_speedup(df):
    """Calculate speedup relative to single-threaded baseline."""
    # Get baseline (1 thread) timings
    baseline = df[df['threads'] == 1].copy()
    baseline = baseline.rename(columns={'elapsed_seconds': 'baseline_time'})
    baseline = baseline[['n', 'm', 'kernel', 'baseline_time']]

    # Merge baseline times
    df = df.merge(baseline, on=['n', 'm', 'kernel'])

    # Calculate speedup
    df['speedup'] = df['baseline_time'] / df['elapsed_seconds']
    df['efficiency'] = df['speedup'] / df['threads'] * 100  # parallel efficiency %

    return df


def generate_report(df, output_dir):
    """Generate markdown report with speedup analysis."""
    report_path = os.path.join(output_dir, "BENCHMARK_V04_RESULTS.md")

    with open(report_path, 'w') as f:
        f.write("# HPCSeries Core v0.4 Benchmark Results\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n\n")
        f.write("This report shows OpenMP parallel speedup for v0.4 batched/axis operations.\n\n")

        # Summary statistics
        max_threads = df['threads'].max()
        max_speedup_row = df[df['threads'] == max_threads].sort_values('speedup', ascending=False).iloc[0]
        avg_speedup = df[df['threads'] == max_threads]['speedup'].mean()

        f.write(f"- **Max threads tested:** {max_threads}\n")
        f.write(f"- **Average speedup ({max_threads} threads):** {avg_speedup:.2f}x\n")
        f.write(f"- **Best speedup:** {max_speedup_row['speedup']:.2f}x ({max_speedup_row['kernel']}, n={max_speedup_row['n']}, m={max_speedup_row['m']})\n\n")

        # Group by kernel and configuration
        f.write("## Speedup by Kernel\n\n")

        for kernel in sorted(df['kernel'].unique()):
            kernel_df = df[df['kernel'] == kernel]
            f.write(f"### {kernel}\n\n")

            # Create pivot table: rows = (n,m), columns = threads
            for (n, m), group in kernel_df.groupby(['n', 'm']):
                f.write(f"**Array size:** n={n}, m={m} ({n*m:,} elements)\n\n")
                f.write("| Threads | Time (s) | Speedup | Efficiency (%) |\n")
                f.write("|---------|----------|---------|----------------|\n")

                for _, row in group.sort_values('threads').iterrows():
                    f.write(f"| {row['threads']} | {row['elapsed_seconds']:.6f} | ")
                    f.write(f"{row['speedup']:.2f}x | {row['efficiency']:.1f}% |\n")

                f.write("\n")

        # Performance categories
        f.write("## Performance Summary\n\n")

        max_thread_df = df[df['threads'] == max_threads].copy()

        # Categorize by speedup
        excellent = max_thread_df[max_thread_df['speedup'] >= max_threads * 0.7]
        good = max_thread_df[(max_thread_df['speedup'] >= max_threads * 0.5) &
                            (max_thread_df['speedup'] < max_threads * 0.7)]
        moderate = max_thread_df[max_thread_df['speedup'] < max_threads * 0.5]

        f.write(f"### Excellent Scaling (≥{max_threads*0.7:.1f}x speedup)\n\n")
        if len(excellent) > 0:
            for kernel in excellent['kernel'].unique():
                f.write(f"- {kernel}\n")
        else:
            f.write("- None\n")
        f.write("\n")

        f.write(f"### Good Scaling ({max_threads*0.5:.1f}x - {max_threads*0.7:.1f}x speedup)\n\n")
        if len(good) > 0:
            for kernel in good['kernel'].unique():
                f.write(f"- {kernel}\n")
        else:
            f.write("- None\n")
        f.write("\n")

        f.write(f"### Moderate Scaling (<{max_threads*0.5:.1f}x speedup)\n\n")
        if len(moderate) > 0:
            for kernel in moderate['kernel'].unique():
                f.write(f"- {kernel}\n")
        else:
            f.write("- None\n")
        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("1. **Best for large arrays**: Kernels with excellent scaling benefit most from parallelization\n")
        f.write("2. **Threshold tuning**: Moderate-scaling kernels may need higher parallelization threshold\n")
        f.write("3. **Memory bandwidth**: Some operations may be memory-bound rather than compute-bound\n\n")

        f.write("## Data Files\n\n")
        f.write("- `speedup_analysis.csv`: Complete speedup data for all configurations\n")
        f.write("- `results_*threads.csv`: Raw benchmark results per thread count\n\n")

    print(f"✓ Report generated: {report_path}")
    return report_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_v04_benchmarks.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")
    df = load_results(results_dir)
    print(f"✓ Loaded {len(df)} benchmark results")

    print("Calculating speedup metrics...")
    df = calculate_speedup(df)

    # Save detailed results
    output_csv = os.path.join(results_dir, "speedup_analysis.csv")
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved speedup analysis: {output_csv}")

    # Generate report
    print("Generating report...")
    report_path = generate_report(df, results_dir)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    max_threads = df['threads'].max()
    summary = df[df['threads'] == max_threads].groupby('kernel')['speedup'].agg(['mean', 'min', 'max'])
    summary = summary.sort_values('mean', ascending=False)

    print(f"\nAverage speedup with {max_threads} threads:\n")
    print(f"{'Kernel':<35} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for kernel, row in summary.iterrows():
        print(f"{kernel:<35} {row['mean']:>7.2f}x {row['min']:>7.2f}x {row['max']:>7.2f}x")

    print(f"\n✓ Complete results in: {results_dir}/")


if __name__ == "__main__":
    main()
