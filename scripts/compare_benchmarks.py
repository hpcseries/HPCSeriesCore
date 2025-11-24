#!/usr/bin/env python3
"""
Benchmark Comparison Tool

Compares benchmark results from different runs, supporting:
- CPU vs GPU comparison
- Historical trend analysis
- Speedup calculations
- Performance regression detection
"""

import argparse
import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_csv_with_metadata(filepath: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Parse CSV file with metadata comments."""
    metadata = {}
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                # Parse metadata
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
            elif line and not line.startswith('#'):
                # Parse CSV data
                break

        # Reset to start of data section
        f.seek(0)
        lines = [l for l in f if not l.strip().startswith('#')]
        reader = csv.DictReader(lines)
        data = list(reader)

    return metadata, data


def aggregate_by_kernel(data: List[Dict[str, str]]) -> Dict[str, Dict[int, float]]:
    """Aggregate timing data by kernel and dataset size."""
    results = defaultdict(dict)

    for row in data:
        kernel = row.get('kernel', 'unknown')
        n = int(row.get('n', 0))
        elapsed = float(row.get('elapsed_seconds', 0))

        # Store timing data: kernel -> {n: elapsed_time}
        results[kernel][n] = elapsed

    return dict(results)


def calculate_speedup(cpu_data: Dict, gpu_data: Dict) -> Dict[str, Dict[int, float]]:
    """Calculate speedup for matching kernels and dataset sizes."""
    speedups = defaultdict(dict)

    for kernel in cpu_data.keys():
        if kernel in gpu_data:
            for n in cpu_data[kernel].keys():
                if n in gpu_data[kernel]:
                    cpu_time = cpu_data[kernel][n]
                    gpu_time = gpu_data[kernel][n]
                    if gpu_time > 0:
                        speedup = cpu_time / gpu_time
                        speedups[kernel][n] = speedup

    return dict(speedups)


def print_comparison_table(cpu_data: Dict, gpu_data: Dict, speedups: Dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("CPU vs GPU Benchmark Comparison")
    print("=" * 100)
    print()

    # Get all kernels
    all_kernels = sorted(set(cpu_data.keys()) | set(gpu_data.keys()))

    for kernel in all_kernels:
        print(f"\n{kernel.upper()}")
        print("-" * 100)
        print(f"{'Dataset Size':>15} | {'CPU Time (ms)':>15} | {'GPU Time (ms)':>15} | {'Speedup':>12} | {'Status':>20}")
        print("-" * 100)

        # Get all dataset sizes for this kernel
        sizes = set()
        if kernel in cpu_data:
            sizes.update(cpu_data[kernel].keys())
        if kernel in gpu_data:
            sizes.update(gpu_data[kernel].keys())

        for n in sorted(sizes):
            cpu_time = cpu_data.get(kernel, {}).get(n, None)
            gpu_time = gpu_data.get(kernel, {}).get(n, None)
            speedup = speedups.get(kernel, {}).get(n, None)

            # Format times in milliseconds
            cpu_time_str = f"{cpu_time * 1000:.2f}" if cpu_time is not None else "N/A"
            gpu_time_str = f"{gpu_time * 1000:.2f}" if gpu_time is not None else "N/A"
            speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"

            # Determine status based on targets
            status = ""
            if speedup is not None:
                if kernel == 'rolling_median':
                    target_min, target_max = 40, 60
                    if speedup >= target_min:
                        status = f"✅ Target: {target_min}-{target_max}x"
                    else:
                        status = f"⚠️  Below {target_min}x target"
                elif kernel in ['median', 'mad']:
                    target_min, target_max = 15, 20
                    if speedup >= target_min:
                        status = f"✅ Target: {target_min}-{target_max}x"
                    else:
                        status = f"⚠️  Below {target_min}x target"
                elif kernel == 'prefix_sum':
                    target_min, target_max = 15, 25
                    if speedup >= target_min:
                        status = f"✅ Target: {target_min}-{target_max}x"
                    else:
                        status = f"⚠️  Below {target_min}x target"
                else:
                    if speedup >= 10:
                        status = "✅ Good (10x+)"
                    elif speedup >= 5:
                        status = "⚠️  Moderate (5-10x)"
                    else:
                        status = "❌ Low (<5x)"

            print(f"{n:>15,} | {cpu_time_str:>15} | {gpu_time_str:>15} | {speedup_str:>12} | {status:>20}")

        print()


def print_summary_statistics(speedups: Dict):
    """Print summary statistics for speedups."""
    print("\n" + "=" * 100)
    print("Summary Statistics")
    print("=" * 100)
    print()
    print(f"{'Kernel':>20} | {'Min Speedup':>15} | {'Max Speedup':>15} | {'Avg Speedup':>15} | {'Status':>20}")
    print("-" * 100)

    for kernel in sorted(speedups.keys()):
        values = list(speedups[kernel].values())
        if values:
            min_speedup = min(values)
            max_speedup = max(values)
            avg_speedup = sum(values) / len(values)

            # Determine overall status
            if kernel == 'rolling_median':
                target = 40
                status = "✅ Meeting target" if avg_speedup >= target else f"⚠️  Below {target}x"
            elif kernel in ['median', 'mad']:
                target = 15
                status = "✅ Meeting target" if avg_speedup >= target else f"⚠️  Below {target}x"
            elif kernel == 'prefix_sum':
                target = 15
                status = "✅ Meeting target" if avg_speedup >= target else f"⚠️  Below {target}x"
            else:
                status = "✅ Good" if avg_speedup >= 10 else "⚠️  Below 10x"

            print(f"{kernel:>20} | {min_speedup:>14.2f}x | {max_speedup:>14.2f}x | {avg_speedup:>14.2f}x | {status:>20}")

    print()


def compare_same_mode(files: List[str], mode: str):
    """Compare multiple runs of the same mode (e.g., comparing 2 GPU runs)."""
    print("\n" + "=" * 100)
    print(f"Comparing {len(files)} {mode.upper()} Benchmark Runs")
    print("=" * 100)
    print()

    all_data = []
    for idx, filepath in enumerate(files, 1):
        metadata, data = parse_csv_with_metadata(filepath)
        aggregated = aggregate_by_kernel(data)

        filename = os.path.basename(filepath)
        timestamp = metadata.get('Timestamp', 'Unknown')

        print(f"Run {idx}: {filename}")
        print(f"  Timestamp: {timestamp}")
        if 'GPU' in metadata:
            print(f"  GPU: {metadata['GPU']}")

        all_data.append({
            'index': idx,
            'filename': filename,
            'timestamp': timestamp,
            'data': aggregated
        })

    print()

    # Compare timing differences between runs
    if len(all_data) >= 2:
        print("\nTiming Comparison (Run 1 vs Latest Run):")
        print("-" * 100)

        run1_data = all_data[0]['data']
        run_latest_data = all_data[-1]['data']

        for kernel in sorted(set(run1_data.keys()) | set(run_latest_data.keys())):
            print(f"\n{kernel.upper()}")
            print(f"{'Dataset Size':>15} | {'Run 1 (ms)':>15} | {'Latest (ms)':>15} | {'Difference':>15} | {'Change':>10}")
            print("-" * 100)

            sizes = set()
            if kernel in run1_data:
                sizes.update(run1_data[kernel].keys())
            if kernel in run_latest_data:
                sizes.update(run_latest_data[kernel].keys())

            for n in sorted(sizes):
                time1 = run1_data.get(kernel, {}).get(n, None)
                time_latest = run_latest_data.get(kernel, {}).get(n, None)

                time1_str = f"{time1 * 1000:.2f}" if time1 is not None else "N/A"
                time_latest_str = f"{time_latest * 1000:.2f}" if time_latest is not None else "N/A"

                if time1 is not None and time_latest is not None:
                    diff = time_latest - time1
                    diff_pct = ((time_latest - time1) / time1) * 100
                    diff_str = f"{diff * 1000:+.2f} ms"
                    change_str = f"{diff_pct:+.1f}%"

                    if diff_pct < -5:
                        change_str += " ✅"
                    elif diff_pct > 5:
                        change_str += " ⚠️"
                else:
                    diff_str = "N/A"
                    change_str = "N/A"

                print(f"{n:>15,} | {time1_str:>15} | {time_latest_str:>15} | {diff_str:>15} | {change_str:>10}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare benchmark results across runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare latest 2 GPU runs
  python3 scripts/compare_benchmarks.py --mode gpu --latest 2

  # Compare CPU vs GPU (uses latest file from each mode)
  python3 scripts/compare_benchmarks.py --compare cpu gpu

  # Compare latest 3 CPU runs
  python3 scripts/compare_benchmarks.py --mode cpu --latest 3
        """
    )
    parser.add_argument('--mode', type=str, choices=['cpu', 'gpu'],
                        help='Mode to analyze (cpu or gpu)')
    parser.add_argument('--latest', type=int, default=2,
                        help='Number of latest benchmark files to compare (default: 2)')
    parser.add_argument('--compare', nargs=2, metavar=('MODE1', 'MODE2'),
                        help='Compare two modes (e.g., cpu gpu)')
    parser.add_argument('--logs-dir', type=str, default='logs/benchmarks',
                        help='Path to benchmark logs directory')
    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    logs_dir = project_root / args.logs_dir

    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        sys.exit(1)

    # Mode 1: Compare CPU vs GPU (different modes)
    if args.compare:
        mode1, mode2 = args.compare

        files1 = sorted(glob.glob(str(logs_dir / mode1 / '*.csv')), reverse=True)
        files2 = sorted(glob.glob(str(logs_dir / mode2 / '*.csv')), reverse=True)

        if not files1:
            print(f"Error: No {mode1} benchmark files found")
            sys.exit(1)
        if not files2:
            print(f"Error: No {mode2} benchmark files found")
            sys.exit(1)

        # Use latest file from each mode
        print(f"Comparing {mode1.upper()} vs {mode2.upper()}")
        print(f"{mode1.upper()}: {os.path.basename(files1[0])}")
        print(f"{mode2.upper()}: {os.path.basename(files2[0])}")

        # Parse data
        _, data1 = parse_csv_with_metadata(files1[0])
        _, data2 = parse_csv_with_metadata(files2[0])

        # Aggregate data
        agg1 = aggregate_by_kernel(data1)
        agg2 = aggregate_by_kernel(data2)

        # Calculate speedups (assume mode1 is baseline)
        if mode1 == 'cpu' and mode2 == 'gpu':
            speedups = calculate_speedup(agg1, agg2)
            print_comparison_table(agg1, agg2, speedups)
            print_summary_statistics(speedups)
        elif mode1 == 'gpu' and mode2 == 'cpu':
            speedups = calculate_speedup(agg2, agg1)
            print_comparison_table(agg2, agg1, speedups)
            print_summary_statistics(speedups)
        else:
            print(f"\nWarning: Speedup calculation typically compares CPU (baseline) to GPU")
            print(f"Showing timing data only:\n")
            print_comparison_table(agg1, agg2, {})

    # Mode 2: Compare multiple runs of same mode
    elif args.mode:
        files = sorted(glob.glob(str(logs_dir / args.mode / '*.csv')), reverse=True)

        if not files:
            print(f"Error: No {args.mode} benchmark files found")
            sys.exit(1)

        # Limit to latest N files
        files = files[:args.latest]

        if len(files) < 2:
            print(f"Error: Need at least 2 benchmark files to compare. Found: {len(files)}")
            sys.exit(1)

        compare_same_mode(files, args.mode)

    else:
        # Default: Compare latest CPU vs GPU
        print("No mode specified. Comparing latest CPU vs GPU runs...\n")

        cpu_files = sorted(glob.glob(str(logs_dir / 'cpu' / '*.csv')), reverse=True)
        gpu_files = sorted(glob.glob(str(logs_dir / 'gpu' / '*.csv')), reverse=True)

        if not cpu_files or not gpu_files:
            print("Error: Need both CPU and GPU benchmark files for default comparison")
            print(f"CPU files found: {len(cpu_files)}")
            print(f"GPU files found: {len(gpu_files)}")
            print("\nTip: Run benchmarks first:")
            print("  ./scripts/run_benchmarks.sh cpu")
            print("  ./scripts/run_benchmarks.sh gpu")
            sys.exit(1)

        print(f"CPU: {os.path.basename(cpu_files[0])}")
        print(f"GPU: {os.path.basename(gpu_files[0])}")

        # Parse data
        _, cpu_data = parse_csv_with_metadata(cpu_files[0])
        _, gpu_data = parse_csv_with_metadata(gpu_files[0])

        # Aggregate and compare
        cpu_agg = aggregate_by_kernel(cpu_data)
        gpu_agg = aggregate_by_kernel(gpu_data)

        speedups = calculate_speedup(cpu_agg, gpu_agg)
        print_comparison_table(cpu_agg, gpu_agg, speedups)
        print_summary_statistics(speedups)

    print("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()
