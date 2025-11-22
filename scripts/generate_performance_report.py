#!/usr/bin/env python3
"""
Performance Report Generator

Analyzes benchmark CSV files from logs/benchmarks/ and generates
comprehensive performance reports including:
- Scaling analysis
- CPU vs GPU comparison
- Speedup metrics
- Performance trends
"""

import argparse
import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


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


def analyze_scaling(data: List[Dict[str, str]], kernel: str) -> Dict[str, float]:
    """Analyze scaling behavior for a specific kernel."""
    kernel_data = [row for row in data if row.get('kernel') == kernel]

    if not kernel_data:
        return {}

    # Sort by dataset size
    kernel_data.sort(key=lambda x: int(x['n']))

    results = {
        'kernel': kernel,
        'min_n': int(kernel_data[0]['n']),
        'max_n': int(kernel_data[-1]['n']),
        'min_time': float(kernel_data[0]['elapsed_seconds']) * 1000,  # ms
        'max_time': float(kernel_data[-1]['elapsed_seconds']) * 1000,  # ms
        'scaling_factor': float(kernel_data[-1]['elapsed_seconds']) / float(kernel_data[0]['elapsed_seconds']),
        'data_points': len(kernel_data)
    }

    # Calculate average speedup if available
    speedups = [float(row.get('speedup', '1.0')) for row in kernel_data if 'speedup' in row]
    if speedups:
        results['avg_speedup'] = sum(speedups) / len(speedups)
        results['max_speedup'] = max(speedups)
        results['min_speedup'] = min(speedups)

    return results


def generate_markdown_report(cpu_results: Dict, gpu_results: Dict, output_file: str):
    """Generate comprehensive markdown performance report."""
    with open(output_file, 'w') as f:
        # Header
        f.write("# HPCSeries Performance Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # CPU Results
        if cpu_results:
            f.write("## CPU Performance\n\n")
            if 'metadata' in cpu_results:
                meta = cpu_results['metadata']
                f.write(f"- **Hostname**: {meta.get('Hostname', 'Unknown')}\n")
                f.write(f"- **Timestamp**: {meta.get('Timestamp', 'Unknown')}\n")
                f.write(f"- **Mode**: {meta.get('Mode', 'cpu')}\n\n")

            if 'kernels' in cpu_results:
                f.write("### Kernel Performance\n\n")
                f.write("| Kernel | Dataset Size Range | Min Time (ms) | Max Time (ms) | Scaling Factor |\n")
                f.write("|--------|-------------------|---------------|---------------|----------------|\n")
                for kernel_name, metrics in cpu_results['kernels'].items():
                    f.write(f"| {kernel_name} | {metrics['min_n']:,} - {metrics['max_n']:,} | "
                            f"{metrics['min_time']:.2f} | {metrics['max_time']:.2f} | "
                            f"{metrics['scaling_factor']:.2f}x |\n")
                f.write("\n")

        # GPU Results
        if gpu_results:
            f.write("## GPU Performance\n\n")
            if 'metadata' in gpu_results:
                meta = gpu_results['metadata']
                f.write(f"- **Hostname**: {meta.get('Hostname', 'Unknown')}\n")
                f.write(f"- **GPU**: {meta.get('GPU', 'Unknown')}\n")
                f.write(f"- **Timestamp**: {meta.get('Timestamp', 'Unknown')}\n\n")

            if 'kernels' in gpu_results:
                f.write("### Kernel Performance\n\n")
                f.write("| Kernel | Dataset Size Range | Min Time (ms) | Max Time (ms) | Avg Speedup | Max Speedup |\n")
                f.write("|--------|-------------------|---------------|---------------|-------------|-------------|\n")
                for kernel_name, metrics in gpu_results['kernels'].items():
                    avg_speedup = metrics.get('avg_speedup', 1.0)
                    max_speedup = metrics.get('max_speedup', 1.0)
                    f.write(f"| {kernel_name} | {metrics['min_n']:,} - {metrics['max_n']:,} | "
                            f"{metrics['min_time']:.2f} | {metrics['max_time']:.2f} | "
                            f"{avg_speedup:.1f}x | {max_speedup:.1f}x |\n")
                f.write("\n")

        # CPU vs GPU Comparison
        if cpu_results and gpu_results and 'kernels' in cpu_results and 'kernels' in gpu_results:
            f.write("## CPU vs GPU Comparison\n\n")
            f.write("### Speedup Analysis\n\n")
            f.write("| Kernel | GPU Speedup (Avg) | GPU Speedup (Max) | Status |\n")
            f.write("|--------|-------------------|-------------------|--------|\n")

            for kernel_name in gpu_results['kernels'].keys():
                if kernel_name in gpu_results['kernels']:
                    gpu_metrics = gpu_results['kernels'][kernel_name]
                    avg_speedup = gpu_metrics.get('avg_speedup', 1.0)
                    max_speedup = gpu_metrics.get('max_speedup', 1.0)

                    # Determine status based on targets
                    if kernel_name == 'rolling_median':
                        target = "40-60x"
                        status = "✅" if avg_speedup >= 40 else "⚠️"
                    elif kernel_name in ['median', 'mad']:
                        target = "15-20x"
                        status = "✅" if avg_speedup >= 15 else "⚠️"
                    elif kernel_name == 'prefix_sum':
                        target = "15-25x"
                        status = "✅" if avg_speedup >= 15 else "⚠️"
                    else:
                        target = "10x+"
                        status = "✅" if avg_speedup >= 10 else "⚠️"

                    f.write(f"| {kernel_name} | {avg_speedup:.1f}x | {max_speedup:.1f}x | {status} ({target} target) |\n")

            f.write("\n")

        # Performance Targets
        f.write("## Performance Targets (Phase 3B)\n\n")
        f.write("| Operation | Target Speedup | Status |\n")
        f.write("|-----------|---------------|--------|\n")
        f.write("| median | 15-20x | ⏳ Verify with GPU tests |\n")
        f.write("| MAD | 15-20x | ⏳ Verify with GPU tests |\n")
        f.write("| rolling_median | 40-60x | ⏳ Verify with GPU tests |\n")
        f.write("| prefix_sum | 15-25x | ⏳ Verify with GPU tests |\n")
        f.write("| reduce_sum | 10x | ⏳ Verify with GPU tests |\n")
        f.write("\n")

        # Next Steps
        f.write("## Next Steps\n\n")
        f.write("1. Analyze scaling behavior for production workloads\n")
        f.write("2. Profile GPU kernels with nsys for optimization opportunities\n")
        f.write("3. Test with real-world data sizes and patterns\n")
        f.write("4. Implement Phase 4B (async transfers, memory pooling)\n")
        f.write("\n")

    print(f"✅ Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate performance report from benchmark logs')
    parser.add_argument('--date', type=str, help='Date filter (YYYYMMDD)', default=None)
    parser.add_argument('--logs-dir', type=str, default='logs/benchmarks',
                        help='Path to benchmark logs directory')
    parser.add_argument('--output-dir', type=str, default='logs/reports',
                        help='Path to output reports directory')
    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    logs_dir = project_root / args.logs_dir
    output_dir = project_root / args.output_dir

    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find latest benchmark files
    cpu_files = sorted(glob.glob(str(logs_dir / 'cpu' / '*.csv')), reverse=True)
    gpu_files = sorted(glob.glob(str(logs_dir / 'gpu' / '*.csv')), reverse=True)

    # Filter by date if specified
    if args.date:
        cpu_files = [f for f in cpu_files if args.date in f]
        gpu_files = [f for f in gpu_files if args.date in f]

    print(f"Found {len(cpu_files)} CPU benchmark files")
    print(f"Found {len(gpu_files)} GPU benchmark files")

    # Analyze CPU results
    cpu_results = {}
    if cpu_files:
        latest_cpu = cpu_files[0]
        print(f"Analyzing CPU: {latest_cpu}")
        metadata, data = parse_csv_with_metadata(latest_cpu)
        cpu_results['metadata'] = metadata
        cpu_results['kernels'] = {}

        # Get unique kernels
        kernels = set(row['kernel'] for row in data if 'kernel' in row)
        for kernel in kernels:
            cpu_results['kernels'][kernel] = analyze_scaling(data, kernel)

    # Analyze GPU results
    gpu_results = {}
    if gpu_files:
        latest_gpu = gpu_files[0]
        print(f"Analyzing GPU: {latest_gpu}")
        metadata, data = parse_csv_with_metadata(latest_gpu)
        gpu_results['metadata'] = metadata
        gpu_results['kernels'] = {}

        # Get unique kernels
        kernels = set(row['kernel'] for row in data if 'kernel' in row)
        for kernel in kernels:
            gpu_results['kernels'][kernel] = analyze_scaling(data, kernel)

    # Generate report
    date_str = args.date or datetime.now().strftime('%Y%m%d')
    output_file = output_dir / f'performance_report_{date_str}.md'

    generate_markdown_report(cpu_results, gpu_results, str(output_file))

    print(f"\n✅ Performance report generated successfully!")
    print(f"📄 Report: {output_file}")


if __name__ == '__main__':
    main()
