"""
HPCSeries Core v0.7 - Command Line Interface
=============================================

Provides CLI commands for calibration, benchmarking, and system introspection.

Commands:
  hpcs calibrate    - Run auto-tuning calibration
  hpcs cpuinfo      - Display CPU topology and SIMD capabilities
  hpcs bench        - Run performance benchmarks
  hpcs version      - Show library version
  hpcs config       - Show configuration file location
  hpcs test         - Run correctness tests
"""

import argparse
import sys
import time
import numpy as np

try:
    import hpcs
except ImportError:
    print("ERROR: Could not import hpcs module. Please install with: pip install -e .")
    sys.exit(1)

__version__ = hpcs.__version__


def cmd_version(args):
    """Display version information."""
    print(f"HPCSeries Core v{__version__}")
    print("High-Performance Computing Series - Optimized Statistical Kernels")
    print()
    print("Features:")
    print("  • SIMD vectorization (AVX-512, AVX2, AVX, SSE2, NEON)")
    print("  • OpenMP parallelization with NUMA awareness")
    print("  • Adaptive auto-tuning (v0.5)")
    print("  • Fortran/C/Python unified API")
    return 0


def cmd_cpuinfo(args):
    """Display CPU topology and SIMD capabilities."""
    print("=== CPU Information ===\n")

    try:
        cpu_info = hpcs.get_cpu_info()
        simd_info = hpcs.simd_info()

        print(f"CPU Vendor:          {cpu_info['cpu_vendor']}")
        print(f"Physical Cores:      {cpu_info['physical_cores']}")
        print(f"Logical Cores:       {cpu_info['logical_cores']}")
        print(f"Optimal Threads:     {cpu_info['optimal_threads']}")
        print()

        print("Cache Hierarchy:")
        print(f"  L1:  {cpu_info['l1_cache_kb']:>6} KB")
        print(f"  L2:  {cpu_info['l2_cache_kb']:>6} KB")
        print(f"  L3:  {cpu_info['l3_cache_kb']:>6} KB")
        print()

        print("NUMA Topology:")
        print(f"  Nodes:               {cpu_info['numa_nodes']}")
        print(f"  Cores per node:      {cpu_info['cores_per_numa']}")
        print()

        print("SIMD Capabilities:")
        print(f"  Active ISA:          {simd_info['isa']}")
        print(f"  Vector width:        {simd_info['width_bytes'] * 8}-bit ({simd_info['width_doubles']} doubles)")
        print(f"  SSE2:                {'✓' if cpu_info['has_sse2'] else '✗'}")
        print(f"  AVX:                 {'✓' if cpu_info['has_avx'] else '✗'}")
        print(f"  AVX2:                {'✓' if cpu_info['has_avx2'] else '✗'}")
        print(f"  AVX-512:             {'✓' if cpu_info['has_avx512'] else '✗'}")
        print(f"  NEON:                {'✓' if cpu_info['has_neon'] else '✗'}")
        print(f"  FMA3:                {'✓' if cpu_info['has_fma3'] else '✗'}")

        return 0
    except Exception as e:
        print(f"ERROR: Failed to get CPU info: {e}")
        return 1


def cmd_bench(args):
    """Run performance benchmarks."""
    print("=== HPCSeries Performance Benchmark ===\n")

    # Test parameters
    n = args.size if hasattr(args, 'size') else 10_000_000
    iterations = args.iterations if hasattr(args, 'iterations') else 10

    print(f"Array size:     {n:,} elements ({n * 8 / 1e6:.2f} MB)")
    print(f"Iterations:     {iterations}")
    print()

    # Generate test data
    print("Generating test data...")
    data = np.random.randn(n)

    # Benchmark functions
    benchmarks = [
        ("sum", lambda: hpcs.sum(data)),
        ("mean", lambda: hpcs.mean(data)),
        ("std", lambda: hpcs.std(data)),
        ("median", lambda: hpcs.median(data)),
        ("rolling_mean(50)", lambda: hpcs.rolling_mean(data, 50)),
    ]

    print("Running benchmarks...\n")
    print(f"{'Function':<20} {'Time (ms)':<12} {'Throughput':<15}")
    print("-" * 50)

    for name, func in benchmarks:
        # Warmup
        _ = func()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            result = func()
        elapsed = (time.perf_counter() - start) / iterations

        throughput = n / elapsed / 1e6  # M elements/sec
        print(f"{name:<20} {elapsed*1000:>8.3f}     {throughput:>8.2f} M elem/s")

    print()
    return 0


def cmd_config(args):
    """Show configuration file location."""
    print("Configuration file location:")
    print("  $HOME/.hpcs/config.json")
    print()
    print("Use 'hpcs calibrate' to generate/update configuration.")
    return 0


def cmd_calibrate(args):
    """Run auto-tuning calibration."""
    import os

    print("=== HPCSeries Auto-Tuning Calibration ===\n")
    print("This will benchmark optimal parallelization thresholds for your system.")
    print()

    # Determine calibration mode
    quick = args.quick if hasattr(args, 'quick') else False
    mode_str = "quick" if quick else "full"
    time_estimate = "5-10 seconds" if quick else "30-60 seconds"

    print(f"Mode: {mode_str} calibration")
    print(f"Estimated time: {time_estimate}")
    print()
    print("Running benchmarks...")

    try:
        # Run calibration
        start = time.perf_counter()
        hpcs.calibrate(quick=quick)
        elapsed = time.perf_counter() - start

        print(f"✓ Calibration completed in {elapsed:.1f}s\n")

        # Save configuration
        config_dir = os.path.expanduser("~/.hpcs")
        config_path = os.path.join(config_dir, "config.json")

        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)

        hpcs.save_calibration_config(config_path)
        print(f"✓ Configuration saved to: {config_path}")
        print()
        print("Calibration complete! Optimal thresholds have been determined.")
        print("These will be used automatically in future sessions.")

        return 0

    except Exception as e:
        print(f"✗ Calibration failed: {e}")
        return 1


def cmd_test(args):
    """Run correctness tests."""
    print("=== HPCSeries Correctness Tests ===\n")

    tests_passed = 0
    tests_failed = 0

    # Test 1: Basic reductions
    print("[1/5] Testing basic reductions...")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        assert abs(hpcs.sum(x) - 15.0) < 1e-10
        assert abs(hpcs.mean(x) - 3.0) < 1e-10
        assert abs(hpcs.min(x) - 1.0) < 1e-10
        assert abs(hpcs.max(x) - 5.0) < 1e-10
        print("  ✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        tests_failed += 1

    # Test 2: Robust statistics
    print("[2/5] Testing robust statistics...")
    try:
        assert abs(hpcs.median(x) - 3.0) < 1e-10
        print("  ✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        tests_failed += 1

    # Test 3: Rolling operations
    print("[3/5] Testing rolling operations...")
    try:
        result = hpcs.rolling_mean(x, 3)
        assert len(result) == len(x)
        print("  ✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        tests_failed += 1

    # Test 4: Large array
    print("[4/5] Testing large arrays...")
    try:
        large = np.random.randn(1_000_000)
        _ = hpcs.sum(large)
        _ = hpcs.rolling_mean(large, 100)
        print("  ✓ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        tests_failed += 1

    # Test 5: Type conversions
    print("[5/5] Testing type conversions...")
    try:
        # List input
        _ = hpcs.sum([1, 2, 3, 4, 5])
        # Integer array
        _ = hpcs.mean(np.array([1, 2, 3], dtype=np.int32))
        print("  ✓ PASS")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        tests_failed += 1

    # Summary
    print()
    print(f"Tests passed: {tests_passed}/5")
    print(f"Tests failed: {tests_failed}/5")

    return 0 if tests_failed == 0 else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='hpcs',
        description='HPCSeries Core - High-Performance Statistical Computing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # version command
    parser_version = subparsers.add_parser('version', help='Show version information')
    parser_version.set_defaults(func=cmd_version)

    # cpuinfo command
    parser_cpuinfo = subparsers.add_parser('cpuinfo', help='Display CPU information')
    parser_cpuinfo.set_defaults(func=cmd_cpuinfo)

    # bench command
    parser_bench = subparsers.add_parser('bench', help='Run performance benchmarks')
    parser_bench.add_argument('--size', type=int, default=10_000_000, help='Array size')
    parser_bench.add_argument('--iterations', type=int, default=10, help='Iterations')
    parser_bench.set_defaults(func=cmd_bench)

    # config command
    parser_config = subparsers.add_parser('config', help='Show configuration location')
    parser_config.set_defaults(func=cmd_config)

    # calibrate command
    parser_calibrate = subparsers.add_parser('calibrate', help='Run auto-tuning calibration')
    parser_calibrate.add_argument('--quick', action='store_true', help='Quick calibration (faster, less accurate)')
    parser_calibrate.set_defaults(func=cmd_calibrate)

    # test command
    parser_test = subparsers.add_parser('test', help='Run correctness tests')
    parser_test.set_defaults(func=cmd_test)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        return args.func(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
