# Logs Directory

This directory contains benchmark results, test logs, and performance reports.

## Directory Structure

```
logs/
├── benchmarks/          # Benchmark timing data (CSV format)
│   ├── cpu/            # CPU-only benchmark results
│   └── gpu/            # GPU-accelerated benchmark results
├── tests/              # Test execution logs
│   ├── cpu/            # CPU test logs
│   └── gpu/            # GPU test logs
└── reports/            # Performance analysis reports
    ├── scaling/        # Scaling analysis
    └── comparisons/    # CPU vs GPU comparisons
```

## File Naming Convention

### Benchmarks
- `benchmark_{kernel}_{mode}_{date}_{time}.csv`
- Example: `benchmark_v03_optimized_gpu_20251122_143025.csv`

### Tests
- `test_{suite}_{mode}_{date}_{time}.log`
- Example: `test_hpcs_gpu_kernels_gpu_20251122_143025.log`

### Reports
- `report_{type}_{date}.md`
- Example: `report_scaling_analysis_20251122.md`

## Benchmark CSV Format

```csv
timestamp,n,kernel,mode,elapsed_ms,throughput_MB_s,speedup
2025-11-22T14:30:25,1000000,median,gpu,2.8,2857.14,17.1
```

## Columns

- **timestamp**: ISO 8601 timestamp
- **n**: Dataset size (number of elements)
- **kernel**: Kernel name (median, mad, rolling_median, etc.)
- **mode**: cpu, gpu, or parallel
- **elapsed_ms**: Execution time in milliseconds
- **throughput_MB_s**: Data throughput (MB/s)
- **speedup**: Speedup vs CPU baseline

## Usage

### Run Benchmarks with Logging

```bash
# CPU benchmarks
docker compose run hpcs-dev bash -c "./docker-build.sh 2>&1 | tee logs/tests/cpu/build_$(date +%Y%m%d_%H%M%S).log"

# GPU benchmarks
docker compose run hpcs-gpu bash -c "./docker-build-gpu.sh 2>&1 | tee logs/tests/gpu/build_$(date +%Y%m%d_%H%M%S).log"
```

### Generate Performance Report

```bash
python3 scripts/generate_report.py --input logs/benchmarks/ --output logs/reports/
```

## Retention Policy

- **Benchmarks**: Keep last 30 days
- **Tests**: Keep last 7 days
- **Reports**: Keep all (archive monthly)

## .gitignore

Logs are excluded from git by default. To include specific reports:
```bash
git add -f logs/reports/report_*.md
```
