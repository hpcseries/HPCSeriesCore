# HPCSeries Performance Report

**Generated**: 2025-11-22 04:52:08

## CPU Performance

- **Hostname**: c8f6aa9cfed2
- **Timestamp**: 2025-11-22T02:47:56+00:00
- **Mode**: cpu

### Kernel Performance

| Kernel | Dataset Size Range | Min Time (ms) | Max Time (ms) | Scaling Factor |
|--------|-------------------|---------------|---------------|----------------|
| robust_zscore | 100,000 - 10,000,000 | 6.90 | 703.68 | 101.92x |
| rolling_mad | 100,000 - 1,000,000 | 433.13 | 1838.83 | 4.25x |
| mad | 100,000 - 10,000,000 | 6.67 | 355.17 | 53.22x |
| rolling_median | 100,000 - 1,000,000 | -571.03 | 526.52 | -0.92x |
| quantile | 100,000 - 10,000,000 | 3.02 | 194.13 | 64.28x |
| median | 100,000 - 10,000,000 | 3.00 | 170.00 | 56.65x |

## Performance Targets (Phase 3B)

| Operation | Target Speedup | Status |
|-----------|---------------|--------|
| median | 15-20x | ⏳ Verify with GPU tests |
| MAD | 15-20x | ⏳ Verify with GPU tests |
| rolling_median | 40-60x | ⏳ Verify with GPU tests |
| prefix_sum | 15-25x | ⏳ Verify with GPU tests |
| reduce_sum | 10x | ⏳ Verify with GPU tests |

## Next Steps

1. Analyze scaling behavior for production workloads
2. Profile GPU kernels with nsys for optimization opportunities
3. Test with real-world data sizes and patterns
4. Implement Phase 4B (async transfers, memory pooling)

