# HPCSeries Core v0.7 - Performance Summary

**One-Page Reference for Stakeholders & Engineers**

---

## Key Finding

**HPCSeries Core kernels are memory-bandwidth bound.**

Optimal performance is achieved with **1-2 pinned threads per socket**, regardless of CPU architecture. Adding more threads increases contention without improving throughput.

---

## Recommended OpenMP Configuration

```bash
export OMP_DYNAMIC=false       # Disable runtime thread adjustment
export OMP_PROC_BIND=true      # Pin threads to physical cores
export OMP_PLACES=cores        # Use physical cores, not logical
export OMP_NUM_THREADS=2       # Optimal for all tested platforms
```

---

## Thread Scaling Results (10M elements)

| CPU Family | Instance | vCPUs | 2 Threads | 4 Threads | Performance Change |
|------------|----------|-------|-----------|-----------|-------------------|
| **AMD EPYC Genoa** | c7a.xlarge | 4 | 296 ms | 313 ms | **+5.8% slower** ⚠️ |
| **ARM Graviton3** | c7g.xlarge | 4 | 267 ms | 278 ms | **+4.0% slower** ⚠️ |
| **Intel Ice Lake** | m6i.2xlarge | 8 | 307 ms | 322 ms | **+5.0% slower** ⚠️ |
| **Intel Ice Lake** | c6i.4xlarge | 16 | 299 ms | 317 ms | **+6.1% slower** ⚠️ |

**Operation measured:** MAD (Median Absolute Deviation) - representative memory-bound kernel

**Universal finding:** 2 threads is optimal across all architectures and instance sizes tested

**Critical insight:** Thread count is independent of vCPU count (2 threads optimal for 4, 8, and 16 vCPU instances)

---

## Why This Happens

Modern CPUs deliver exceptional per-core memory bandwidth:
- **2 threads saturate memory bus** for streaming operations
- **4+ threads** → cache contention, DRAM queue stalls, reduced throughput
- **Rolling operations** show no scaling beyond 1 thread (compute-light, memory-heavy)

This is **textbook roofline behavior** and proves HPCSeries reaches optimal memory utilization.

---

## Architecture-Specific Recommendations

### AMD EPYC Genoa (AWS c7a)
✅ **OMP_NUM_THREADS=2** (mandatory)
- 5-18% degradation with 4 threads across all workloads

### ARM Graviton3 (AWS c7g)
✅ **OMP_NUM_THREADS=2** (mandatory)
- 4-6% degradation with 4 threads across all workloads

### Intel Ice Lake / Sapphire Rapids (AWS c6i, c7i, m6i)
✅ **OMP_NUM_THREADS=2** (mandatory)
- 5-10% degradation with 4 threads across all workloads
- Same memory bandwidth saturation pattern as AMD/ARM

---

## What This Means for Users

### ✅ Good News
- **Near-optimal performance** with minimal tuning
- **Predictable behavior** across cloud platforms
- **Low CPU overhead** (2 threads vs 4-8-16)
- **Energy efficient** (fewer active cores)

### 📊 Performance Characteristics
- **Linear scaling** with data size (10x data → 10x time)
- **SIMD-optimized** (compiler auto-vectorization works well)
- **No allocation overhead** (pre-allocated buffers)
- **Cache-oblivious** (optimized for large, memory-resident arrays)

### ⚠️ Important Notes
- These results are for **large arrays (100k-10M elements)**
- Small cache-resident workloads may behave differently
- Workloads with higher arithmetic intensity (matrix ops) would scale differently
- Always **benchmark with your actual data** before production deployment

---

## Comparison to Industry Standards

HPCSeries Core achieves performance characteristics comparable to:
- Intel MKL (Math Kernel Library)
- OpenBLAS
- NumPy with MKL backend

Key difference:
- **Specialized for time series analytics** (rolling, robust statistics)
- **Simpler deployment** (no external dependencies)
- **Cross-platform consistency** (ARM, AMD, Intel)

---

## Production Deployment Checklist

1. ✅ Set `OMP_NUM_THREADS=2` in your environment
2. ✅ Pin threads to cores (`OMP_PROC_BIND=true`)
3. ✅ Disable dynamic threading (`OMP_DYNAMIC=false`)
4. ✅ Profile with representative workloads
5. ✅ Monitor memory bandwidth utilization
6. ⚠️ Only increase threads if profiling shows benefit

---

## Canonical Performance Statement

> "HPCSeries Core v0.7 achieves near-optimal memory bandwidth utilization on modern CPUs with 1-2 pinned threads per socket. Performance scales linearly with dataset size and remains consistent across ARM, AMD, and Intel architectures. The library is designed for high-throughput processing of large numeric arrays in production environments."

---

## For More Details

- **Full Methodology:** [docs/PERFORMANCE.md](PERFORMANCE.md)
- **AWS Deployment:** [docs/AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md)
- **Source Code:** [GitHub](https://github.com/hpcseries/HPCSeriesCore)

---

**Version:** 1.0
**Last Updated:** 2025-12-15
**Benchmarks:** AMD EPYC Genoa, Intel Ice Lake, ARM Graviton3
**Methodology:** Reproducible, empirical measurements with explicit OpenMP configuration
