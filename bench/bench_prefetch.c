/**
 * HPCSeries Core - Prefetch Optimization Benchmark
 * =================================================
 *
 * Measures performance impact of explicit prefetch hints added in v0.6.
 *
 * Tests 3 variants:
 * 1. SIMD-only (no prefetch)
 * 2. SIMD + Prefetch
 * 3. Parallel + SIMD
 *
 * Expected results:
 * - Small arrays (<100K): Prefetch no benefit or slight harm
 * - Medium arrays (100K-1M): Prefetch 10-20% faster
 * - Large arrays (>1M): Parallel is fastest
 *
 * Usage:
 *   ./bench_prefetch
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// External functions from hpcs_reduce_simd.c
extern double reduce_sum_openmp_simd(const double *x, int n);
extern double reduce_sum_prefetch_simd(const double *x, int n);
extern double reduce_sum_parallel_simd(const double *x, int n);

extern double reduce_mean_openmp_simd(const double *x, int n);
extern double reduce_mean_prefetch_simd(const double *x, int n);
extern double reduce_mean_parallel_simd(const double *x, int n);

// External functions from hpcs_rolling_simd.c
extern void rolling_mean_openmp_simd(const double *x, int n, int window, double *result);
extern void rolling_mean_prefetch_simd(const double *x, int n, int window, double *result);
extern void rolling_mean_parallel_simd(const double *x, int n, int window, double *result);

// ============================================================================
// Benchmark Utilities
// ============================================================================

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void init_random_data(double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (double)rand() / RAND_MAX * 100.0 - 50.0;
    }
}

// ============================================================================
// Reduction Benchmarks
// ============================================================================

void benchmark_reduction(const char *op_name, int n, int iterations,
                        double (*func_simd)(const double*, int),
                        double (*func_prefetch)(const double*, int),
                        double (*func_parallel)(const double*, int)) {

    // Allocate and initialize data
    double *x = (double*)malloc(n * sizeof(double));
    init_random_data(x, n);

    printf("\n%s - Array size: %d elements (%.2f MB)\n", op_name, n, n * sizeof(double) / 1e6);
    printf("%-25s %12s %12s %10s\n", "Variant", "Time (ms)", "Throughput", "vs SIMD");
    printf("%-25s %12s %12s %10s\n", "-------", "--------", "----------", "-------");

    // Benchmark 1: SIMD-only (no prefetch)
    double t_start = get_time_ms();
    volatile double result1;
    for (int iter = 0; iter < iterations; iter++) {
        result1 = func_simd(x, n);
    }
    double t_simd = (get_time_ms() - t_start) / iterations;
    double throughput_simd = (n * sizeof(double) / 1e9) / (t_simd / 1000.0);  // GB/s
    printf("%-25s %12.3f %10.2f GB/s %8.2fx\n",
           "SIMD-only", t_simd, throughput_simd, 1.0);

    // Benchmark 2: SIMD + Prefetch
    t_start = get_time_ms();
    volatile double result2;
    for (int iter = 0; iter < iterations; iter++) {
        result2 = func_prefetch(x, n);
    }
    double t_prefetch = (get_time_ms() - t_start) / iterations;
    double throughput_prefetch = (n * sizeof(double) / 1e9) / (t_prefetch / 1000.0);
    double speedup_prefetch = t_simd / t_prefetch;
    printf("%-25s %12.3f %10.2f GB/s %8.2fx %s\n",
           "SIMD + Prefetch", t_prefetch, throughput_prefetch, speedup_prefetch,
           speedup_prefetch > 1.05 ? "✓" : speedup_prefetch < 0.95 ? "⚠" : "~");

    // Benchmark 3: Parallel + SIMD
    t_start = get_time_ms();
    volatile double result3;
    for (int iter = 0; iter < iterations; iter++) {
        result3 = func_parallel(x, n);
    }
    double t_parallel = (get_time_ms() - t_start) / iterations;
    double throughput_parallel = (n * sizeof(double) / 1e9) / (t_parallel / 1000.0);
    double speedup_parallel = t_simd / t_parallel;
    printf("%-25s %12.3f %10.2f GB/s %8.2fx\n",
           "Parallel + SIMD", t_parallel, throughput_parallel, speedup_parallel);

    // Summary
    double prefetch_benefit = (speedup_prefetch - 1.0) * 100.0;
    printf("\nPrefetch benefit: %+.1f%%\n", prefetch_benefit);

    if (prefetch_benefit > 5.0) {
        printf("✓ Prefetch is effective at this size\n");
    } else if (prefetch_benefit < -5.0) {
        printf("⚠ Prefetch is harmful at this size (cache pollution)\n");
    } else {
        printf("~ Prefetch has minimal impact at this size\n");
    }

    free(x);
}

// ============================================================================
// Rolling Operation Benchmarks
// ============================================================================

void benchmark_rolling(const char *op_name, int n, int window, int iterations) {

    // Allocate and initialize data
    double *x = (double*)malloc(n * sizeof(double));
    double *result = (double*)malloc(n * sizeof(double));
    init_random_data(x, n);

    int n_windows = n - window + 1;
    printf("\n%s - Array: %d elements, Window: %d\n", op_name, n, window);
    printf("%-25s %12s %12s %10s\n", "Variant", "Time (ms)", "Windows/sec", "vs SIMD");
    printf("%-25s %12s %12s %10s\n", "-------", "--------", "-----------", "-------");

    // Benchmark 1: SIMD-only
    double t_start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        rolling_mean_openmp_simd(x, n, window, result);
    }
    double t_simd = (get_time_ms() - t_start) / iterations;
    double windows_per_sec_simd = (n_windows * 1000.0) / t_simd;
    printf("%-25s %12.3f %10.1fM/s %8.2fx\n",
           "SIMD-only", t_simd, windows_per_sec_simd / 1e6, 1.0);

    // Benchmark 2: SIMD + Prefetch
    t_start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        rolling_mean_prefetch_simd(x, n, window, result);
    }
    double t_prefetch = (get_time_ms() - t_start) / iterations;
    double windows_per_sec_prefetch = (n_windows * 1000.0) / t_prefetch;
    double speedup_prefetch = t_simd / t_prefetch;
    printf("%-25s %12.3f %10.1fM/s %8.2fx %s\n",
           "SIMD + Prefetch", t_prefetch, windows_per_sec_prefetch / 1e6, speedup_prefetch,
           speedup_prefetch > 1.05 ? "✓" : speedup_prefetch < 0.95 ? "⚠" : "~");

    // Benchmark 3: Parallel + SIMD
    t_start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        rolling_mean_parallel_simd(x, n, window, result);
    }
    double t_parallel = (get_time_ms() - t_start) / iterations;
    double windows_per_sec_parallel = (n_windows * 1000.0) / t_parallel;
    double speedup_parallel = t_simd / t_parallel;
    printf("%-25s %12.3f %10.1fM/s %8.2fx\n",
           "Parallel + SIMD", t_parallel, windows_per_sec_parallel / 1e6, speedup_parallel);

    double prefetch_benefit = (speedup_prefetch - 1.0) * 100.0;
    printf("\nPrefetch benefit: %+.1f%%\n", prefetch_benefit);

    free(x);
    free(result);
}

// ============================================================================
// Main Benchmark Suite
// ============================================================================

int main() {
    srand(42);  // Reproducible results

    int num_threads = omp_get_max_threads();

    printf("============================================================\n");
    printf("HPCSeries Core v0.6 - Prefetch Optimization Benchmark\n");
    printf("============================================================\n");
    printf("\nSystem Info:\n");
    printf("  CPU Threads: %d\n", num_threads);
    printf("  OMP_NUM_THREADS: %d\n", omp_get_max_threads());
    printf("\nLegend:\n");
    printf("  ✓ = Prefetch is beneficial (>5%% speedup)\n");
    printf("  ~ = Prefetch has minimal impact (±5%%)\n");
    printf("  ⚠ = Prefetch is harmful (<-5%% slowdown)\n");

    printf("\n============================================================\n");
    printf("PART 1: Reduction Operations (Sum, Mean)\n");
    printf("============================================================\n");

    // Test various array sizes
    int sizes[] = {10000, 50000, 100000, 500000, 1000000, 5000000};
    int iterations[] = {1000, 500, 200, 100, 50, 10};

    for (int i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        benchmark_reduction("reduce_sum", sizes[i], iterations[i],
                           reduce_sum_openmp_simd,
                           reduce_sum_prefetch_simd,
                           reduce_sum_parallel_simd);
    }

    printf("\n============================================================\n");
    printf("PART 2: Rolling Operations (Rolling Mean)\n");
    printf("============================================================\n");

    int rolling_sizes[] = {100000, 500000, 1000000, 5000000};
    int rolling_iterations[] = {100, 50, 20, 10};
    int window = 50;

    for (int i = 0; i < sizeof(rolling_sizes) / sizeof(rolling_sizes[0]); i++) {
        benchmark_rolling("rolling_mean", rolling_sizes[i], window, rolling_iterations[i]);
    }

    printf("\n============================================================\n");
    printf("Summary\n");
    printf("============================================================\n");
    printf("\nExpected Results:\n");
    printf("  Small arrays (<100K):   Prefetch has no benefit or slight harm\n");
    printf("  Medium arrays (100K-1M): Prefetch should show 10-20%% speedup\n");
    printf("  Large arrays (>1M):      Parallel+SIMD is fastest overall\n");
    printf("\n");

    return 0;
}
