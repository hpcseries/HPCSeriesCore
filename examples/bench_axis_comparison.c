/**
 * Comprehensive Axis Operations Benchmark
 *
 * Compares performance of different implementations:
 * 1. Fortran OpenMP (existing)
 * 2. C SIMD-only (new)
 * 3. C Parallel-only (new)
 * 4. C Hybrid SIMD+Parallel (new)
 * 5. C Cache-blocked (new)
 *
 * Goal: Determine the winner for each array size range and update
 * architecture-aware dispatch thresholds accordingly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// Fortran prototypes (existing implementations)
extern void hpcs_reduce_sum_axis0(const double *x, int n, int m, double *y, int *status);
extern void hpcs_reduce_mean_axis0(const double *x, int n, int m, double *y, int *status);

// C SIMD prototypes (new implementations)
extern void reduce_sum_axis0_simd_only(const double *x, int n, int m, double *y);
extern void reduce_sum_axis0_parallel_only(const double *x, int n, int m, double *y);
extern void reduce_sum_axis0_hybrid(const double *x, int n, int m, double *y);
extern void reduce_sum_axis0_blocked(const double *x, int n, int m, double *y);
extern void reduce_mean_axis0_simd_only(const double *x, int n, int m, double *y);
extern void reduce_mean_axis0_hybrid(const double *x, int n, int m, double *y);

// CPU detection
extern void hpcs_cpu_detect_init(void);

typedef struct {
    int num_physical_cores;
    int num_logical_cores;
    int l1_cache_size_kb;
    int l2_cache_size_kb;
    int l3_cache_size_kb;
    int optimal_threads;
    int numa_nodes;
    int cores_per_numa_node;
    int *core_to_numa_map;
    int has_sse2;
    int has_avx;
    int has_avx2;
    int has_avx512;
    int has_neon;
    int has_fma3;
    int simd_width_bits;
    char cpu_vendor[64];
    char cpu_model[128];
    int initialized;
} hpcs_cpu_info_t;

extern void hpcs_cpu_detect_enhanced(hpcs_cpu_info_t *info);

// ============================================================================
// Timing utilities
// ============================================================================

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================================
// Test data generation
// ============================================================================

static void generate_test_matrix(double *x, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        x[i] = (double)(rand() % 1000) / 100.0 - 5.0;  // Random [-5, 5]
    }
}

// ============================================================================
// Correctness validation
// ============================================================================

static int validate_results(const double *y1, const double *y2, int m, const char *name1, const char *name2) {
    double max_error = 0.0;
    int errors = 0;

    for (int i = 0; i < m; i++) {
        double error = fabs(y1[i] - y2[i]);
        if (error > 1e-6) {
            errors++;
            if (errors <= 3) {  // Show first 3 errors
                fprintf(stderr, "❌ Mismatch at col %d: %s=%.10f, %s=%.10f (error=%.2e)\n",
                        i, name1, y1[i], name2, y2[i], error);
            }
        }
        if (error > max_error) max_error = error;
    }

    if (errors == 0) {
        printf("✓ Validation PASSED (max error: %.2e)\n", max_error);
        return 1;
    } else {
        printf("❌ Validation FAILED (%d errors, max error: %.2e)\n", errors, max_error);
        return 0;
    }
}

// ============================================================================
// Benchmark runner
// ============================================================================

typedef struct {
    const char *name;
    void (*func)(const double*, int, int, double*);
    double time_sec;
    double throughput_gelem_sec;
} benchmark_result_t;

static void run_benchmark(const char *name,
                         void (*func)(const double*, int, int, double*),
                         const double *x, int n, int m, double *y,
                         int iterations, benchmark_result_t *result) {
    // Warmup
    func(x, n, m, y);

    // Timed run
    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        func(x, n, m, y);
    }
    double elapsed = get_time() - start;

    result->name = name;
    result->func = func;
    result->time_sec = elapsed / iterations;
    result->throughput_gelem_sec = ((double)n * m) / (result->time_sec * 1e9);
}

// Fortran wrapper
static void fortran_sum_wrapper(const double *x, int n, int m, double *y) {
    int status;
    hpcs_reduce_sum_axis0(x, n, m, y, &status);
}

static void fortran_mean_wrapper(const double *x, int n, int m, double *y) {
    int status;
    hpcs_reduce_mean_axis0(x, n, m, y, &status);
}

// ============================================================================
// Main benchmark suite
// ============================================================================

int main(void) {
    printf("=== HPCSeries Axis Operations Benchmark v0.6 ===\n\n");

    // Initialize CPU detection
    hpcs_cpu_detect_init();
    hpcs_cpu_info_t cpu_info;
    memset(&cpu_info, 0, sizeof(cpu_info));
    hpcs_cpu_detect_enhanced(&cpu_info);

    printf("=== System Information ===\n");
    printf("CPU:             %s\n", cpu_info.cpu_vendor);
    printf("Logical cores:   %d\n", cpu_info.num_logical_cores);
    printf("Physical cores:  %d\n", cpu_info.num_physical_cores);
    printf("Optimal threads: %d\n", cpu_info.optimal_threads);
    printf("L1 cache:        %d KB\n", cpu_info.l1_cache_size_kb);
    printf("L2 cache:        %d KB\n", cpu_info.l2_cache_size_kb);
    printf("L3 cache:        %d KB\n", cpu_info.l3_cache_size_kb);
    printf("SIMD width:      %d bits", cpu_info.simd_width_bits);
    if (cpu_info.has_avx512) printf(" (AVX-512)");
    else if (cpu_info.has_avx2) printf(" (AVX2)");
    else if (cpu_info.has_avx) printf(" (AVX)");
    else if (cpu_info.has_sse2) printf(" (SSE2)");
    printf("\n\n");

    // Test configurations: (rows, columns, iterations)
    typedef struct {
        const char *size_name;
        int n, m;
        int iterations;
    } test_config_t;

    test_config_t configs[] = {
        {"Tiny (100×100)", 100, 100, 1000},
        {"Small (1K×100)", 1000, 100, 500},
        {"Medium (10K×100)", 10000, 100, 100},
        {"Large (100K×100)", 100000, 100, 50},
        {"Very Large (1M×100)", 1000000, 100, 10},
        {"Wide (10K×1K)", 10000, 1000, 20},
        {"Tall (100K×10)", 100000, 10, 50},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("=== BENCHMARK: reduce_sum_axis0 ===\n\n");

    for (int cfg_idx = 0; cfg_idx < num_configs; cfg_idx++) {
        test_config_t *cfg = &configs[cfg_idx];
        int n = cfg->n;
        int m = cfg->m;
        int iterations = cfg->iterations;

        printf("--- Test: %s (%d rows × %d cols = %.2f MB) ---\n",
               cfg->size_name, n, m, (n * m * sizeof(double)) / 1e6);

        // Allocate data
        double *x = (double*)malloc(n * m * sizeof(double));
        double *y_fortran = (double*)malloc(m * sizeof(double));
        double *y_simd = (double*)malloc(m * sizeof(double));
        double *y_parallel = (double*)malloc(m * sizeof(double));
        double *y_hybrid = (double*)malloc(m * sizeof(double));
        double *y_blocked = (double*)malloc(m * sizeof(double));

        if (!x || !y_fortran || !y_simd || !y_parallel || !y_hybrid || !y_blocked) {
            fprintf(stderr, "❌ Memory allocation failed\n");
            exit(1);
        }

        generate_test_matrix(x, n, m);

        // Run benchmarks
        benchmark_result_t results[5];

        run_benchmark("Fortran OpenMP", fortran_sum_wrapper, x, n, m, y_fortran, iterations, &results[0]);
        run_benchmark("C SIMD-only", reduce_sum_axis0_simd_only, x, n, m, y_simd, iterations, &results[1]);
        run_benchmark("C Parallel-only", reduce_sum_axis0_parallel_only, x, n, m, y_parallel, iterations, &results[2]);
        run_benchmark("C Hybrid SIMD+Parallel", reduce_sum_axis0_hybrid, x, n, m, y_hybrid, iterations, &results[3]);
        run_benchmark("C Cache-blocked", reduce_sum_axis0_blocked, x, n, m, y_blocked, iterations, &results[4]);

        // Validate correctness
        printf("\nValidation (vs Fortran OpenMP baseline):\n");
        validate_results(y_fortran, y_simd, m, "Fortran", "C SIMD-only");
        validate_results(y_fortran, y_parallel, m, "Fortran", "C Parallel-only");
        validate_results(y_fortran, y_hybrid, m, "Fortran", "C Hybrid");
        validate_results(y_fortran, y_blocked, m, "Fortran", "C Blocked");

        // Print performance results
        printf("\nPerformance:\n");
        printf("%-25s %12s %15s %10s\n", "Implementation", "Time (ms)", "Throughput", "Speedup");
        printf("%-25s %12s %15s %10s\n", "-------------------------", "------------", "---------------", "----------");

        double baseline_time = results[0].time_sec;

        for (int i = 0; i < 5; i++) {
            double speedup = baseline_time / results[i].time_sec;
            const char *winner = (speedup > 1.05) ? "  ⚡ FASTER" : (speedup < 0.95) ? "  SLOWER" : "";

            printf("%-25s %10.3f ms %11.2f Gelem/s %8.2fx%s\n",
                   results[i].name,
                   results[i].time_sec * 1000,
                   results[i].throughput_gelem_sec,
                   speedup,
                   winner);
        }

        // Find winner
        int winner_idx = 0;
        double best_time = results[0].time_sec;
        for (int i = 1; i < 5; i++) {
            if (results[i].time_sec < best_time) {
                best_time = results[i].time_sec;
                winner_idx = i;
            }
        }

        printf("\n🏆 WINNER: %s (%.2fx faster than Fortran OpenMP)\n",
               results[winner_idx].name,
               baseline_time / results[winner_idx].time_sec);

        printf("\n");

        free(x);
        free(y_fortran);
        free(y_simd);
        free(y_parallel);
        free(y_hybrid);
        free(y_blocked);
    }

    printf("\n=== RECOMMENDATIONS ===\n\n");
    printf("Based on these benchmarks, update the dispatch thresholds in hpcs_axis_simd.c:\n\n");
    printf("static axis_thresholds_t g_axis_thresholds = {\n");
    printf("    .simd_only_threshold = ???,    // Use SIMD-only below this size\n");
    printf("    .parallel_threshold = ???,     // Use Parallel-only between simd and hybrid\n");
    printf("    .hybrid_threshold = ???,       // Use Hybrid above this size\n");
    printf("    .blocked_threshold = ???       // Use Cache-blocking above this size\n");
    printf("};\n\n");

    printf("Suggested thresholds (based on typical results):\n");
    printf("- simd_only_threshold: 10000 (tiny/small arrays)\n");
    printf("- parallel_threshold: 100000 (medium arrays)\n");
    printf("- hybrid_threshold: 500000 (large arrays)\n");
    printf("- blocked_threshold: 5000000 (very large arrays)\n\n");

    printf("✓ Benchmark complete!\n");

    return 0;
}
