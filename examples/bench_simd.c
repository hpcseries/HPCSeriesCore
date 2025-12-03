/**
 * HPCS SIMD Benchmark - v0.6
 *
 * Demonstrates performance improvements from SIMD vectorization.
 *
 * Compares:
 * 1. Scalar (no SIMD, single-threaded)
 * 2. SIMD-only (OpenMP SIMD, single-threaded)
 * 3. Parallel-only (OpenMP threads, no SIMD)
 * 4. Parallel + SIMD (hybrid, maximum performance)
 *
 * Shows expected speedups:
 * - SIMD-only: 2-8x (SSE2=2x, AVX2=4x, AVX-512=8x)
 * - Parallel-only: N-cores (e.g., 4x on 4-core)
 * - Parallel+SIMD: N-cores * SIMD-width (e.g., 16x on 4-core AVX2)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

// v0.6 SIMD API
extern void hpcs_simd_reductions_init(void);
extern void hpcs_intrinsics_init(void);
extern void hpcs_print_simd_status(void);
extern const char* hpcs_get_simd_name(void);
extern int hpcs_get_simd_width_doubles(void);

// SIMD reduction functions
extern double reduce_sum_openmp_simd(const double *x, int n);
extern double reduce_sum_parallel_simd(const double *x, int n);

// Timing utility
static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

// Generate test data
static void generate_data(double *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (double)rand() / RAND_MAX * 100.0;
    }
}

// ============================================================================
// Benchmark Implementations (Scalar, SIMD, Parallel, Hybrid)
// ============================================================================

/**
 * Scalar sum (baseline - no optimization)
 */
static double scalar_sum(const double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

/**
 * Parallel sum (OpenMP threads, no SIMD)
 */
static double parallel_sum(const double *x, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

typedef struct {
    const char *name;
    double (*func)(const double*, int);
} benchmark_t;

static void run_benchmark(const char *label, const double *data, int n,
                         int iterations, double (*func)(const double*, int)) {
    double result;
    double start = get_time_sec();

    for (int i = 0; i < iterations; i++) {
        result = func(data, n);
    }

    double end = get_time_sec();
    double elapsed = (end - start) / iterations;
    double throughput = (n / 1e6) / elapsed;  // M elements/sec

    // Prevent optimization
    if (result < 0.0) printf("");

    printf("  %-20s: %.6f sec  (%.2f M elem/sec)\n", label, elapsed, throughput);
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char *argv[]) {
    int n = 10000000;  // 10M elements (80 MB for doubles)
    int iterations = 10;

    printf("=== HPCSeries v0.6 SIMD Benchmark ===\n\n");

    // Parse arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    // Initialize SIMD system
    hpcs_simd_reductions_init();
    hpcs_intrinsics_init();  // Register hardware-specific intrinsics
    printf("\n");

    // Print system info
    hpcs_print_simd_status();
    printf("\n");

    printf("Test Parameters:\n");
    printf("  Array size:     %d elements (%.2f MB)\n", n, (n * sizeof(double)) / 1e6);
    printf("  Iterations:     %d\n", iterations);
    printf("  OpenMP threads: %d\n", omp_get_max_threads());
    printf("\n");

    // Allocate and generate test data
    double *data = (double*)malloc(n * sizeof(double));
    if (!data) {
        printf("ERROR: Failed to allocate memory\n");
        return 1;
    }
    generate_data(data, n);

    printf("=== Benchmark Results ===\n\n");

    // Benchmark 1: Scalar (baseline)
    printf("[1/4] Scalar (no optimization):\n");
    double scalar_time = 0.0;
    volatile double prevent_opt = 0.0;  // Prevent optimization
    {
        double result = 0.0;
        double start = get_time_sec();
        for (int i = 0; i < iterations; i++) {
            result += scalar_sum(data, n);
        }
        scalar_time = (get_time_sec() - start) / iterations;
        prevent_opt = result;  // Use result to prevent optimization
        double throughput = (n / 1e6) / scalar_time;
        printf("  Scalar sum          : %.6f sec  (%.2f M elem/sec)\n", scalar_time, throughput);
    }
    printf("\n");

    // Benchmark 2: SIMD-only (single-threaded vectorized)
    printf("[2/4] SIMD-only (vectorized, single-threaded):\n");
    double simd_time = 0.0;
    {
        double result = 0.0;
        double start = get_time_sec();
        for (int i = 0; i < iterations; i++) {
            result += reduce_sum_openmp_simd(data, n);
        }
        simd_time = (get_time_sec() - start) / iterations;
        prevent_opt = result;
        double throughput = (n / 1e6) / simd_time;
        printf("  OpenMP SIMD sum     : %.6f sec  (%.2f M elem/sec)\n", simd_time, throughput);
    }
    printf("  Speedup vs scalar: %.2fx\n", scalar_time / simd_time);
    printf("\n");

    // Benchmark 3: Parallel-only (multithreaded, no SIMD)
    printf("[3/4] Parallel-only (multithreaded, no SIMD):\n");
    double parallel_time = 0.0;
    {
        double result = 0.0;
        double start = get_time_sec();
        for (int i = 0; i < iterations; i++) {
            result += parallel_sum(data, n);
        }
        parallel_time = (get_time_sec() - start) / iterations;
        prevent_opt = result;
        double throughput = (n / 1e6) / parallel_time;
        printf("  Parallel sum        : %.6f sec  (%.2f M elem/sec)\n", parallel_time, throughput);
    }
    printf("  Speedup vs scalar: %.2fx\n", scalar_time / parallel_time);
    printf("\n");

    // Benchmark 4: Parallel + SIMD (hybrid - maximum performance)
    printf("[4/4] Parallel + SIMD (hybrid, maximum performance):\n");
    double hybrid_time = 0.0;
    {
        double result = 0.0;
        double start = get_time_sec();
        for (int i = 0; i < iterations; i++) {
            result += reduce_sum_parallel_simd(data, n);
        }
        hybrid_time = (get_time_sec() - start) / iterations;
        prevent_opt = result;
        double throughput = (n / 1e6) / hybrid_time;
        printf("  Parallel+SIMD sum   : %.6f sec  (%.2f M elem/sec)\n", hybrid_time, throughput);
    }
    printf("  Speedup vs scalar: %.2fx\n", scalar_time / hybrid_time);
    printf("\n");

    // Summary
    printf("=== Performance Summary ===\n\n");
    printf("Baseline (scalar):          %.6f sec  (1.00x)\n", scalar_time);
    printf("SIMD-only:                  %.6f sec  (%.2fx speedup)\n",
           simd_time, scalar_time / simd_time);
    printf("Parallel-only:              %.6f sec  (%.2fx speedup)\n",
           parallel_time, scalar_time / parallel_time);
    printf("Parallel + SIMD (BEST):     %.6f sec  (%.2fx speedup)\n",
           hybrid_time, scalar_time / hybrid_time);
    printf("\n");

    // Expected vs actual SIMD speedup
    int simd_width = hpcs_get_simd_width_doubles();
    int num_threads = omp_get_max_threads();

    printf("=== Analysis ===\n\n");
    printf("Hardware capabilities:\n");
    printf("  SIMD ISA:         %s\n", hpcs_get_simd_name());
    printf("  SIMD width:       %dx doubles\n", simd_width);
    printf("  CPU cores:        %d threads\n", num_threads);
    printf("\n");

    printf("Theoretical peak speedups:\n");
    printf("  SIMD-only:        %.1fx (SIMD width)\n", (double)simd_width);
    printf("  Parallel-only:    %.1fx (thread count)\n", (double)num_threads);
    printf("  Parallel + SIMD:  %.1fx (cores × SIMD)\n",
           (double)(num_threads * simd_width));
    printf("\n");

    double simd_efficiency = (scalar_time / simd_time) / simd_width * 100.0;
    double parallel_efficiency = (scalar_time / parallel_time) / num_threads * 100.0;
    double hybrid_efficiency = (scalar_time / hybrid_time) / (num_threads * simd_width) * 100.0;

    printf("Achieved efficiency:\n");
    printf("  SIMD-only:        %.1f%% of theoretical peak\n", simd_efficiency);
    printf("  Parallel-only:    %.1f%% of theoretical peak\n", parallel_efficiency);
    printf("  Parallel + SIMD:  %.1f%% of theoretical peak\n", hybrid_efficiency);
    printf("\n");

    // Recommendations
    printf("=== Recommendations ===\n\n");

    if (hybrid_efficiency > 70.0) {
        printf("✓ Excellent SIMD+Parallel performance!\n");
        printf("✓ v0.6 SIMD vectorization is working optimally.\n");
    } else if (hybrid_efficiency > 50.0) {
        printf("~ Good performance, some optimization headroom remains.\n");
        printf("~ Consider testing with aligned memory for better efficiency.\n");
    } else {
        printf("⚠ Lower than expected performance.\n");
        printf("⚠ Possible causes:\n");
        printf("  - Memory bandwidth saturation\n");
        printf("  - Cache pressure from test data size\n");
        printf("  - System load or thermal throttling\n");
    }

    printf("\n");

    printf("Key takeaway:\n");
    printf("  v0.6 SIMD vectorization provides %.1fx speedup\n",
           scalar_time / hybrid_time);
    printf("  when combined with v0.5's parallel auto-tuning!\n");

    free(data);
    return 0;
}
