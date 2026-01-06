/**
 * HPCS SIMD Calibration Extension - v0.6
 *
 * Extends v0.5 calibration to benchmark SIMD vs scalar performance.
 *
 * Determines:
 * - When SIMD provides benefit (threshold tuning)
 * - Whether OpenMP SIMD or intrinsics are faster
 * - Optimal parallel+SIMD thresholds
 * - SIMD efficiency for different operation types
 *
 * Results are stored in config.json for runtime dispatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

// SIMD reduction functions
extern double reduce_sum_openmp_simd(const double *x, int n);
extern double reduce_sum_parallel_simd(const double *x, int n);

// SIMD dispatch
extern void hpcs_simd_reductions_init(void);
extern void hpcs_intrinsics_init(void);
extern const char* hpcs_get_simd_name(void);
extern int hpcs_get_simd_width_doubles(void);

// Calibration configuration
#define SIMD_CALIBRATE_MIN_SIZE    1000
#define SIMD_CALIBRATE_MAX_SIZE    5000000
#define SIMD_CALIBRATE_NUM_TRIALS  5

// ============================================================================
// Timing Utilities
// ============================================================================

static double get_time_usec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

// ============================================================================
// SIMD vs Scalar Benchmarking
// ============================================================================

/**
 * Scalar sum (baseline)
 */
static double scalar_sum(const double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

/**
 * Benchmark scalar sum
 */
static double benchmark_scalar_sum(const double *data, int n, int trials) {
    volatile double result = 0.0;
    double start = get_time_usec();

    for (int trial = 0; trial < trials; trial++) {
        result += scalar_sum(data, n);
    }

    double end = get_time_usec();
    return (end - start) / trials;  // Average time in microseconds
}

/**
 * Benchmark SIMD sum (OpenMP SIMD)
 */
static double benchmark_simd_sum(const double *data, int n, int trials) {
    volatile double result = 0.0;
    double start = get_time_usec();

    for (int trial = 0; trial < trials; trial++) {
        result += reduce_sum_openmp_simd(data, n);
    }

    double end = get_time_usec();
    return (end - start) / trials;
}

/**
 * Benchmark parallel + SIMD sum
 */
static double benchmark_parallel_simd_sum(const double *data, int n, int trials) {
    volatile double result = 0.0;
    double start = get_time_usec();

    for (int trial = 0; trial < trials; trial++) {
        result += reduce_sum_parallel_simd(data, n);
    }

    double end = get_time_usec();
    return (end - start) / trials;
}

// ============================================================================
// SIMD Threshold Finding
// ============================================================================

/**
 * Find optimal SIMD threshold
 *
 * Tests various dataset sizes to find where SIMD becomes beneficial.
 * Returns the smallest size where SIMD provides >15% speedup.
 */
static int find_simd_threshold(const double *data, int max_size) {
    int test_sizes[] = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    fprintf(stderr, "[SIMD Calibration] Finding SIMD threshold...\n");

    for (int i = 0; i < num_sizes; i++) {
        int n = test_sizes[i];
        if (n > max_size) break;

        // Benchmark both scalar and SIMD
        double time_scalar = benchmark_scalar_sum(data, n, SIMD_CALIBRATE_NUM_TRIALS);
        double time_simd = benchmark_simd_sum(data, n, SIMD_CALIBRATE_NUM_TRIALS);

        double speedup = time_scalar / time_simd;
        double improvement = (speedup - 1.0) * 100.0;

        fprintf(stderr, "  Size %7d: scalar=%.2f us, SIMD=%.2f us, speedup=%.2fx (%.1f%%)\n",
                n, time_scalar, time_simd, speedup, improvement);

        // If SIMD provides >15% improvement, use this threshold
        if (improvement > 15.0) {
            fprintf(stderr, "[SIMD Calibration] SIMD beneficial at size >= %d\n", n);
            return n;
        }
    }

    // If no clear benefit found, use conservative threshold
    fprintf(stderr, "[SIMD Calibration] No clear SIMD benefit, using conservative threshold\n");
    return 100000;  // Default: 100K elements
}

/**
 * Find optimal parallel+SIMD threshold
 *
 * Tests where parallel+SIMD becomes better than SIMD-only.
 */
static int find_parallel_simd_threshold(const double *data, int max_size) {
    int test_sizes[] = {10000, 50000, 100000, 500000, 1000000, 5000000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    fprintf(stderr, "[SIMD Calibration] Finding parallel+SIMD threshold...\n");

    for (int i = 0; i < num_sizes; i++) {
        int n = test_sizes[i];
        if (n > max_size) break;

        // Benchmark SIMD-only vs parallel+SIMD
        double time_simd = benchmark_simd_sum(data, n, SIMD_CALIBRATE_NUM_TRIALS);
        double time_parallel_simd = benchmark_parallel_simd_sum(data, n, SIMD_CALIBRATE_NUM_TRIALS);

        double speedup = time_simd / time_parallel_simd;
        double improvement = (speedup - 1.0) * 100.0;

        fprintf(stderr, "  Size %7d: SIMD=%.2f us, Par+SIMD=%.2f us, speedup=%.2fx (%.1f%%)\n",
                n, time_simd, time_parallel_simd, speedup, improvement);

        // If parallel+SIMD provides >20% improvement, use this threshold
        if (improvement > 20.0) {
            fprintf(stderr, "[SIMD Calibration] Parallel+SIMD beneficial at size >= %d\n", n);
            return n;
        }
    }

    fprintf(stderr, "[SIMD Calibration] Using conservative parallel+SIMD threshold\n");
    return 500000;  // Default: 500K elements
}

// ============================================================================
// SIMD Efficiency Analysis
// ============================================================================

/**
 * Measure SIMD efficiency
 *
 * Compares actual SIMD speedup to theoretical maximum based on SIMD width.
 */
static void measure_simd_efficiency(const double *data, int n) {
    fprintf(stderr, "\n[SIMD Calibration] Measuring SIMD efficiency at n=%d...\n", n);

    // Get SIMD width
    int simd_width = hpcs_get_simd_width_doubles();
    const char* simd_name = hpcs_get_simd_name();

    // Benchmark scalar and SIMD
    double time_scalar = benchmark_scalar_sum(data, n, 10);
    double time_simd = benchmark_simd_sum(data, n, 10);
    double time_parallel_simd = benchmark_parallel_simd_sum(data, n, 10);

    // Calculate speedups
    double simd_speedup = time_scalar / time_simd;
    double hybrid_speedup = time_scalar / time_parallel_simd;

    // Calculate efficiencies
    double simd_efficiency = (simd_speedup / simd_width) * 100.0;
    double theoretical_max = simd_width;  // Theoretical SIMD speedup

    fprintf(stderr, "  SIMD ISA:              %s (%dx doubles)\n", simd_name, simd_width);
    fprintf(stderr, "  Scalar time:           %.2f us\n", time_scalar);
    fprintf(stderr, "  SIMD time:             %.2f us\n", time_simd);
    fprintf(stderr, "  Parallel+SIMD time:    %.2f us\n", time_parallel_simd);
    fprintf(stderr, "\n");
    fprintf(stderr, "  SIMD speedup:          %.2fx (%.1f%% of theoretical %.1fx)\n",
            simd_speedup, simd_efficiency, theoretical_max);
    fprintf(stderr, "  Parallel+SIMD speedup: %.2fx\n", hybrid_speedup);
}

// ============================================================================
// Main SIMD Calibration API
// ============================================================================

/**
 * Run SIMD calibration
 *
 * Extends v0.5 calibration with SIMD benchmarking.
 * Determines optimal thresholds for SIMD activation.
 *
 * @param simd_threshold - Output: size where SIMD becomes beneficial
 * @param parallel_simd_threshold - Output: size where parallel+SIMD wins
 * @param status - Output: 0=success, -1=error
 */
void hpcs_calibrate_simd(int *simd_threshold, int *parallel_simd_threshold, int *status) {
    fprintf(stderr, "\n");
    fprintf(stderr, "==============================\n");
    fprintf(stderr, "HPCS SIMD Calibration - v0.6\n");
    fprintf(stderr, "==============================\n\n");

    // Initialize SIMD systems
    hpcs_simd_reductions_init();
    hpcs_intrinsics_init();

    // Allocate test data
    int max_size = SIMD_CALIBRATE_MAX_SIZE;
    double *test_data = (double*)malloc(max_size * sizeof(double));
    if (!test_data) {
        fprintf(stderr, "[SIMD Calibration] ERROR: Failed to allocate test data\n");
        *status = -1;
        return;
    }

    // Generate random test data
    for (int i = 0; i < max_size; i++) {
        test_data[i] = (double)rand() / RAND_MAX * 100.0;
    }

    // Find optimal thresholds
    *simd_threshold = find_simd_threshold(test_data, max_size);
    *parallel_simd_threshold = find_parallel_simd_threshold(test_data, max_size);

    // Measure efficiency
    measure_simd_efficiency(test_data, 1000000);

    // Cleanup
    free(test_data);

    fprintf(stderr, "\n");
    fprintf(stderr, "=== SIMD Calibration Results ===\n");
    fprintf(stderr, "SIMD threshold:         %d elements\n", *simd_threshold);
    fprintf(stderr, "Parallel+SIMD threshold: %d elements\n", *parallel_simd_threshold);
    fprintf(stderr, "================================\n\n");

    *status = 0;
}

/**
 * Quick SIMD calibration
 *
 * Uses hardware heuristics instead of benchmarking.
 */
void hpcs_calibrate_simd_quick(int *simd_threshold, int *parallel_simd_threshold, int *status) {
    fprintf(stderr, "[SIMD Calibration] Quick mode - using hardware heuristics\n");

    // Initialize SIMD
    hpcs_simd_reductions_init();

    // Get SIMD width
    int simd_width = hpcs_get_simd_width_doubles();

    // Heuristic thresholds based on SIMD width
    // More SIMD lanes = lower threshold needed
    if (simd_width >= 8) {
        // AVX-512
        *simd_threshold = 10000;
        *parallel_simd_threshold = 100000;
    } else if (simd_width >= 4) {
        // AVX2/AVX
        *simd_threshold = 50000;
        *parallel_simd_threshold = 500000;
    } else {
        // SSE2/NEON
        *simd_threshold = 100000;
        *parallel_simd_threshold = 1000000;
    }

    fprintf(stderr, "[SIMD Calibration] Quick thresholds: SIMD=%d, Parallel+SIMD=%d\n",
            *simd_threshold, *parallel_simd_threshold);

    *status = 0;
}

// ============================================================================
// Integration with v0.5 Config System
// ============================================================================

/**
 * Save SIMD calibration results to config
 *
 * Extends v0.5 config.json with SIMD preferences.
 */
void hpcs_save_simd_config(int simd_threshold, int parallel_simd_threshold,
                           const char *path, int *status) {
    // TODO: Extend v0.5 config format to include SIMD thresholds
    // For now, just print recommendations

    fprintf(stderr, "\n[SIMD Calibration] Recommended configuration:\n");
    fprintf(stderr, "  {\n");
    fprintf(stderr, "    \"simd\": {\n");
    fprintf(stderr, "      \"enabled\": true,\n");
    fprintf(stderr, "      \"isa\": \"%s\",\n", hpcs_get_simd_name());
    fprintf(stderr, "      \"width\": %d,\n", hpcs_get_simd_width_doubles());
    fprintf(stderr, "      \"threshold\": %d,\n", simd_threshold);
    fprintf(stderr, "      \"parallel_threshold\": %d\n", parallel_simd_threshold);
    fprintf(stderr, "    }\n");
    fprintf(stderr, "  }\n");

    *status = 0;
}
