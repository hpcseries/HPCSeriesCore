/**
 * Fortran-SIMD Integration Test - v0.6
 *
 * Tests that Fortran can successfully call SIMD-accelerated reduction
 * kernels through the bridge layer and get correct results.
 *
 * Tests:
 * 1. Correctness: SIMD results match scalar Fortran results
 * 2. Performance: SIMD provides measurable speedup
 * 3. API compatibility: Fortran signatures work correctly
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Fortran scalar reduction functions (original)
extern void hpcs_reduce_sum(const double *x, int n, double *out, int *status);
extern void hpcs_reduce_mean(const double *x, int n, double *out, int *status);
extern void hpcs_reduce_min(const double *x, int n, double *out, int *status);
extern void hpcs_reduce_max(const double *x, int n, double *out, int *status);

// Fortran SIMD reduction functions (v0.6)
extern void hpcs_reduce_sum_simd(const double *x, int n, double *out, int *status);
extern void hpcs_reduce_mean_simd(const double *x, int n, double *out, int *status);
extern void hpcs_reduce_min_simd(const double *x, int n, double *out, int *status);
extern void hpcs_reduce_max_simd(const double *x, int n, double *out, int *status);

// SIMD initialization
extern void hpcs_simd_reductions_init(void);
extern void hpcs_intrinsics_init(void);
extern const char* hpcs_get_simd_name(void);
extern int hpcs_get_simd_width_doubles(void);

// Timing utility
static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

// Test data generation
static void generate_test_data(double *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (double)(i + 1);  // 1, 2, 3, ..., n (simple for verification)
    }
}

// Correctness test
static int test_correctness(const double *data, int n) {
    double scalar_result, simd_result;
    int status;
    int passed = 1;

    printf("=== Correctness Tests ===\n\n");

    // Test 1: Sum
    hpcs_reduce_sum(data, n, &scalar_result, &status);
    hpcs_reduce_sum_simd(data, n, &simd_result, &status);
    double expected_sum = (double)n * (n + 1) / 2.0;  // Sum of 1..n
    double sum_error = fabs(simd_result - scalar_result);
    printf("[1/4] Sum:  scalar=%.2f  simd=%.2f  expected=%.2f  error=%.2e  %s\n",
           scalar_result, simd_result, expected_sum, sum_error,
           sum_error < 1e-9 ? "✓ PASS" : "✗ FAIL");
    if (sum_error >= 1e-9) passed = 0;

    // Test 2: Mean
    hpcs_reduce_mean(data, n, &scalar_result, &status);
    hpcs_reduce_mean_simd(data, n, &simd_result, &status);
    double expected_mean = (n + 1) / 2.0;
    double mean_error = fabs(simd_result - scalar_result);
    printf("[2/4] Mean: scalar=%.2f  simd=%.2f  expected=%.2f  error=%.2e  %s\n",
           scalar_result, simd_result, expected_mean, mean_error,
           mean_error < 1e-9 ? "✓ PASS" : "✗ FAIL");
    if (mean_error >= 1e-9) passed = 0;

    // Test 3: Min
    hpcs_reduce_min(data, n, &scalar_result, &status);
    hpcs_reduce_min_simd(data, n, &simd_result, &status);
    double expected_min = 1.0;
    double min_error = fabs(simd_result - scalar_result);
    printf("[3/4] Min:  scalar=%.2f  simd=%.2f  expected=%.2f  error=%.2e  %s\n",
           scalar_result, simd_result, expected_min, min_error,
           min_error < 1e-9 ? "✓ PASS" : "✗ FAIL");
    if (min_error >= 1e-9) passed = 0;

    // Test 4: Max
    hpcs_reduce_max(data, n, &scalar_result, &status);
    hpcs_reduce_max_simd(data, n, &simd_result, &status);
    double expected_max = (double)n;
    double max_error = fabs(simd_result - scalar_result);
    printf("[4/4] Max:  scalar=%.2f  simd=%.2f  expected=%.2f  error=%.2e  %s\n",
           scalar_result, simd_result, expected_max, max_error,
           max_error < 1e-9 ? "✓ PASS" : "✗ FAIL");
    if (max_error >= 1e-9) passed = 0;

    printf("\n");
    return passed;
}

// Performance test
static void test_performance(const double *data, int n, int iterations) {
    double scalar_result, simd_result;
    int status;

    printf("=== Performance Tests ===\n\n");
    printf("Array size: %d elements (%.2f MB)\n", n, (n * sizeof(double)) / 1e6);
    printf("Iterations: %d\n\n", iterations);

    // Benchmark: Sum (scalar)
    double scalar_start = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        hpcs_reduce_sum(data, n, &scalar_result, &status);
    }
    double scalar_time = (get_time_sec() - scalar_start) / iterations;

    // Benchmark: Sum (SIMD)
    double simd_start = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        hpcs_reduce_sum_simd(data, n, &simd_result, &status);
    }
    double simd_time = (get_time_sec() - simd_start) / iterations;

    double speedup = scalar_time / simd_time;
    double throughput_scalar = (n / 1e6) / scalar_time;
    double throughput_simd = (n / 1e6) / simd_time;

    printf("Reduce Sum Performance:\n");
    printf("  Scalar (Fortran): %.6f sec  (%.2f M elem/sec)\n",
           scalar_time, throughput_scalar);
    printf("  SIMD (v0.6):      %.6f sec  (%.2f M elem/sec)\n",
           simd_time, throughput_simd);
    printf("  Speedup:          %.2fx\n", speedup);
    printf("\n");
}

int main(int argc, char *argv[]) {
    int n = 10000000;  // 10M elements (80 MB)
    int iterations = 10;

    printf("=== Fortran-SIMD Integration Test (v0.6) ===\n\n");

    // Parse arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    // Initialize SIMD system
    printf("Initializing SIMD system...\n");
    hpcs_simd_reductions_init();
    hpcs_intrinsics_init();
    printf("SIMD ISA: %s (%dx doubles)\n\n",
           hpcs_get_simd_name(), hpcs_get_simd_width_doubles());

    // Allocate test data
    double *data = (double*)malloc(n * sizeof(double));
    if (!data) {
        fprintf(stderr, "ERROR: Failed to allocate memory\n");
        return 1;
    }
    generate_test_data(data, n);

    // Run correctness tests (small array for verification)
    int correctness_passed = test_correctness(data, 100);

    // Run performance tests (large array)
    test_performance(data, n, iterations);

    // Summary
    printf("=== Summary ===\n\n");
    if (correctness_passed) {
        printf("✓ All correctness tests PASSED\n");
        printf("✓ Fortran-SIMD integration is working correctly!\n");
    } else {
        printf("✗ Some correctness tests FAILED\n");
        printf("✗ Please review SIMD implementation\n");
    }

    free(data);
    return correctness_passed ? 0 : 1;
}
