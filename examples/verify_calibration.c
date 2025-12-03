/**
 * Calibration Verification Benchmark
 *
 * Tests actual HPCS operations before/after calibration to measure
 * performance improvements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// HPCS functions
extern void hpcs_cpu_detect_init(void);
extern void hpcs_load_config(const char *path, int *status);

// Synthetic benchmark kernel (OpenMP parallel reduction)
static double synthetic_sum(const double *x, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

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

// Benchmark a simple operation
static double benchmark_operation(const double *data, int n, int iterations) {
    double result;

    double start = get_time_sec();

    for (int i = 0; i < iterations; i++) {
        result = synthetic_sum(data, n);
    }

    double end = get_time_sec();
    double elapsed = (end - start) / iterations;

    // Prevent optimization
    if (result < 0.0) printf("");

    return elapsed;
}

int main(int argc, char *argv[]) {
    int n = 5000000;  // 5M elements
    int iterations = 10;
    int status;

    printf("=== HPCS Calibration Verification ===\n\n");

    // Parse arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    printf("Test Parameters:\n");
    printf("  Array size:  %d elements\n", n);
    printf("  Iterations:  %d\n", iterations);
    printf("\n");

    // Allocate test data
    double *data = (double*)malloc(n * sizeof(double));
    if (!data) {
        printf("ERROR: Failed to allocate memory\n");
        return 1;
    }
    generate_data(data, n);

    // Test 1: Without calibration config (hardware defaults)
    printf("[1/2] Testing with hardware defaults...\n");
    hpcs_cpu_detect_init();
    double time_default = benchmark_operation(data, n, iterations);
    printf("      Average time: %.6f seconds\n", time_default);
    printf("      Throughput:   %.2f M elements/sec\n\n",
           (n / 1e6) / time_default);

    // Test 2: With calibration config
    printf("[2/2] Testing with calibration config...\n");
    hpcs_load_config(NULL, &status);  // Load ~/.hpcseries/config.json

    if (status != 0) {
        printf("      WARNING: Config not loaded (status=%d)\n", status);
        printf("      Using hardware defaults instead\n");
    } else {
        printf("      Config loaded successfully\n");
    }

    double time_calibrated = benchmark_operation(data, n, iterations);
    printf("      Average time: %.6f seconds\n", time_calibrated);
    printf("      Throughput:   %.2f M elements/sec\n\n",
           (n / 1e6) / time_calibrated);

    // Calculate improvement
    double improvement = ((time_default - time_calibrated) / time_default) * 100.0;

    printf("=== Results ===\n");
    printf("Default time:     %.6f seconds\n", time_default);
    printf("Calibrated time:  %.6f seconds\n", time_calibrated);

    if (improvement > 0) {
        printf("Improvement:      %.1f%% faster ✓\n", improvement);
    } else {
        printf("Improvement:      %.1f%% slower\n", -improvement);
        printf("\nNote: For this simple operation, calibration may not help.\n");
        printf("      Try with more complex operations (median, rolling ops).\n");
    }

    printf("\n");
    printf("Recommendation:\n");
    if (improvement > 5.0) {
        printf("  ✓ Calibration provided significant benefit!\n");
        printf("  ✓ Keep using calibrated config for production.\n");
    } else if (improvement > 0) {
        printf("  ~ Modest improvement. Calibration is working but gains are small.\n");
        printf("  ~ Consider testing with your actual workloads.\n");
    } else {
        printf("  ⚠ No improvement detected.\n");
        printf("  ⚠ Your workload may not benefit from parallel tuning.\n");
        printf("  ⚠ Or the dataset size is too small to show differences.\n");
    }

    free(data);
    return 0;
}
