/*
 * Benchmark harness for HPCSeries Core v0.2 kernels.
 *
 * This program benchmarks all v0.2 functions (Phases 1-6) including:
 * - Phase 1: Reduce operations (mean, variance, std)
 * - Phase 2: Rolling operations (variance, std)
 * - Phase 3: Group reduce variance
 * - Phase 4: Data utilities (normalize, fill_forward, fill_backward)
 * - Phase 5: Anomaly detection
 * - Phase 6: Parallel implementations
 *
 * Tests multiple array sizes to demonstrate:
 * - Performance scaling with data size
 * - Parallel threshold behavior (100K elements)
 * - Serial vs parallel speedup
 *
 * Output format: n,kernel,elapsed_seconds,throughput_melem_per_sec
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../include/hpcs_core.h"

/* Timing utility using CLOCK_MONOTONIC */
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Generate random data in range [0, 1] */
static void generate_data(double *data, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        data[i] = (double)rand() / RAND_MAX;
    }
}

/* Generate group IDs for grouped operations */
static void generate_groups(int *group_ids, int n, int n_groups) {
    for (int i = 0; i < n; i++) {
        group_ids[i] = i % n_groups;
    }
}

/* Inject some NaN values for fill operations */
static void inject_nans(double *data, int n, int nan_every) {
    for (int i = 0; i < n; i += nan_every) {
        data[i] = NAN;
    }
}

/* Macro for timing a kernel and printing result */
#define BENCH_KERNEL(n, name, code) do { \
    double start = get_time(); \
    code; \
    double end = get_time(); \
    double elapsed = end - start; \
    double throughput = (elapsed > 0.0) ? ((double)(n) / 1e6 / elapsed) : 0.0; \
    printf("%d,%s,%.6f,%.2f\n", n, name, elapsed, throughput); \
} while(0)

int main(void) {
    /* Array sizes: 10K (below threshold), 100K (at threshold), 1M, 10M */
    const int sizes[] = {10000, 100000, 1000000, 10000000};
    const int num_sizes = 4;

    const int window = 100;      /* Window size for rolling operations */
    const int n_groups = 100;    /* Number of groups for group operations */
    const double threshold = 3.0; /* Z-score threshold for anomaly detection */

    int status;

    /* CSV header */
    printf("n,kernel,elapsed_seconds,throughput_melem_per_sec\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];

        /* Allocate buffers */
        double *data = (double*)malloc(n * sizeof(double));
        double *out = (double*)malloc(n * sizeof(double));
        int *anomalies = (int*)malloc(n * sizeof(int));
        int *group_ids = (int*)malloc(n * sizeof(int));
        double *group_out = (double*)malloc(n_groups * sizeof(double));

        /* Generate test data */
        generate_data(data, n, 42);
        generate_groups(group_ids, n, n_groups);

        /* ================================================================== */
        /* Phase 1: Reduce Operations                                        */
        /* ================================================================== */

        /* Reduce mean */
        {
            double mean;
            BENCH_KERNEL(n, "reduce_mean",
                hpcs_reduce_mean(data, n, &mean, &status));
        }

        /* Reduce variance */
        {
            double variance;
            BENCH_KERNEL(n, "reduce_variance",
                hpcs_reduce_variance(data, n, &variance, &status));
        }

        /* Reduce std */
        {
            double std;
            BENCH_KERNEL(n, "reduce_std",
                hpcs_reduce_std(data, n, &std, &status));
        }

        /* Reduce min */
        {
            double min_val;
            BENCH_KERNEL(n, "reduce_min",
                hpcs_reduce_min(data, n, &min_val, &status));
        }

        /* Reduce max */
        {
            double max_val;
            BENCH_KERNEL(n, "reduce_max",
                hpcs_reduce_max(data, n, &max_val, &status));
        }

        /* ================================================================== */
        /* Phase 2: Rolling Operations                                       */
        /* ================================================================== */

        /* Rolling variance */
        BENCH_KERNEL(n, "rolling_variance",
            hpcs_rolling_variance(data, n, window, out, &status));

        /* Rolling std */
        BENCH_KERNEL(n, "rolling_std",
            hpcs_rolling_std(data, n, window, out, &status));

        /* ================================================================== */
        /* Phase 3: Group Reduce Variance                                    */
        /* ================================================================== */

        BENCH_KERNEL(n, "group_reduce_variance",
            hpcs_group_reduce_variance(data, n, group_ids, n_groups,
                                      group_out, &status));

        /* ================================================================== */
        /* Phase 4: Data Utilities                                           */
        /* ================================================================== */

        /* Normalize min-max */
        BENCH_KERNEL(n, "normalize_minmax",
            hpcs_normalize_minmax(data, n, out, &status));

        /* Fill forward - test with some NaN values */
        {
            double *data_with_nan = (double*)malloc(n * sizeof(double));
            generate_data(data_with_nan, n, 42);
            inject_nans(data_with_nan, n, 1000);  /* NaN every 1000 elements */

            BENCH_KERNEL(n, "fill_forward",
                hpcs_fill_forward(data_with_nan, n, out, &status));

            free(data_with_nan);
        }

        /* Fill backward */
        {
            double *data_with_nan = (double*)malloc(n * sizeof(double));
            generate_data(data_with_nan, n, 42);
            inject_nans(data_with_nan, n, 1000);

            BENCH_KERNEL(n, "fill_backward",
                hpcs_fill_backward(data_with_nan, n, out, &status));

            free(data_with_nan);
        }

        /* ================================================================== */
        /* Phase 5: Anomaly Detection                                        */
        /* ================================================================== */

        BENCH_KERNEL(n, "detect_anomalies",
            hpcs_detect_anomalies(data, n, threshold, anomalies, &status));

        /* ================================================================== */
        /* Phase 6: Parallel Implementations                                 */
        /* ================================================================== */

        /* Reduce mean parallel */
        {
            double mean;
            BENCH_KERNEL(n, "reduce_mean_parallel",
                hpcs_reduce_mean_parallel(data, n, &mean, &status));
        }

        /* Reduce variance parallel */
        {
            double variance;
            BENCH_KERNEL(n, "reduce_variance_parallel",
                hpcs_reduce_variance_parallel(data, n, &variance, &status));
        }

        /* Reduce std parallel */
        {
            double std;
            BENCH_KERNEL(n, "reduce_std_parallel",
                hpcs_reduce_std_parallel(data, n, &std, &status));
        }

        /* Group reduce variance parallel */
        BENCH_KERNEL(n, "group_reduce_variance_parallel",
            hpcs_group_reduce_variance_parallel(data, n, group_ids, n_groups,
                                               group_out, &status));

        /* Cleanup */
        free(data);
        free(out);
        free(anomalies);
        free(group_ids);
        free(group_out);
    }

    return 0;
}
