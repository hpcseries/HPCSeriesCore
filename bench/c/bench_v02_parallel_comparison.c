/*
 * Parallel vs Serial Comparison Benchmark for HPCSeries Core v0.2
 *
 * This benchmark specifically compares serial and parallel implementations
 * of Phase 6 functions to measure speedup characteristics. It helps answer:
 * - What is the actual speedup on this system?
 * - At what array size does parallelization become beneficial?
 * - How does speedup scale with problem size?
 *
 * Tests array sizes both below and above the parallel threshold (100K)
 * to demonstrate the automatic fallback behavior.
 *
 * Output format: n,kernel,serial_time,parallel_time,speedup
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../include/hpcs_core.h"

/* Timing utility */
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Generate random data */
static void generate_data(double *data, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        data[i] = (double)rand() / RAND_MAX;
    }
}

/* Generate group IDs */
static void generate_groups(int *group_ids, int n, int n_groups) {
    for (int i = 0; i < n; i++) {
        group_ids[i] = i % n_groups;
    }
}

/* Run multiple iterations and return median time */
static double benchmark_median(void (*func)(void), int iterations) {
    double *times = (double*)malloc(iterations * sizeof(double));

    /* Warm-up run */
    func();

    /* Timed runs */
    for (int i = 0; i < iterations; i++) {
        double start = get_time();
        func();
        double end = get_time();
        times[i] = end - start;
    }

    /* Simple median (assuming odd iterations) */
    /* For production, use proper sorting */
    double median = times[iterations / 2];

    free(times);
    return median;
}

int main(void) {
    /* Array sizes spanning the threshold */
    const int sizes[] = {
        10000,      /* 10K - well below threshold */
        50000,      /* 50K - below threshold */
        100000,     /* 100K - at threshold */
        200000,     /* 200K - above threshold */
        500000,     /* 500K - well above threshold */
        1000000,    /* 1M - large array */
        5000000,    /* 5M - very large */
        10000000    /* 10M - very large */
    };
    const int num_sizes = 8;
    const int n_groups = 100;
    const int iterations = 5;  /* Number of runs for median */

    int status;

    /* CSV header */
    printf("n,kernel,serial_time_sec,parallel_time_sec,speedup\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];

        /* Allocate buffers */
        double *data = (double*)malloc(n * sizeof(double));
        int *group_ids = (int*)malloc(n * sizeof(int));
        double *group_out = (double*)malloc(n_groups * sizeof(double));

        /* Generate test data */
        generate_data(data, n, 42);
        generate_groups(group_ids, n, n_groups);

        /* ================================================================== */
        /* Reduce Mean: Serial vs Parallel                                   */
        /* ================================================================== */
        {
            double mean_serial, mean_parallel;

            /* Serial timing */
            double start_serial = get_time();
            for (int i = 0; i < iterations; i++) {
                hpcs_reduce_mean(data, n, &mean_serial, &status);
            }
            double end_serial = get_time();
            double time_serial = (end_serial - start_serial) / iterations;

            /* Parallel timing */
            double start_parallel = get_time();
            for (int i = 0; i < iterations; i++) {
                hpcs_reduce_mean_parallel(data, n, &mean_parallel, &status);
            }
            double end_parallel = get_time();
            double time_parallel = (end_parallel - start_parallel) / iterations;

            double speedup = time_serial / time_parallel;
            printf("%d,reduce_mean,%.6f,%.6f,%.2f\n",
                   n, time_serial, time_parallel, speedup);
        }

        /* ================================================================== */
        /* Reduce Variance: Serial vs Parallel                               */
        /* ================================================================== */
        {
            double var_serial, var_parallel;

            double start_serial = get_time();
            for (int i = 0; i < iterations; i++) {
                hpcs_reduce_variance(data, n, &var_serial, &status);
            }
            double end_serial = get_time();
            double time_serial = (end_serial - start_serial) / iterations;

            double start_parallel = get_time();
            for (int i = 0; i < iterations; i++) {
                hpcs_reduce_variance_parallel(data, n, &var_parallel, &status);
            }
            double end_parallel = get_time();
            double time_parallel = (end_parallel - start_parallel) / iterations;

            double speedup = time_serial / time_parallel;
            printf("%d,reduce_variance,%.6f,%.6f,%.2f\n",
                   n, time_serial, time_parallel, speedup);
        }

        /* ================================================================== */
        /* Reduce Std: Serial vs Parallel                                    */
        /* ================================================================== */
        {
            double std_serial, std_parallel;

            double start_serial = get_time();
            for (int i = 0; i < iterations; i++) {
                hpcs_reduce_std(data, n, &std_serial, &status);
            }
            double end_serial = get_time();
            double time_serial = (end_serial - start_serial) / iterations;

            double start_parallel = get_time();
            for (int i = 0; i < iterations; i++) {
                hpcs_reduce_std_parallel(data, n, &std_parallel, &status);
            }
            double end_parallel = get_time();
            double time_parallel = (end_parallel - start_parallel) / iterations;

            double speedup = time_serial / time_parallel;
            printf("%d,reduce_std,%.6f,%.6f,%.2f\n",
                   n, time_serial, time_parallel, speedup);
        }

        /* ================================================================== */
        /* Group Reduce Variance: Serial vs Parallel                         */
        /* ================================================================== */
        {
            double start_serial = get_time();
            for (int i = 0; i < iterations; i++) {
                hpcs_group_reduce_variance(data, n, group_ids, n_groups,
                                          group_out, &status);
            }
            double end_serial = get_time();
            double time_serial = (end_serial - start_serial) / iterations;

            double start_parallel = get_time();
            for (int i = 0; i < iterations; i++) {
                hpcs_group_reduce_variance_parallel(data, n, group_ids, n_groups,
                                                   group_out, &status);
            }
            double end_parallel = get_time();
            double time_parallel = (end_parallel - start_parallel) / iterations;

            double speedup = time_serial / time_parallel;
            printf("%d,group_reduce_variance,%.6f,%.6f,%.2f\n",
                   n, time_serial, time_parallel, speedup);
        }

        /* Cleanup */
        free(data);
        free(group_ids);
        free(group_out);
    }

    return 0;
}
