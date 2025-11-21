/*
 * Test for Phase 6 v0.2: Parallel Implementations
 *
 * Tests parallel versions of reduce operations and verifies:
 * 1. Correctness: Results match serial versions
 * 2. Threshold behavior: Falls back to serial for small arrays
 * 3. Large array handling: Works correctly with OpenMP parallelization
 */

#include "../include/hpcs_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-9
#define SMALL_SIZE 1000        // Below threshold (100000)
#define LARGE_SIZE 200000      // Above threshold

int main(void)
{
    int status;
    clock_t start, end;
    double cpu_time;

    printf("Testing Phase 6 v0.2 Parallel Implementations\n");
    printf("==============================================\n");
    printf("Parallel threshold: 100000 elements\n\n");

    /* ===================================================================== */
    /* Test 1: Parallel reduce_mean - small array (serial fallback)         */
    /* ===================================================================== */
    printf("Test 1: hpcs_reduce_mean_parallel (small array, n=%d)\n", SMALL_SIZE);
    double *x1 = (double*)malloc(SMALL_SIZE * sizeof(double));
    for (int i = 0; i < SMALL_SIZE; i++) {
        x1[i] = (double)(i % 100);  // Values 0-99 repeating
    }

    double mean_serial, mean_parallel;
    hpcs_reduce_mean(x1, SMALL_SIZE, &mean_serial, &status);
    hpcs_reduce_mean_parallel(x1, SMALL_SIZE, &mean_parallel, &status);

    printf("  Serial result:   %.6f\n", mean_serial);
    printf("  Parallel result: %.6f\n", mean_parallel);
    printf("  Expected: ~49.5 (average of 0-99)\n");

    if (status == 0 && fabs(mean_serial - mean_parallel) < EPSILON) {
        printf("  ✓ PASS (results match)\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        free(x1);
        return 1;
    }
    free(x1);

    /* ===================================================================== */
    /* Test 2: Parallel reduce_mean - large array (parallel execution)      */
    /* ===================================================================== */
    printf("Test 2: hpcs_reduce_mean_parallel (large array, n=%d)\n", LARGE_SIZE);
    double *x2 = (double*)malloc(LARGE_SIZE * sizeof(double));
    for (int i = 0; i < LARGE_SIZE; i++) {
        x2[i] = (double)(i % 1000);  // Values 0-999 repeating
    }

    double mean2_serial, mean2_parallel;

    start = clock();
    hpcs_reduce_mean(x2, LARGE_SIZE, &mean2_serial, &status);
    end = clock();
    double time_serial = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    start = clock();
    hpcs_reduce_mean_parallel(x2, LARGE_SIZE, &mean2_parallel, &status);
    end = clock();
    double time_parallel = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("  Serial result:   %.6f (%.2f ms)\n", mean2_serial, time_serial);
    printf("  Parallel result: %.6f (%.2f ms)\n", mean2_parallel, time_parallel);
    printf("  Expected: ~499.5 (average of 0-999)\n");

    if (status == 0 && fabs(mean2_serial - mean2_parallel) < EPSILON) {
        printf("  ✓ PASS (results match)\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        free(x2);
        return 1;
    }
    free(x2);

    /* ===================================================================== */
    /* Test 3: Parallel reduce_variance - small array                       */
    /* ===================================================================== */
    printf("Test 3: hpcs_reduce_variance_parallel (small array, n=%d)\n", SMALL_SIZE);
    double *x3 = (double*)malloc(SMALL_SIZE * sizeof(double));
    for (int i = 0; i < SMALL_SIZE; i++) {
        x3[i] = 10.0 + (double)(i % 10);  // Values 10-19 repeating
    }

    double var_serial, var_parallel;
    hpcs_reduce_variance(x3, SMALL_SIZE, &var_serial, &status);
    hpcs_reduce_variance_parallel(x3, SMALL_SIZE, &var_parallel, &status);

    printf("  Serial result:   %.6f\n", var_serial);
    printf("  Parallel result: %.6f\n", var_parallel);

    if (status == 0 && fabs(var_serial - var_parallel) < EPSILON) {
        printf("  ✓ PASS (results match)\n\n");
    } else {
        printf("  ✗ FAIL (difference: %.9f)\n\n", fabs(var_serial - var_parallel));
        free(x3);
        return 1;
    }
    free(x3);

    /* ===================================================================== */
    /* Test 4: Parallel reduce_std - large array                            */
    /* ===================================================================== */
    printf("Test 4: hpcs_reduce_std_parallel (large array, n=%d)\n", LARGE_SIZE);
    double *x4 = (double*)malloc(LARGE_SIZE * sizeof(double));
    for (int i = 0; i < LARGE_SIZE; i++) {
        x4[i] = 100.0 + (double)(i % 100);  // Values 100-199 repeating
    }

    double std_serial, std_parallel;
    hpcs_reduce_std(x4, LARGE_SIZE, &std_serial, &status);
    hpcs_reduce_std_parallel(x4, LARGE_SIZE, &std_parallel, &status);

    printf("  Serial result:   %.6f\n", std_serial);
    printf("  Parallel result: %.6f\n", std_parallel);

    if (status == 0 && fabs(std_serial - std_parallel) < EPSILON) {
        printf("  ✓ PASS (results match)\n\n");
    } else {
        printf("  ✗ FAIL (difference: %.9f)\n\n", fabs(std_serial - std_parallel));
        free(x4);
        return 1;
    }
    free(x4);

    /* ===================================================================== */
    /* Test 5: Parallel group_reduce_variance - small array                 */
    /* ===================================================================== */
    printf("Test 5: hpcs_group_reduce_variance_parallel (small, n=%d, groups=10)\n", SMALL_SIZE);
    double *x5 = (double*)malloc(SMALL_SIZE * sizeof(double));
    int *gids5 = (int*)malloc(SMALL_SIZE * sizeof(int));
    int n_groups5 = 10;

    for (int i = 0; i < SMALL_SIZE; i++) {
        x5[i] = (double)(i % 50);  // Values 0-49
        gids5[i] = i % n_groups5;  // Assign to groups 0-9
    }

    double *var5_serial = (double*)malloc(n_groups5 * sizeof(double));
    double *var5_parallel = (double*)malloc(n_groups5 * sizeof(double));

    hpcs_group_reduce_variance(x5, SMALL_SIZE, gids5, n_groups5, var5_serial, &status);
    hpcs_group_reduce_variance_parallel(x5, SMALL_SIZE, gids5, n_groups5, var5_parallel, &status);

    printf("  Comparing variances for %d groups:\n", n_groups5);
    int match5 = 1;
    for (int g = 0; g < n_groups5; g++) {
        if (fabs(var5_serial[g] - var5_parallel[g]) > EPSILON) {
            printf("  Group %d: Serial=%.6f, Parallel=%.6f (MISMATCH)\n",
                   g, var5_serial[g], var5_parallel[g]);
            match5 = 0;
        }
    }

    if (status == 0 && match5) {
        printf("  ✓ PASS (all group variances match)\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        free(x5); free(gids5); free(var5_serial); free(var5_parallel);
        return 1;
    }
    free(x5); free(gids5); free(var5_serial); free(var5_parallel);

    /* ===================================================================== */
    /* Test 6: Parallel group_reduce_variance - large array                 */
    /* ===================================================================== */
    printf("Test 6: hpcs_group_reduce_variance_parallel (large, n=%d, groups=100)\n", LARGE_SIZE);
    double *x6 = (double*)malloc(LARGE_SIZE * sizeof(double));
    int *gids6 = (int*)malloc(LARGE_SIZE * sizeof(int));
    int n_groups6 = 100;

    for (int i = 0; i < LARGE_SIZE; i++) {
        x6[i] = (double)(i % 1000);  // Values 0-999
        gids6[i] = i % n_groups6;    // Assign to groups 0-99
    }

    double *var6_serial = (double*)malloc(n_groups6 * sizeof(double));
    double *var6_parallel = (double*)malloc(n_groups6 * sizeof(double));

    start = clock();
    hpcs_group_reduce_variance(x6, LARGE_SIZE, gids6, n_groups6, var6_serial, &status);
    end = clock();
    time_serial = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    start = clock();
    hpcs_group_reduce_variance_parallel(x6, LARGE_SIZE, gids6, n_groups6, var6_parallel, &status);
    end = clock();
    time_parallel = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("  Serial time:   %.2f ms\n", time_serial);
    printf("  Parallel time: %.2f ms\n", time_parallel);

    int match6 = 1;
    for (int g = 0; g < n_groups6; g++) {
        if (fabs(var6_serial[g] - var6_parallel[g]) > EPSILON) {
            match6 = 0;
            break;
        }
    }

    if (status == 0 && match6) {
        printf("  ✓ PASS (all %d group variances match)\n\n", n_groups6);
    } else {
        printf("  ✗ FAIL\n\n");
        free(x6); free(gids6); free(var6_serial); free(var6_parallel);
        return 1;
    }
    free(x6); free(gids6); free(var6_serial); free(var6_parallel);

    /* ===================================================================== */
    /* Test 7: Error handling - invalid inputs                              */
    /* ===================================================================== */
    printf("Test 7: Error handling (invalid inputs)\n");
    double x7[] = {1.0, 2.0, 3.0};
    double out7;

    // Test n = 0
    hpcs_reduce_mean_parallel(x7, 0, &out7, &status);
    printf("  reduce_mean_parallel with n=0: Status=%d ", status);
    if (status == 1) {
        printf("✓\n");
    } else {
        printf("✗ (expected HPCS_ERR_INVALID_ARGS)\n");
        return 1;
    }

    // Test negative n
    hpcs_reduce_variance_parallel(x7, -1, &out7, &status);
    printf("  reduce_variance_parallel with n=-1: Status=%d ", status);
    if (status == 1) {
        printf("✓\n");
    } else {
        printf("✗ (expected HPCS_ERR_INVALID_ARGS)\n");
        return 1;
    }

    printf("  ✓ PASS\n\n");

    /* ===================================================================== */
    /* Test 8: Consistency check - all methods                              */
    /* ===================================================================== */
    printf("Test 8: Consistency check (serial vs parallel for all functions)\n");
    const int n8 = 150000;  // Above threshold
    double *x8 = (double*)malloc(n8 * sizeof(double));
    for (int i = 0; i < n8; i++) {
        x8[i] = 50.0 + (double)(i % 200);  // Values 50-249
    }

    double mean8_s, mean8_p, var8_s, var8_p, std8_s, std8_p;

    hpcs_reduce_mean(x8, n8, &mean8_s, &status);
    hpcs_reduce_mean_parallel(x8, n8, &mean8_p, &status);

    hpcs_reduce_variance(x8, n8, &var8_s, &status);
    hpcs_reduce_variance_parallel(x8, n8, &var8_p, &status);

    hpcs_reduce_std(x8, n8, &std8_s, &status);
    hpcs_reduce_std_parallel(x8, n8, &std8_p, &status);

    int all_match = 1;
    if (fabs(mean8_s - mean8_p) > EPSILON) {
        printf("  Mean mismatch: serial=%.6f, parallel=%.6f\n", mean8_s, mean8_p);
        all_match = 0;
    }
    if (fabs(var8_s - var8_p) > EPSILON) {
        printf("  Variance mismatch: serial=%.6f, parallel=%.6f\n", var8_s, var8_p);
        all_match = 0;
    }
    if (fabs(std8_s - std8_p) > EPSILON) {
        printf("  Std mismatch: serial=%.6f, parallel=%.6f\n", std8_s, std8_p);
        all_match = 0;
    }

    if (all_match) {
        printf("  Mean:     %.6f\n", mean8_p);
        printf("  Variance: %.6f\n", var8_p);
        printf("  Std:      %.6f\n", std8_p);
        printf("  ✓ PASS (all functions consistent)\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        free(x8);
        return 1;
    }
    free(x8);

    printf("==============================================\n");
    printf("All Phase 6 tests passed! ✓\n");
    printf("\nNote: Parallel speedup depends on system, data size,\n");
    printf("and OpenMP thread count (set OMP_NUM_THREADS).\n");
    return 0;
}
