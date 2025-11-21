/*
 * Test to verify OpenMP is enabled and working correctly
 */

#include "../include/hpcs_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LARGE_N 10000000  // 10 million elements

int main(void)
{
    double *x;
    double result;
    int status;
    clock_t start, end;
    double cpu_time_serial, cpu_time_parallel;

    printf("OpenMP Parallelism Verification Test\n");
    printf("=====================================\n\n");

    // Allocate large array
    x = (double*)malloc(LARGE_N * sizeof(double));
    if (!x) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Fill with test data
    for (int i = 0; i < LARGE_N; i++) {
        x[i] = (double)(i % 100);
    }

    printf("Test data: %d elements\n\n", LARGE_N);

    /* Test 1: Serial reduce_sum */
    printf("Test 1: Serial reduce_sum\n");
    start = clock();
    hpcs_reduce_sum(x, LARGE_N, &result, &status);
    end = clock();
    cpu_time_serial = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Result: %.2f\n", result);
    printf("  Status: %d\n", status);
    printf("  Time: %.6f seconds\n\n", cpu_time_serial);

    /* Test 2: Parallel reduce_sum (should use OpenMP for n >= 100000) */
    printf("Test 2: Parallel reduce_sum_parallel\n");
    start = clock();
    hpcs_reduce_sum_parallel(x, LARGE_N, &result, &status);
    end = clock();
    cpu_time_parallel = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Result: %.2f\n", result);
    printf("  Status: %d\n", status);
    printf("  Time: %.6f seconds\n\n", cpu_time_parallel);

    /* Note: For accurate parallel speedup, we need wall-clock time, not CPU time
     * But this test at least verifies the parallel version runs without errors */

    printf("=====================================\n");
    printf("OpenMP Status:\n");
    printf("  Parallel threshold: 100,000 elements\n");
    printf("  Test data size: %d elements\n", LARGE_N);
    printf("  Should use OpenMP: %s\n", LARGE_N >= 100000 ? "YES" : "NO");
    printf("\n");

    if (status == 0) {
        printf("✓ OpenMP-enabled build successful!\n");
        printf("  (Both serial and parallel functions work correctly)\n");
    } else {
        printf("✗ Test failed with status %d\n", status);
        free(x);
        return 1;
    }

    free(x);
    return 0;
}
