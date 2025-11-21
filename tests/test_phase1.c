/*
 * Test for Phase 1 v0.2 kernels: reduce_mean, reduce_variance, reduce_std
 */

#include "../include/hpcs_core.h"
#include <stdio.h>
#include <math.h>

#define EPSILON 1e-9

int main(void)
{
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int n = 5;
    double result;
    int status;

    printf("Testing Phase 1 v0.2 Statistical Reductions\n");
    printf("============================================\n\n");

    /* Test data: {1, 2, 3, 4, 5}
     * Expected mean: 3.0
     * Expected variance: 2.0 (population)
     * Expected std: sqrt(2.0) ≈ 1.414213562
     */

    /* Test 1: hpcs_reduce_mean */
    printf("Test 1: hpcs_reduce_mean\n");
    hpcs_reduce_mean(x, n, &result, &status);
    printf("  Status: %d\n", status);
    printf("  Result: %.9f\n", result);
    printf("  Expected: 3.0\n");
    if (status == 0 && fabs(result - 3.0) < EPSILON) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 2: hpcs_reduce_variance */
    printf("Test 2: hpcs_reduce_variance\n");
    hpcs_reduce_variance(x, n, &result, &status);
    printf("  Status: %d\n", status);
    printf("  Result: %.9f\n", result);
    printf("  Expected: 2.0\n");
    if (status == 0 && fabs(result - 2.0) < EPSILON) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 3: hpcs_reduce_std */
    printf("Test 3: hpcs_reduce_std\n");
    hpcs_reduce_std(x, n, &result, &status);
    printf("  Status: %d\n", status);
    printf("  Result: %.9f\n", result);
    printf("  Expected: 1.414213562\n");
    if (status == 0 && fabs(result - sqrt(2.0)) < EPSILON) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 4: Edge case - single element */
    printf("Test 4: Single element (variance should be 0)\n");
    double x_single[] = {42.0};
    hpcs_reduce_variance(x_single, 1, &result, &status);
    printf("  Status: %d\n", status);
    printf("  Variance: %.9f\n", result);
    if (status == 0 && fabs(result - 0.0) < EPSILON) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 5: Error handling - invalid n */
    printf("Test 5: Error handling (n <= 0)\n");
    hpcs_reduce_mean(x, 0, &result, &status);
    printf("  Status: %d (expected: 1 = HPCS_ERR_INVALID_ARGS)\n", status);
    if (status == 1) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    printf("============================================\n");
    printf("All Phase 1 tests passed! ✓\n");
    return 0;
}
