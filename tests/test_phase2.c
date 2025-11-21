/*
 * Test for Phase 2 v0.2 kernels: rolling_variance, rolling_std
 */

#include "../include/hpcs_core.h"
#include <stdio.h>
#include <math.h>

#define EPSILON 1e-9

void print_array(const char *name, const double *arr, int n) {
    printf("  %s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%.4f ", arr[i]);
    }
    printf("\n");
}

int main(void)
{
    /* Test data: {1, 2, 3, 4, 5} with window=3
     *
     * Window 1 (i=0): [1]         -> mean=1, var=0
     * Window 2 (i=1): [1,2]       -> mean=1.5, var=0.25
     * Window 3 (i=2): [1,2,3]     -> mean=2, var=0.6667
     * Window 4 (i=3): [2,3,4]     -> mean=3, var=0.6667
     * Window 5 (i=4): [3,4,5]     -> mean=4, var=0.6667
     *
     * Expected variance (population): [0, 0.25, 0.6667, 0.6667, 0.6667]
     * Expected std: sqrt of variance
     */

    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int n = 5;
    int window = 3;
    double result[5];
    int status;

    printf("Testing Phase 2 v0.2 Rolling Operations\n");
    printf("========================================\n\n");
    printf("Test data: {1, 2, 3, 4, 5}, window=%d\n\n", window);

    /* Test 1: hpcs_rolling_variance */
    printf("Test 1: hpcs_rolling_variance\n");
    hpcs_rolling_variance(x, n, window, result, &status);
    printf("  Status: %d\n", status);
    print_array("Variance", result, n);
    printf("  Expected: [0.0000, 0.2500, 0.6667, 0.6667, 0.6667]\n");

    /* Verify results */
    double expected_var[] = {0.0, 0.25, 2.0/3.0, 2.0/3.0, 2.0/3.0};
    int pass = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(result[i] - expected_var[i]) > EPSILON) {
            pass = 0;
            printf("  ✗ FAIL at index %d: got %.6f, expected %.6f\n", i, result[i], expected_var[i]);
        }
    }
    if (pass && status == 0) {
        printf("  ✓ PASS\n\n");
    } else if (!pass) {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 2: hpcs_rolling_std */
    printf("Test 2: hpcs_rolling_std\n");
    hpcs_rolling_std(x, n, window, result, &status);
    printf("  Status: %d\n", status);
    print_array("Std Dev", result, n);

    double expected_std[] = {0.0, 0.5, sqrt(2.0/3.0), sqrt(2.0/3.0), sqrt(2.0/3.0)};
    printf("  Expected: [0.0000, 0.5000, 0.8165, 0.8165, 0.8165]\n");

    pass = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(result[i] - expected_std[i]) > EPSILON) {
            pass = 0;
            printf("  ✗ FAIL at index %d: got %.6f, expected %.6f\n", i, result[i], expected_std[i]);
        }
    }
    if (pass && status == 0) {
        printf("  ✓ PASS\n\n");
    } else if (!pass) {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 3: Edge case - window size 1 */
    printf("Test 3: Edge case - window=1 (variance should be 0)\n");
    hpcs_rolling_variance(x, n, 1, result, &status);
    printf("  Status: %d\n", status);
    print_array("Variance", result, n);

    pass = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(result[i] - 0.0) > EPSILON) {
            pass = 0;
        }
    }
    if (pass && status == 0) {
        printf("  ✓ PASS (all variances are 0.0)\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 4: Error handling */
    printf("Test 4: Error handling (n <= 0)\n");
    hpcs_rolling_variance(x, 0, window, result, &status);
    printf("  Status: %d (expected: 1 = HPCS_ERR_INVALID_ARGS)\n", status);
    if (status == 1) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 5: Error handling - invalid window */
    printf("Test 5: Error handling (window <= 0)\n");
    hpcs_rolling_variance(x, n, 0, result, &status);
    printf("  Status: %d (expected: 1 = HPCS_ERR_INVALID_ARGS)\n", status);
    if (status == 1) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    printf("========================================\n");
    printf("All Phase 2 tests passed! ✓\n");
    return 0;
}
