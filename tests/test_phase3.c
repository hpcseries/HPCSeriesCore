/*
 * Test for Phase 3 v0.2 kernel: group_reduce_variance
 */

#include "../include/hpcs_core.h"
#include <stdio.h>
#include <math.h>

#define EPSILON 1e-9

void print_array_double(const char *name, const double *arr, int n) {
    printf("  %s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%.4f ", arr[i]);
    }
    printf("\n");
}

void print_array_int(const char *name, const int *arr, int n) {
    printf("  %s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main(void)
{
    /* Test data: {1, 2, 3, 4, 5, 6} with 2 groups
     *
     * Group 0: [1, 2, 3]  -> mean=2, variance=2/3=0.6667
     * Group 1: [4, 5, 6]  -> mean=5, variance=2/3=0.6667
     *
     * Calculation for Group 0:
     *   mean = (1+2+3)/3 = 2
     *   variance = ((1-2)² + (2-2)² + (3-2)²) / 3 = (1+0+1)/3 = 2/3
     */

    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int group_ids[] = {0, 0, 0, 1, 1, 1};  // Two groups of 3 elements each
    int n = 6;
    int n_groups = 2;
    double result[2];
    int status;

    printf("Testing Phase 3 v0.2 Group Reduce Variance\n");
    printf("===========================================\n\n");

    printf("Test Setup:\n");
    print_array_double("Data", x, n);
    print_array_int("Groups", group_ids, n);
    printf("  Number of groups: %d\n\n", n_groups);

    /* Test 1: hpcs_group_reduce_variance - basic test */
    printf("Test 1: hpcs_group_reduce_variance\n");
    hpcs_group_reduce_variance(x, n, group_ids, n_groups, result, &status);
    printf("  Status: %d\n", status);
    print_array_double("Variance", result, n_groups);
    printf("  Expected: [0.6667, 0.6667]\n");

    double expected_var = 2.0 / 3.0;
    if (status == 0 &&
        fabs(result[0] - expected_var) < EPSILON &&
        fabs(result[1] - expected_var) < EPSILON) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n");
        printf("    Got: [%.6f, %.6f]\n", result[0], result[1]);
        printf("    Expected: [%.6f, %.6f]\n", expected_var, expected_var);
        return 1;
    }

    /* Test 2: Unbalanced groups */
    printf("Test 2: Unbalanced groups\n");
    double x2[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int group_ids2[] = {0, 0, 0, 1, 1};  // Group 0 has 3, Group 1 has 2
    /* Group 0: [1,2,3] -> mean=2, var=2/3
     * Group 1: [4,5]   -> mean=4.5, var=0.25
     */

    hpcs_group_reduce_variance(x2, 5, group_ids2, 2, result, &status);
    printf("  Status: %d\n", status);
    print_array_double("Variance", result, 2);
    printf("  Expected: [0.6667, 0.2500]\n");

    if (status == 0 &&
        fabs(result[0] - 2.0/3.0) < EPSILON &&
        fabs(result[1] - 0.25) < EPSILON) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 3: Single element per group */
    printf("Test 3: Single element per group (variance should be 0)\n");
    double x3[] = {1.0, 2.0, 3.0};
    int group_ids3[] = {0, 1, 2};  // 3 groups, 1 element each
    double result3[3];

    hpcs_group_reduce_variance(x3, 3, group_ids3, 3, result3, &status);
    printf("  Status: %d\n", status);
    print_array_double("Variance", result3, 3);
    printf("  Expected: [0.0000, 0.0000, 0.0000]\n");

    if (status == 0 &&
        fabs(result3[0]) < EPSILON &&
        fabs(result3[1]) < EPSILON &&
        fabs(result3[2]) < EPSILON) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 4: Empty group (should return NaN) */
    printf("Test 4: Empty group (should return NaN)\n");
    double x4[] = {1.0, 2.0, 3.0};
    int group_ids4[] = {0, 0, 0};  // Only group 0, group 1 is empty
    double result4[2];

    hpcs_group_reduce_variance(x4, 3, group_ids4, 2, result4, &status);
    printf("  Status: %d\n", status);
    printf("  Group 0 variance: %.4f\n", result4[0]);
    printf("  Group 1 variance: %.4f (should be NaN)\n", result4[1]);

    if (status == 0 &&
        fabs(result4[0] - 2.0/3.0) < EPSILON &&
        isnan(result4[1])) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 5: Error handling - invalid n */
    printf("Test 5: Error handling (n <= 0)\n");
    hpcs_group_reduce_variance(x, 0, group_ids, n_groups, result, &status);
    printf("  Status: %d (expected: 1 = HPCS_ERR_INVALID_ARGS)\n", status);
    if (status == 1) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 6: Error handling - invalid n_groups */
    printf("Test 6: Error handling (n_groups <= 0)\n");
    hpcs_group_reduce_variance(x, n, group_ids, 0, result, &status);
    printf("  Status: %d (expected: 1 = HPCS_ERR_INVALID_ARGS)\n", status);
    if (status == 1) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* Test 7: Integration with reduce_variance (single group should match) */
    printf("Test 7: Integration test (single group vs reduce_variance)\n");
    int group_ids_single[] = {0, 0, 0, 0, 0, 0};  // All in group 0
    double group_result[1];
    double reduce_result;

    hpcs_group_reduce_variance(x, n, group_ids_single, 1, group_result, &status);
    hpcs_reduce_variance(x, n, &reduce_result, &status);

    printf("  Group variance:  %.6f\n", group_result[0]);
    printf("  Reduce variance: %.6f\n", reduce_result);

    if (fabs(group_result[0] - reduce_result) < EPSILON) {
        printf("  ✓ PASS (results match)\n\n");
    } else {
        printf("  ✗ FAIL (results don't match)\n\n");
        return 1;
    }

    printf("===========================================\n");
    printf("All Phase 3 tests passed! ✓\n");
    return 0;
}
