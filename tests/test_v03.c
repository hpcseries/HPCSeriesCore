/*
 * Test Suite for HPCSeries Core v0.3
 * Robust Statistics & Data Quality Functions
 *
 * Tests all 8 new v0.3 kernels:
 * - hpcs_median
 * - hpcs_mad
 * - hpcs_quantile
 * - hpcs_rolling_median
 * - hpcs_rolling_mad
 * - hpcs_clip
 * - hpcs_winsorize_by_quantiles
 * - hpcs_robust_zscore
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/hpcs_core.h"

#define EPSILON 1e-6
#define ASSERT_NEAR(a, b, msg) \
    if (fabs((a) - (b)) > EPSILON) { \
        printf("  ✗ FAIL: %s (got %.6f, expected %.6f)\n", msg, a, b); \
        return 0; \
    }

#define ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) { \
        printf("  ✗ FAIL: %s (got %d, expected %d)\n", msg, a, b); \
        return 0; \
    }

#define ASSERT_TRUE(cond, msg) \
    if (!(cond)) { \
        printf("  ✗ FAIL: %s\n", msg); \
        return 0; \
    }

/* Test 1: hpcs_median - basic median computation */
int test_median_basic() {
    printf("\nTest 1: hpcs_median (basic)\n");

    /* Odd length array */
    double x1[] = {5.0, 1.0, 3.0, 9.0, 7.0};
    double median1;
    int status1;
    hpcs_median(x1, 5, &median1, &status1);
    ASSERT_EQ(status1, 0, "Status should be 0");
    ASSERT_NEAR(median1, 5.0, "Median of [5,1,3,9,7]");

    /* Even length array */
    double x2[] = {1.0, 2.0, 3.0, 4.0};
    double median2;
    int status2;
    hpcs_median(x2, 4, &median2, &status2);
    ASSERT_EQ(status2, 0, "Status should be 0");
    ASSERT_NEAR(median2, 2.5, "Median of [1,2,3,4]");

    /* Single element */
    double x3[] = {42.0};
    double median3;
    int status3;
    hpcs_median(x3, 1, &median3, &status3);
    ASSERT_EQ(status3, 0, "Status should be 0");
    ASSERT_NEAR(median3, 42.0, "Median of single element");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 2: hpcs_mad - median absolute deviation */
int test_mad_basic() {
    printf("\nTest 2: hpcs_mad (basic)\n");

    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double mad;
    int status;
    hpcs_mad(x, 9, &mad, &status);
    ASSERT_EQ(status, 0, "Status should be 0");
    /* Median is 5.0, deviations: [4,3,2,1,0,1,2,3,4], median of deviations: 2.0 */
    ASSERT_NEAR(mad, 2.0, "MAD of [1..9]");

    /* Constant array (degenerate) */
    double x2[] = {5.0, 5.0, 5.0, 5.0};
    double mad2;
    int status2;
    hpcs_mad(x2, 4, &mad2, &status2);
    ASSERT_EQ(status2, 2, "Status should be 2 (degenerate)");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 3: hpcs_quantile - quantile computation */
int test_quantile_basic() {
    printf("\nTest 3: hpcs_quantile (basic)\n");

    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double value;
    int status;

    /* q = 0.0 (minimum) */
    hpcs_quantile(x, 10, 0.0, &value, &status);
    ASSERT_EQ(status, 0, "Status should be 0");
    ASSERT_NEAR(value, 1.0, "0.0-quantile (min)");

    /* q = 1.0 (maximum) */
    hpcs_quantile(x, 10, 1.0, &value, &status);
    ASSERT_EQ(status, 0, "Status should be 0");
    ASSERT_NEAR(value, 10.0, "1.0-quantile (max)");

    /* q = 0.5 (median) */
    hpcs_quantile(x, 10, 0.5, &value, &status);
    ASSERT_EQ(status, 0, "Status should be 0");
    ASSERT_NEAR(value, 5.5, "0.5-quantile (median)");

    /* q = 0.25 (first quartile) */
    hpcs_quantile(x, 10, 0.25, &value, &status);
    ASSERT_EQ(status, 0, "Status should be 0");
    /* Type 7: h = (10-1)*0.25 + 1 = 3.25, interpolate between 3rd and 4th */
    ASSERT_NEAR(value, 3.25, "0.25-quantile");

    /* Invalid q */
    hpcs_quantile(x, 10, 1.5, &value, &status);
    ASSERT_EQ(status, 1, "Status should be 1 (invalid q)");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 4: hpcs_rolling_median - rolling window median */
int test_rolling_median_basic() {
    printf("\nTest 4: hpcs_rolling_median (basic)\n");

    double x[] = {1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0};
    double y[7];
    int status;

    hpcs_rolling_median(x, 7, 3, y, &status);
    ASSERT_EQ(status, 0, "Status should be 0");

    /* First 2 values should be NaN */
    ASSERT_TRUE(isnan(y[0]), "y[0] should be NaN");
    ASSERT_TRUE(isnan(y[1]), "y[1] should be NaN");

    /* y[2] = median([1,5,3]) = 3.0 */
    ASSERT_NEAR(y[2], 3.0, "y[2] = median([1,5,3])");

    /* y[3] = median([5,3,7]) = 5.0 */
    ASSERT_NEAR(y[3], 5.0, "y[3] = median([5,3,7])");

    /* y[4] = median([3,7,2]) = 3.0 */
    ASSERT_NEAR(y[4], 3.0, "y[4] = median([3,7,2])");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 5: hpcs_rolling_mad - rolling window MAD */
int test_rolling_mad_basic() {
    printf("\nTest 5: hpcs_rolling_mad (basic)\n");

    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double y[9];
    int status;

    hpcs_rolling_mad(x, 9, 5, y, &status);
    ASSERT_EQ(status, 0, "Status should be 0");

    /* First 4 values should be NaN */
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(isnan(y[i]), "Early values should be NaN");
    }

    /* y[4] = MAD([1,2,3,4,5]): median=3, deviations=[2,1,0,1,2], MAD=1 */
    ASSERT_NEAR(y[4], 1.0, "y[4] = MAD([1,2,3,4,5])");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 6: hpcs_clip - value clipping */
int test_clip_basic() {
    printf("\nTest 6: hpcs_clip (basic)\n");

    double x[] = {1.0, 5.0, 10.0, 15.0, 20.0};
    int status;

    hpcs_clip(x, 5, 5.0, 15.0, &status);
    ASSERT_EQ(status, 0, "Status should be 0");

    ASSERT_NEAR(x[0], 5.0, "x[0] clipped to 5.0");
    ASSERT_NEAR(x[1], 5.0, "x[1] stays 5.0");
    ASSERT_NEAR(x[2], 10.0, "x[2] stays 10.0");
    ASSERT_NEAR(x[3], 15.0, "x[3] stays 15.0");
    ASSERT_NEAR(x[4], 15.0, "x[4] clipped to 15.0");

    /* Invalid arguments */
    double x2[] = {1.0, 2.0};
    int status2;
    hpcs_clip(x2, 2, 10.0, 5.0, &status2);  /* min > max */
    ASSERT_EQ(status2, 1, "Status should be 1 (invalid bounds)");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 7: hpcs_winsorize_by_quantiles - quantile-based clipping */
int test_winsorize_basic() {
    printf("\nTest 7: hpcs_winsorize_by_quantiles (basic)\n");

    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    int status;

    /* Winsorize to [0.1, 0.9] quantiles */
    hpcs_winsorize_by_quantiles(x, 10, 0.1, 0.9, &status);
    ASSERT_EQ(status, 0, "Status should be 0");

    /* 0.1-quantile ≈ 1.9, 0.9-quantile ≈ 9.1 */
    /* First element should be clipped up, last element clipped down */
    ASSERT_TRUE(x[0] >= 1.0, "x[0] should be >= 1.0");
    ASSERT_TRUE(x[9] <= 10.0, "x[9] should be <= 10.0");

    /* Middle values should remain unchanged */
    ASSERT_NEAR(x[4], 5.0, "x[4] should stay 5.0");
    ASSERT_NEAR(x[5], 6.0, "x[5] should stay 6.0");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 8: hpcs_robust_zscore - robust z-score computation */
int test_robust_zscore_basic() {
    printf("\nTest 8: hpcs_robust_zscore (basic)\n");

    /* Data with one clear outlier */
    double x[] = {10.0, 11.0, 10.5, 10.2, 9.8, 10.3, 9.9, 50.0};
    double y[8];
    int status;

    hpcs_robust_zscore(x, 8, y, &status);
    ASSERT_EQ(status, 0, "Status should be 0");

    /* Median ≈ 10.15, MAD ≈ 0.25 */
    /* The outlier (50.0) should have a very high z-score */
    ASSERT_TRUE(fabs(y[7]) > 10.0, "Outlier should have high robust z-score");

    /* Normal values should have small z-scores */
    for (int i = 0; i < 7; i++) {
        ASSERT_TRUE(fabs(y[i]) < 3.0, "Normal values should have |z| < 3");
    }

    /* Degenerate case: constant array */
    double x2[] = {5.0, 5.0, 5.0, 5.0};
    double y2[4];
    int status2;
    hpcs_robust_zscore(x2, 4, y2, &status2);
    ASSERT_EQ(status2, 2, "Status should be 2 (degenerate MAD)");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 9: NaN handling */
int test_nan_handling() {
    printf("\nTest 9: NaN Handling\n");

    double x[] = {1.0, NAN, 3.0, 4.0, 5.0};
    double median;
    int status;

    hpcs_median(x, 5, &median, &status);
    ASSERT_EQ(status, 0, "Status should be 0");
    ASSERT_TRUE(isnan(median), "Median should be NaN when input contains NaN");

    /* MAD with NaN */
    double mad;
    hpcs_mad(x, 5, &mad, &status);
    ASSERT_EQ(status, 0, "Status should be 0");
    ASSERT_TRUE(isnan(mad), "MAD should be NaN when input contains NaN");

    /* Quantile with NaN */
    double value;
    hpcs_quantile(x, 5, 0.5, &value, &status);
    ASSERT_EQ(status, 0, "Status should be 0");
    ASSERT_TRUE(isnan(value), "Quantile should be NaN when input contains NaN");

    printf("  ✓ PASS\n");
    return 1;
}

/* Test 10: Edge cases and error handling */
int test_edge_cases() {
    printf("\nTest 10: Edge Cases\n");

    double result;
    int status;

    /* Empty array */
    hpcs_median(NULL, 0, &result, &status);
    ASSERT_EQ(status, 1, "n=0 should return error");

    /* Invalid quantile */
    double x[] = {1.0, 2.0, 3.0};
    hpcs_quantile(x, 3, -0.5, &result, &status);
    ASSERT_EQ(status, 1, "q<0 should return error");

    hpcs_quantile(x, 3, 1.5, &result, &status);
    ASSERT_EQ(status, 1, "q>1 should return error");

    /* Invalid rolling window */
    double y[3];
    hpcs_rolling_median(x, 3, 0, y, &status);
    ASSERT_EQ(status, 1, "window=0 should return error");

    hpcs_rolling_median(x, 3, 10, y, &status);
    ASSERT_EQ(status, 1, "window>n should return error");

    printf("  ✓ PASS\n");
    return 1;
}

/* Main test runner */
int main() {
    printf("==============================================\n");
    printf("HPCSeries Core v0.3 Test Suite\n");
    printf("Robust Statistics & Data Quality\n");
    printf("==============================================\n");

    int passed = 0;
    int total = 10;

    passed += test_median_basic();
    passed += test_mad_basic();
    passed += test_quantile_basic();
    passed += test_rolling_median_basic();
    passed += test_rolling_mad_basic();
    passed += test_clip_basic();
    passed += test_winsorize_basic();
    passed += test_robust_zscore_basic();
    passed += test_nan_handling();
    passed += test_edge_cases();

    printf("\n==============================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("==============================================\n");

    return (passed == total) ? 0 : 1;
}
