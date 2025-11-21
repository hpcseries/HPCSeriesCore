/*
 * Test for Phase 5 v0.2: Anomaly Detection
 */

#include "../include/hpcs_core.h"
#include <stdio.h>
#include <math.h>

#define EPSILON 1e-9

void print_array_double(const char *name, const double *arr, int n) {
    printf("  %s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%.2f ", arr[i]);
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
    int status;

    printf("Testing Phase 5 v0.2 Anomaly Detection\n");
    printf("======================================\n\n");

    /* ===================================================================== */
    /* Test 1: Basic anomaly detection with 3-sigma rule                    */
    /* ===================================================================== */
    printf("Test 1: Basic anomaly detection (3-sigma rule)\n");
    /* Data: many normal values around 10, with one extreme outlier */
    double x1[] = {10.0, 9.5, 10.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.4,
                   10.0, 9.6, 10.2, 9.9, 10.1, 100.0, 10.0, 9.8, 10.0, 10.2};
    int anomaly1[20];
    int n1 = 20;
    double threshold1 = 3.0;

    hpcs_detect_anomalies(x1, n1, threshold1, anomaly1, &status);
    printf("  Status: %d\n", status);
    print_array_double("Data     ", x1, n1);
    print_array_int   ("Anomalies", anomaly1, n1);

    /* With many normal values around 10, mean ≈ 14.5 (skewed by outlier)
     * 100.0 is an extreme outlier that exceeds 3-sigma threshold
     * Index 15 should be flagged
     */
    printf("  Expected: Index 15 (100.0) flagged as anomaly\n");

    if (status == 0 && anomaly1[15] == 1) {
        // Verify others are not flagged
        int pass = 1;
        for (int i = 0; i < n1; i++) {
            if (i != 15 && anomaly1[i] != 0) {
                pass = 0;
                break;
            }
        }
        if (pass) {
            printf("  ✓ PASS\n\n");
        } else {
            printf("  ✗ FAIL (other values incorrectly flagged)\n\n");
            return 1;
        }
    } else {
        printf("  ✗ FAIL\n\n");
        printf("  Debug: anomaly[15]=%d\n", anomaly1[15]);
        return 1;
    }

    /* ===================================================================== */
    /* Test 2: No anomalies - all values within threshold                   */
    /* ===================================================================== */
    printf("Test 2: No anomalies (all values normal)\n");
    double x2[] = {10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.3, 9.7, 10.1};
    int anomaly2[10];
    int n2 = 10;
    double threshold2 = 3.0;

    hpcs_detect_anomalies(x2, n2, threshold2, anomaly2, &status);
    printf("  Status: %d\n", status);
    print_array_double("Data     ", x2, n2);
    print_array_int   ("Anomalies", anomaly2, n2);
    printf("  Expected: All 0 (no anomalies)\n");

    int all_normal = 1;
    for (int i = 0; i < n2; i++) {
        if (anomaly2[i] != 0) {
            all_normal = 0;
            break;
        }
    }

    if (status == 0 && all_normal) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* ===================================================================== */
    /* Test 3: Constant array (std = 0)                                     */
    /* ===================================================================== */
    printf("Test 3: Constant array (std = 0, no anomalies possible)\n");
    double x3[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    int anomaly3[5];
    int n3 = 5;
    double threshold3 = 3.0;

    hpcs_detect_anomalies(x3, n3, threshold3, anomaly3, &status);
    printf("  Status: %d\n", status);
    print_array_double("Data     ", x3, n3);
    print_array_int   ("Anomalies", anomaly3, n3);
    printf("  Expected: All 0 (std = 0, no anomalies)\n");

    all_normal = 1;
    for (int i = 0; i < n3; i++) {
        if (anomaly3[i] != 0) {
            all_normal = 0;
            break;
        }
    }

    if (status == 0 && all_normal) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* ===================================================================== */
    /* Test 4: Different threshold (tighter = more anomalies)               */
    /* ===================================================================== */
    printf("Test 4: Tighter threshold (2-sigma instead of 3-sigma)\n");
    double x4[] = {10.0, 10.0, 10.0, 10.0, 15.0, 10.0, 10.0, 10.0};
    int anomaly4[8];
    int n4 = 8;
    double threshold4 = 2.0;  // More strict

    hpcs_detect_anomalies(x4, n4, threshold4, anomaly4, &status);
    printf("  Status: %d\n", status);
    printf("  Threshold: %.1f\n", threshold4);
    print_array_double("Data     ", x4, n4);
    print_array_int   ("Anomalies", anomaly4, n4);

    /* With tighter threshold, 15.0 should still be flagged */
    printf("  Expected: Index 4 (15.0) flagged as anomaly\n");

    if (status == 0 && anomaly4[4] == 1) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* ===================================================================== */
    /* Test 5: Edge case - n = 1 (cannot compute variance)                  */
    /* ===================================================================== */
    printf("Test 5: Edge case (n = 1, no variance)\n");
    double x5[] = {10.0};
    int anomaly5[1];
    int n5 = 1;
    double threshold5 = 3.0;

    hpcs_detect_anomalies(x5, n5, threshold5, anomaly5, &status);
    printf("  Status: %d\n", status);
    print_array_double("Data     ", x5, n5);
    print_array_int   ("Anomalies", anomaly5, n5);
    printf("  Expected: Status 0, anomaly = 0 (n < 2, cannot detect)\n");

    if (status == 0 && anomaly5[0] == 0) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* ===================================================================== */
    /* Test 6: Error handling - invalid inputs                              */
    /* ===================================================================== */
    printf("Test 6: Error handling (n <= 0, negative threshold)\n");
    double x6[] = {1.0, 2.0, 3.0};
    int anomaly6[3];

    // Test n = 0
    hpcs_detect_anomalies(x6, 0, 3.0, anomaly6, &status);
    printf("  n=0: Status=%d (expected: 1 = HPCS_ERR_INVALID_ARGS)\n", status);
    if (status != 1) {
        printf("  ✗ FAIL (n=0 should return error)\n\n");
        return 1;
    }

    // Test negative threshold
    hpcs_detect_anomalies(x6, 3, -1.0, anomaly6, &status);
    printf("  threshold=-1.0: Status=%d (expected: 1 = HPCS_ERR_INVALID_ARGS)\n", status);
    if (status != 1) {
        printf("  ✗ FAIL (negative threshold should return error)\n\n");
        return 1;
    }

    printf("  ✓ PASS\n\n");

    /* ===================================================================== */
    /* Test 7: Real-world example - sensor data with outlier                */
    /* ===================================================================== */
    printf("Test 7: Real-world example (temperature sensor with spike)\n");
    /* Normal temperature readings around 20-25°C, with one sensor error at 100°C */
    double x7[] = {22.5, 23.1, 21.8, 22.9, 23.5, 22.0, 100.0, 23.2, 22.7, 21.9,
                   22.4, 23.0, 22.6, 21.5, 22.8};
    int anomaly7[15];
    int n7 = 15;
    double threshold7 = 3.0;

    hpcs_detect_anomalies(x7, n7, threshold7, anomaly7, &status);
    printf("  Status: %d\n", status);
    printf("  Scenario: Temperature sensor readings (°C)\n");
    print_array_double("Temp (°C)", x7, n7);
    print_array_int   ("Anomalies", anomaly7, n7);
    printf("  Expected: Index 6 (100.0°C) flagged as sensor error\n");

    if (status == 0 && anomaly7[6] == 1) {
        int others_normal = 1;
        for (int i = 0; i < n7; i++) {
            if (i != 6 && anomaly7[i] != 0) {
                others_normal = 0;
                break;
            }
        }
        if (others_normal) {
            printf("  ✓ PASS\n\n");
        } else {
            printf("  ✗ FAIL (other values incorrectly flagged)\n\n");
            return 1;
        }
    } else {
        printf("  ✗ FAIL\n\n");
        return 1;
    }

    /* ===================================================================== */
    /* Test 8: Symmetric outliers (both high and low)                       */
    /* ===================================================================== */
    printf("Test 8: Symmetric outliers (high and low extremes)\n");
    /* Need more data points and more extreme outliers */
    double x8[] = {-30.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                   10.0, 10.0, 10.0, 10.0, 50.0};
    int anomaly8[15];
    int n8 = 15;
    double threshold8 = 2.5;

    hpcs_detect_anomalies(x8, n8, threshold8, anomaly8, &status);
    printf("  Status: %d\n", status);
    print_array_double("Data     ", x8, n8);
    print_array_int   ("Anomalies", anomaly8, n8);
    printf("  Expected: Indices 0 (-30.0) and 14 (50.0) flagged\n");

    if (status == 0 && anomaly8[0] == 1 && anomaly8[14] == 1) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        printf("  Debug: anomaly[0]=%d, anomaly[14]=%d\n", anomaly8[0], anomaly8[14]);
        return 1;
    }

    /* ===================================================================== */
    /* Test 9: Integration - preprocess then detect anomalies               */
    /* ===================================================================== */
    printf("Test 9: Integration test (fill missing values then detect anomalies)\n");
    /* Data with NaN that will be filled, then analyzed for anomalies */
    /* Need more data points so outlier doesn't skew statistics too much */
    double x9_raw[] = {10.0, NAN, 10.0, 10.0, NAN, 10.0, 10.0, 10.0, 100.0,
                       10.0, 10.0, NAN, 10.0};
    double x9_filled[13];
    int anomaly9[13];
    int n9 = 13;

    // Step 1: Fill forward
    hpcs_fill_forward(x9_raw, n9, x9_filled, &status);
    printf("  Step 1 - Fill forward:\n");
    print_array_double("  Raw   ", x9_raw, n9);
    print_array_double("  Filled", x9_filled, n9);

    // Step 2: Detect anomalies
    hpcs_detect_anomalies(x9_filled, n9, 3.0, anomaly9, &status);
    printf("  Step 2 - Detect anomalies:\n");
    print_array_int   ("  Anomalies", anomaly9, n9);
    printf("  Expected: Index 8 (100.0) flagged as anomaly\n");

    if (status == 0 && anomaly9[8] == 1) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
        printf("  Debug: anomaly[8]=%d\n", anomaly9[8]);
        return 1;
    }

    printf("======================================\n");
    printf("All Phase 5 tests passed! ✓\n");
    return 0;
}
