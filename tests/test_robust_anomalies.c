/*
 * Test Suite for Robust Anomaly Detection Functions (v0.3+)
 *
 * Tests for:
 * - hpcs_detect_anomalies_robust
 * - hpcs_remove_outliers_iterative
 *
 * These functions address Phase 5 recommendations for robust outlier detection.
 */

#include "hpcs_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define TOLERANCE 1e-9
#define TEST_PASS "\033[32m✓ PASS\033[0m"
#define TEST_FAIL "\033[31m✗ FAIL\033[0m"

static int test_count = 0;
static int pass_count = 0;

void assert_int(int actual, int expected, const char *test_name) {
    test_count++;
    if (actual == expected) {
        printf("%s: %s (expected %d, got %d)\n", TEST_PASS, test_name, expected, actual);
        pass_count++;
    } else {
        printf("%s: %s (expected %d, got %d)\n", TEST_FAIL, test_name, expected, actual);
    }
}

void assert_double(double actual, double expected, const char *test_name) {
    test_count++;
    if (fabs(actual - expected) < TOLERANCE) {
        printf("%s: %s (expected %.6f, got %.6f)\n", TEST_PASS, test_name, expected, actual);
        pass_count++;
    } else {
        printf("%s: %s (expected %.6f, got %.6f)\n", TEST_FAIL, test_name, expected, actual);
    }
}

void assert_status(int status, int expected, const char *test_name) {
    test_count++;
    if (status == expected) {
        printf("%s: %s (status %d)\n", TEST_PASS, test_name, status);
        pass_count++;
    } else {
        printf("%s: %s (expected status %d, got %d)\n", TEST_FAIL, test_name, expected, status);
    }
}

/* ============================================================================
 * Test 1: Basic Robust Anomaly Detection
 * ============================================================================ */
void test_basic_robust_detection() {
    printf("\n--- Test 1: Basic Robust Anomaly Detection ---\n");

    // Data with clear outliers
    double data[] = {10.0, 12.0, 11.0, 13.0, 100.0, 12.0, 11.5, -50.0, 12.5};
    int anomaly[9];
    int status;

    hpcs_detect_anomalies_robust(data, 9, 3.0, anomaly, &status);

    assert_status(status, HPCS_SUCCESS, "Status is success");
    assert_int(anomaly[0], 0, "Normal value (10.0) not flagged");
    assert_int(anomaly[1], 0, "Normal value (12.0) not flagged");
    assert_int(anomaly[4], 1, "Outlier (100.0) flagged");
    assert_int(anomaly[7], 1, "Outlier (-50.0) flagged");
}

/* ============================================================================
 * Test 2: Comparison with Classical Method (Masking Problem)
 * ============================================================================ */
void test_masking_resistance() {
    printf("\n--- Test 2: Robust Method Avoids Masking Problem ---\n");

    // Small dataset with one extreme outlier
    // Classical method would fail here because outlier affects mean/std
    double data[] = {10.0, 12.0, 13.0, 11.0, 14.0, 12.0, 100.0};
    int anomaly[7];
    int status;

    hpcs_detect_anomalies_robust(data, 7, 3.0, anomaly, &status);

    assert_status(status, HPCS_SUCCESS, "Status is success");

    // Robust method should detect the outlier
    assert_int(anomaly[6], 1, "Outlier (100.0) detected despite small sample");

    // Normal values should not be flagged
    for (int i = 0; i < 6; i++) {
        char msg[100];
        sprintf(msg, "Normal value at index %d not flagged", i);
        assert_int(anomaly[i], 0, msg);
    }
}

/* ============================================================================
 * Test 3: No Anomalies in Normal Distribution
 * ============================================================================ */
void test_no_anomalies() {
    printf("\n--- Test 3: No Anomalies in Normal Distribution ---\n");

    // Normal distribution centered at 50
    double data[] = {48.0, 49.0, 50.0, 51.0, 52.0, 49.5, 50.5, 48.5, 51.5};
    int anomaly[9];
    int status;

    hpcs_detect_anomalies_robust(data, 9, 3.0, anomaly, &status);

    assert_status(status, HPCS_SUCCESS, "Status is success");

    int total_anomalies = 0;
    for (int i = 0; i < 9; i++) {
        total_anomalies += anomaly[i];
    }
    assert_int(total_anomalies, 0, "No anomalies in normal distribution");
}

/* ============================================================================
 * Test 4: Constant Array (MAD = 0)
 * ============================================================================ */
void test_constant_array() {
    printf("\n--- Test 4: Constant Array (MAD = 0) ---\n");

    // All values are the same
    double data[] = {42.0, 42.0, 42.0, 42.0, 42.0};
    int anomaly[5];
    int status;

    hpcs_detect_anomalies_robust(data, 5, 3.0, anomaly, &status);

    assert_status(status, HPCS_SUCCESS, "Status is success for constant array");

    // All values should be non-anomalies (MAD=0 means no variation)
    for (int i = 0; i < 5; i++) {
        assert_int(anomaly[i], 0, "Constant value not flagged");
    }
}

/* ============================================================================
 * Test 5: Different Thresholds
 * ============================================================================ */
void test_different_thresholds() {
    printf("\n--- Test 5: Different Thresholds (2-sigma vs 3-sigma) ---\n");

    double data[] = {10.0, 11.0, 12.0, 13.0, 14.0, 25.0};  // 25.0 is moderate outlier
    int anomaly_2sigma[6], anomaly_3sigma[6];
    int status;

    // Test with 2-sigma threshold (more sensitive)
    hpcs_detect_anomalies_robust(data, 6, 2.0, anomaly_2sigma, &status);
    assert_status(status, HPCS_SUCCESS, "2-sigma test succeeds");

    // Test with 3-sigma threshold (less sensitive)
    hpcs_detect_anomalies_robust(data, 6, 3.0, anomaly_3sigma, &status);
    assert_status(status, HPCS_SUCCESS, "3-sigma test succeeds");

    // 2-sigma should detect more anomalies than 3-sigma
    int count_2sigma = 0, count_3sigma = 0;
    for (int i = 0; i < 6; i++) {
        count_2sigma += anomaly_2sigma[i];
        count_3sigma += anomaly_3sigma[i];
    }

    printf("2-sigma detected %d anomalies, 3-sigma detected %d anomalies\n",
           count_2sigma, count_3sigma);

    test_count++;
    if (count_2sigma >= count_3sigma) {
        printf("%s: 2-sigma more sensitive than 3-sigma\n", TEST_PASS);
        pass_count++;
    } else {
        printf("%s: 2-sigma should detect >= anomalies than 3-sigma\n", TEST_FAIL);
    }
}

/* ============================================================================
 * Test 6: Edge Cases
 * ============================================================================ */
void test_edge_cases() {
    printf("\n--- Test 6: Edge Cases ---\n");

    // Test n=1
    double data1[1] = {42.0};
    int anomaly1[1];
    int status;

    hpcs_detect_anomalies_robust(data1, 1, 3.0, anomaly1, &status);
    assert_status(status, HPCS_SUCCESS, "n=1 handled gracefully");
    assert_int(anomaly1[0], 0, "Single value not flagged as anomaly");

    // Test n=0 (invalid)
    hpcs_detect_anomalies_robust(data1, 0, 3.0, anomaly1, &status);
    assert_status(status, HPCS_ERR_INVALID_ARGS, "n=0 returns error");

    // Test negative threshold (invalid)
    double data2[] = {10.0, 20.0};
    int anomaly2[2];
    hpcs_detect_anomalies_robust(data2, 2, -1.0, anomaly2, &status);
    assert_status(status, HPCS_ERR_INVALID_ARGS, "Negative threshold returns error");
}

/* ============================================================================
 * Test 7: Sensor Data Example
 * ============================================================================ */
void test_sensor_data() {
    printf("\n--- Test 7: Real-World Sensor Data ---\n");

    // Temperature sensor with spike
    double temps[] = {22.5, 23.1, 21.8, 22.9, 100.0, 23.2, 22.7, 22.3};
    int anomaly[8];
    int status;

    hpcs_detect_anomalies_robust(temps, 8, 3.0, anomaly, &status);

    assert_status(status, HPCS_SUCCESS, "Sensor data analysis succeeds");
    assert_int(anomaly[4], 1, "Sensor spike (100.0°C) detected");

    // Normal readings should not be flagged
    int normal_count = 0;
    for (int i = 0; i < 8; i++) {
        if (i != 4 && anomaly[i] == 0) normal_count++;
    }
    assert_int(normal_count, 7, "All normal sensor readings preserved");
}

/* ============================================================================
 * Test 8: Basic Iterative Outlier Removal
 * ============================================================================ */
void test_basic_iterative_removal() {
    printf("\n--- Test 8: Basic Iterative Outlier Removal ---\n");

    double data[] = {10.0, 12.0, 11.0, 100.0, 13.0, -50.0, 12.5};
    double cleaned[7];
    int n_clean, iterations, status;

    hpcs_remove_outliers_iterative(data, 7, 3.0, 10, cleaned, &n_clean, &iterations, &status);

    assert_status(status, HPCS_SUCCESS, "Iterative removal succeeds");
    assert_int(n_clean, 5, "Cleaned data has 5 values (2 outliers removed)");

    printf("Iterations performed: %d\n", iterations);

    // Check that cleaned data doesn't contain outliers
    int has_100 = 0, has_minus50 = 0;
    for (int i = 0; i < n_clean; i++) {
        if (fabs(cleaned[i] - 100.0) < TOLERANCE) has_100 = 1;
        if (fabs(cleaned[i] + 50.0) < TOLERANCE) has_minus50 = 1;
    }
    assert_int(has_100, 0, "Outlier 100.0 removed from cleaned data");
    assert_int(has_minus50, 0, "Outlier -50.0 removed from cleaned data");
}

/* ============================================================================
 * Test 9: Iterative Removal Convergence
 * ============================================================================ */
void test_iterative_convergence() {
    printf("\n--- Test 9: Iterative Removal Convergence ---\n");

    // Data with nested outliers (removing first outlier reveals second)
    double data[] = {10.0, 11.0, 12.0, 13.0, 14.0, 30.0, 100.0};
    double cleaned[7];
    int n_clean, iterations, status;

    hpcs_remove_outliers_iterative(data, 7, 2.5, 10, cleaned, &n_clean, &iterations, &status);

    assert_status(status, HPCS_SUCCESS, "Convergence test succeeds");

    printf("Cleaned data size: %d, iterations: %d\n", n_clean, iterations);

    test_count++;
    if (iterations > 1) {
        printf("%s: Multiple iterations performed (detected nested outliers)\n", TEST_PASS);
        pass_count++;
    } else {
        printf("%s: Expected multiple iterations for nested outliers\n", TEST_FAIL);
    }
}

/* ============================================================================
 * Test 10: Max Iterations Limit
 * ============================================================================ */
void test_max_iterations() {
    printf("\n--- Test 10: Max Iterations Limit ---\n");

    double data[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    double cleaned[5];
    int n_clean, iterations, status;

    // Use very low threshold to force many iterations
    hpcs_remove_outliers_iterative(data, 5, 0.5, 3, cleaned, &n_clean, &iterations, &status);

    assert_status(status, HPCS_SUCCESS, "Max iterations test succeeds");

    test_count++;
    if (iterations <= 3) {
        printf("%s: Iterations limited to max_iter (actual: %d)\n", TEST_PASS, iterations);
        pass_count++;
    } else {
        printf("%s: Iterations (%d) exceeded max_iter (3)\n", TEST_FAIL, iterations);
    }
}

/* ============================================================================
 * Test 11: All Data Removed
 * ============================================================================ */
void test_all_removed() {
    printf("\n--- Test 11: Edge Case - All Data Removed ---\n");

    // Highly scattered data with very strict threshold
    double data[] = {1.0, 10.0, 20.0, 30.0, 40.0};
    double cleaned[5];
    int n_clean, iterations, status;

    // Very strict threshold that removes everything
    hpcs_remove_outliers_iterative(data, 5, 0.01, 10, cleaned, &n_clean, &iterations, &status);

    assert_status(status, HPCS_SUCCESS, "All-removed case succeeds");

    test_count++;
    if (n_clean <= 1) {
        printf("%s: Correctly handles case where most/all data removed (n_clean=%d)\n",
               TEST_PASS, n_clean);
        pass_count++;
    } else {
        printf("%s: Expected very few cleaned values with strict threshold\n", TEST_FAIL);
    }
}

/* ============================================================================
 * Test 12: Clean Data (No Removal Needed)
 * ============================================================================ */
void test_clean_data_no_removal() {
    printf("\n--- Test 12: Clean Data (No Removal Needed) ---\n");

    // Perfectly normal data
    double data[] = {10.0, 11.0, 12.0, 13.0, 14.0};
    double cleaned[5];
    int n_clean, iterations, status;

    hpcs_remove_outliers_iterative(data, 5, 3.0, 10, cleaned, &n_clean, &iterations, &status);

    assert_status(status, HPCS_SUCCESS, "Clean data test succeeds");
    assert_int(n_clean, 5, "All data preserved (no outliers)");
    assert_int(iterations, 1, "Converged in 1 iteration (no outliers found)");

    // Check all values preserved
    for (int i = 0; i < 5; i++) {
        assert_double(cleaned[i], data[i], "Data value preserved");
    }
}

/* ============================================================================
 * Test 13: Invalid Arguments for Iterative Removal
 * ============================================================================ */
void test_iterative_invalid_args() {
    printf("\n--- Test 13: Invalid Arguments for Iterative Removal ---\n");

    double data[] = {10.0, 20.0};
    double cleaned[2];
    int n_clean, iterations, status;

    // Test n <= 0
    hpcs_remove_outliers_iterative(data, 0, 3.0, 10, cleaned, &n_clean, &iterations, &status);
    assert_status(status, HPCS_ERR_INVALID_ARGS, "n=0 returns error");

    // Test negative threshold
    hpcs_remove_outliers_iterative(data, 2, -1.0, 10, cleaned, &n_clean, &iterations, &status);
    assert_status(status, HPCS_ERR_INVALID_ARGS, "Negative threshold returns error");

    // Test max_iter <= 0
    hpcs_remove_outliers_iterative(data, 2, 3.0, 0, cleaned, &n_clean, &iterations, &status);
    assert_status(status, HPCS_ERR_INVALID_ARGS, "max_iter=0 returns error");
}

/* ============================================================================
 * Test 14: Comparison of Methods
 * ============================================================================ */
void test_comparison_classical_vs_robust() {
    printf("\n--- Test 14: Comparison - Classical vs Robust Detection ---\n");

    // Data where classical method would fail (outliers mask themselves)
    double data[] = {10.0, 12.0, 11.0, 13.0, 100.0, 200.0, 12.5};
    int anomaly_robust[7];
    int status;

    // Note: We only have robust method implemented, but this demonstrates it works
    hpcs_detect_anomalies_robust(data, 7, 3.0, anomaly_robust, &status);

    assert_status(status, HPCS_SUCCESS, "Robust detection succeeds");

    int outlier_count = 0;
    for (int i = 0; i < 7; i++) {
        outlier_count += anomaly_robust[i];
    }

    test_count++;
    if (outlier_count >= 2) {
        printf("%s: Robust method detects multiple outliers (count=%d)\n",
               TEST_PASS, outlier_count);
        pass_count++;
    } else {
        printf("%s: Should detect at least 2 outliers (100.0 and 200.0)\n", TEST_FAIL);
    }
}

/* ============================================================================
 * Test 15: Basic Rolling Anomaly Detection
 * ============================================================================ */
void test_basic_rolling_detection() {
    printf("\n--- Test 15: Basic Rolling Anomaly Detection ---\n");

    // Time series with spike
    double data[] = {10.0, 11.0, 12.0, 100.0, 11.5, 12.5, 10.5, 11.0};
    int anomaly[8];
    int status;
    int window = 3;

    hpcs_rolling_anomalies(data, 8, window, 3.0, anomaly, &status);

    assert_status(status, HPCS_SUCCESS, "Rolling detection succeeds");

    // First window-1 elements should be 0 (no detection)
    assert_int(anomaly[0], 0, "Before window: no detection");
    assert_int(anomaly[1], 0, "Before window: no detection");

    // Spike should be detected
    assert_int(anomaly[3], 1, "Spike (100.0) detected");

    printf("Anomaly flags: [");
    for (int i = 0; i < 8; i++) {
        printf("%d%s", anomaly[i], (i < 7) ? ", " : "");
    }
    printf("]\n");
}

/* ============================================================================
 * Test 16: Rolling Detection on Trend
 * ============================================================================ */
void test_rolling_on_trend() {
    printf("\n--- Test 16: Rolling Detection Adapts to Trend ---\n");

    // Upward trend with one outlier
    double data[] = {10.0, 20.0, 30.0, 40.0, 100.0, 50.0, 60.0, 70.0};
    int anomaly[8];
    int status;
    int window = 3;

    hpcs_rolling_anomalies(data, 8, window, 3.0, anomaly, &status);

    assert_status(status, HPCS_SUCCESS, "Trend detection succeeds");

    // Outlier should be detected despite trend
    assert_int(anomaly[4], 1, "Outlier detected in trending data");

    // Trend values should not be flagged
    assert_int(anomaly[5], 0, "Trend value not flagged");
    assert_int(anomaly[6], 0, "Trend value not flagged");
    assert_int(anomaly[7], 0, "Trend value not flagged");
}

/* ============================================================================
 * Test 17: Rolling Detection Window Size
 * ============================================================================ */
void test_rolling_window_size() {
    printf("\n--- Test 17: Different Window Sizes ---\n");

    double data[] = {10.0, 11.0, 12.0, 11.5, 50.0, 12.5, 11.0, 10.5};
    int anomaly_small[8], anomaly_large[8];
    int status;

    // Small window (more sensitive to local changes)
    hpcs_rolling_anomalies(data, 8, 3, 3.0, anomaly_small, &status);
    assert_status(status, HPCS_SUCCESS, "Small window detection succeeds");

    // Large window (less sensitive)
    hpcs_rolling_anomalies(data, 8, 5, 3.0, anomaly_large, &status);
    assert_status(status, HPCS_SUCCESS, "Large window detection succeeds");

    // Both should detect the spike at index 4
    assert_int(anomaly_small[4], 1, "Small window detects spike");
    assert_int(anomaly_large[4], 1, "Large window detects spike");

    printf("Small window (3): detected spike\n");
    printf("Large window (5): detected spike\n");
}

/* ============================================================================
 * Test 18: Rolling Detection Edge Cases
 * ============================================================================ */
void test_rolling_edge_cases() {
    printf("\n--- Test 18: Rolling Detection Edge Cases ---\n");

    double data[] = {10.0, 20.0};
    int anomaly[2];
    int status;

    // Window = n (entire array)
    hpcs_rolling_anomalies(data, 2, 2, 3.0, anomaly, &status);
    assert_status(status, HPCS_SUCCESS, "window=n works");

    // Window > n (invalid)
    hpcs_rolling_anomalies(data, 2, 3, 3.0, anomaly, &status);
    assert_status(status, HPCS_ERR_INVALID_ARGS, "window>n returns error");

    // Negative threshold (invalid)
    hpcs_rolling_anomalies(data, 2, 2, -1.0, anomaly, &status);
    assert_status(status, HPCS_ERR_INVALID_ARGS, "Negative threshold returns error");
}

/* ============================================================================
 * Test 19: Rolling vs Global Detection
 * ============================================================================ */
void test_rolling_vs_global() {
    printf("\n--- Test 19: Rolling vs Global Detection Comparison ---\n");

    // Non-stationary data: first half low, second half high
    double data[] = {10.0, 11.0, 12.0, 11.5, 50.0, 51.0, 52.0, 51.5};
    int anomaly_rolling[8], anomaly_global[8];
    int status;

    // Rolling detection (adapts to local level)
    hpcs_rolling_anomalies(data, 8, 3, 3.0, anomaly_rolling, &status);
    assert_status(status, HPCS_SUCCESS, "Rolling detection succeeds");

    // Global detection (uses entire dataset statistics)
    hpcs_detect_anomalies_robust(data, 8, 3.0, anomaly_global, &status);
    assert_status(status, HPCS_SUCCESS, "Global detection succeeds");

    printf("Rolling anomalies: [");
    for (int i = 0; i < 8; i++) printf("%d%s", anomaly_rolling[i], (i<7)?", ":"");
    printf("]\n");

    printf("Global anomalies:  [");
    for (int i = 0; i < 8; i++) printf("%d%s", anomaly_global[i], (i<7)?", ":"");
    printf("]\n");

    // Rolling should adapt better to level shift
    test_count++;
    if (anomaly_rolling[6] == 0 || anomaly_rolling[7] == 0) {
        printf("%s: Rolling adapts to level shift (doesn't flag new level as outlier)\n", TEST_PASS);
        pass_count++;
    } else {
        printf("%s: Rolling should adapt to level changes\n", TEST_FAIL);
    }
}

/* ============================================================================
 * Test 20: Rolling Detection with Varying Data
 * ============================================================================ */
void test_rolling_constant_window() {
    printf("\n--- Test 20: Rolling Detection with Varying Data ---\n");

    // Data with some variation followed by spike
    // Window needs sufficient variation for MAD > 0
    double data[] = {10.0, 11.0, 12.0, 11.0, 10.5, 100.0, 11.5, 12.0, 10.0, 11.0};
    int anomaly[10];
    int status;

    hpcs_rolling_anomalies(data, 10, 5, 3.0, anomaly, &status);

    assert_status(status, HPCS_SUCCESS, "Rolling detection succeeds");

    // Spike should be detected (window has variation, so MAD > 0)
    assert_int(anomaly[5], 1, "Spike detected with varied window");

    // Normal values after spike should not be flagged
    assert_int(anomaly[6], 0, "Normal value not flagged");
    assert_int(anomaly[7], 0, "Normal value not flagged");
    assert_int(anomaly[8], 0, "Normal value not flagged");
    assert_int(anomaly[9], 0, "Normal value not flagged");
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */
int main() {
    printf("========================================\n");
    printf("Robust Anomaly Detection Test Suite\n");
    printf("========================================\n");

    // Run all tests
    test_basic_robust_detection();
    test_masking_resistance();
    test_no_anomalies();
    test_constant_array();
    test_different_thresholds();
    test_edge_cases();
    test_sensor_data();
    test_basic_iterative_removal();
    test_iterative_convergence();
    test_max_iterations();
    test_all_removed();
    test_clean_data_no_removal();
    test_iterative_invalid_args();
    test_comparison_classical_vs_robust();

    // Rolling anomaly detection tests
    test_basic_rolling_detection();
    test_rolling_on_trend();
    test_rolling_window_size();
    test_rolling_edge_cases();
    test_rolling_vs_global();
    test_rolling_constant_window();

    // Summary
    printf("\n========================================\n");
    printf("Test Summary: %d/%d tests passed\n", pass_count, test_count);
    printf("========================================\n");

    if (pass_count == test_count) {
        printf("\n\033[32m✓ All tests PASSED!\033[0m\n\n");
        return 0;
    } else {
        printf("\n\033[31m✗ Some tests FAILED\033[0m\n\n");
        return 1;
    }
}
