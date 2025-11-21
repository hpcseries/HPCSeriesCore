/*
 * ============================================================================
 * HPCSeries Core v0.4 - Phase 2 Acceleration Test Suite
 * ============================================================================
 *
 * Tests the Phase 2 GPU acceleration infrastructure:
 *   - Backend initialization
 *   - Memory management (host-device transfers)
 *   - HIGH PRIORITY kernel wrappers (median, MAD, rolling_median)
 *   - Example reduction wrapper (reduce_sum)
 *
 * Test Coverage:
 *   - Backend initialization and idempotence
 *   - Memory copy operations (to/from device)
 *   - Kernel wrapper correctness (CPU fallback verification)
 *   - Error handling and edge cases
 *   - Integration with Phase 1A policy management
 *
 * Build: cmake .. && make test_phase2_accel
 * Run: ./test_phase2_accel
 *
 * Expected Behavior:
 *   - CPU-only builds: All tests pass using CPU fallback
 *   - GPU builds: Tests pass using GPU acceleration (future)
 *
 * Author: HPCSeries Core Team
 * Version: 0.4.0-phase2
 * Date: 2025-11-21
 *
 * ============================================================================
 */

#include "hpcs_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Test Framework
 * ============================================================================ */

static int test_count = 0;
static int test_pass = 0;
static int test_fail = 0;

#define TEST(name) \
    do { \
        printf("\n[TEST %d] %s\n", ++test_count, name); \
    } while(0)

#define ASSERT(condition, message) \
    do { \
        if (condition) { \
            printf("  ✓ %s\n", message); \
            test_pass++; \
        } else { \
            printf("  ✗ FAIL: %s\n", message); \
            test_fail++; \
        } \
    } while(0)

#define ASSERT_EQ(actual, expected, message) \
    do { \
        if ((actual) == (expected)) { \
            printf("  ✓ %s (got %d)\n", message, actual); \
            test_pass++; \
        } else { \
            printf("  ✗ FAIL: %s (expected %d, got %d)\n", \
                   message, expected, actual); \
            test_fail++; \
        } \
    } while(0)

#define ASSERT_NEAR(actual, expected, tolerance, message) \
    do { \
        double diff = fabs((actual) - (expected)); \
        if (diff < (tolerance)) { \
            printf("  ✓ %s (got %.6f, expected %.6f)\n", message, (double)(actual), (double)(expected)); \
            test_pass++; \
        } else { \
            printf("  ✗ FAIL: %s (expected %.6f, got %.6f, diff=%.6f)\n", \
                   message, (double)(expected), (double)(actual), diff); \
            test_fail++; \
        } \
    } while(0)

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

// Comparison function for qsort (for reference median calculation)
static int compare_doubles(const void* a, const void* b) {
    double diff = (*(double*)a - *(double*)b);
    return (diff > 0) - (diff < 0);
}

// Reference median calculation
static double reference_median(double* data, int n) {
    double* temp = (double*)malloc(n * sizeof(double));
    memcpy(temp, data, n * sizeof(double));
    qsort(temp, n, sizeof(double), compare_doubles);

    double result;
    if (n % 2 == 0) {
        result = (temp[n/2 - 1] + temp[n/2]) / 2.0;
    } else {
        result = temp[n/2];
    }

    free(temp);
    return result;
}

// Reference MAD calculation
static double reference_mad(double* data, int n) {
    double median = reference_median(data, n);
    double* abs_dev = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        abs_dev[i] = fabs(data[i] - median);
    }

    double result = reference_median(abs_dev, n);
    free(abs_dev);
    return result;
}

/* ============================================================================
 * Test Suite: Backend Initialization
 * ============================================================================ */

/*
 * Test 1: Basic Backend Initialization
 */
void test_backend_init() {
    TEST("Backend initialization");

    int status;
    hpcs_accel_init(&status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Backend initialization succeeded");
}

/*
 * Test 2: Idempotent Initialization
 */
void test_backend_init_idempotent() {
    TEST("Idempotent backend initialization");

    int status1, status2, status3;

    hpcs_accel_init(&status1);
    hpcs_accel_init(&status2);
    hpcs_accel_init(&status3);

    ASSERT_EQ(status1, HPCS_SUCCESS, "First init succeeded");
    ASSERT_EQ(status2, HPCS_SUCCESS, "Second init succeeded");
    ASSERT_EQ(status3, HPCS_SUCCESS, "Third init succeeded");

    printf("  → Multiple init calls are safe (idempotent)\n");
}

/* ============================================================================
 * Test Suite: Memory Management
 * ============================================================================ */

/*
 * Test 3: Copy to Device - Valid Data
 */
void test_copy_to_device_valid() {
    TEST("Copy to device with valid data");

    const int n = 100;
    double* host_data = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        host_data[i] = (double)i;
    }

    int status;
    void* device_ptr = NULL;

    hpcs_accel_copy_to_device(host_data, n, &device_ptr, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Copy to device succeeded");
    ASSERT(device_ptr != NULL, "Device pointer is not NULL");

    // In CPU-only mode, device_ptr should equal host_data
    printf("  ℹ CPU-only mode: device_ptr = host_ptr (no actual copy)\n");

    free(host_data);
}

/*
 * Test 4: Copy to Device - Invalid Arguments
 */
void test_copy_to_device_invalid() {
    TEST("Copy to device with invalid arguments");

    int status;
    void* device_ptr;
    double data[10];

    // Test with n <= 0
    hpcs_accel_copy_to_device(data, 0, &device_ptr, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=0");

    hpcs_accel_copy_to_device(data, -1, &device_ptr, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=-1");

    // Test with NULL host pointer
    hpcs_accel_copy_to_device(NULL, 10, &device_ptr, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected NULL host_ptr");
}

/*
 * Test 5: Copy from Device - Valid Data
 */
void test_copy_from_device_valid() {
    TEST("Copy from device with valid data");

    const int n = 50;
    double* device_data = (double*)malloc(n * sizeof(double));
    double* host_data = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        device_data[i] = (double)(i * 2);
    }

    int status;
    hpcs_accel_copy_from_device(device_data, n, host_data, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Copy from device succeeded");

    printf("  ℹ CPU-only mode: no-op (data already on host)\n");

    free(device_data);
    free(host_data);
}

/*
 * Test 6: Copy from Device - Invalid Arguments
 */
void test_copy_from_device_invalid() {
    TEST("Copy from device with invalid arguments");

    int status;
    double data[10];

    // Test with n <= 0
    hpcs_accel_copy_from_device(data, 0, data, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=0");

    hpcs_accel_copy_from_device(data, -1, data, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=-1");

    // Test with NULL pointers
    hpcs_accel_copy_from_device(NULL, 10, data, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected NULL device_ptr");

    hpcs_accel_copy_from_device(data, 10, NULL, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected NULL host_ptr");
}

/* ============================================================================
 * Test Suite: HIGH PRIORITY Kernel Wrappers
 * ============================================================================ */

/*
 * Test 7: Median Kernel - Simple Odd Count
 */
void test_accel_median_odd() {
    TEST("Median kernel with odd count");

    double data[] = {5.0, 1.0, 3.0, 2.0, 4.0};
    int n = 5;

    double median_val;
    int status;

    hpcs_accel_median(data, n, &median_val, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Median calculation succeeded");
    ASSERT_NEAR(median_val, 3.0, 1e-9, "Median is 3.0");
}

/*
 * Test 8: Median Kernel - Simple Even Count
 */
void test_accel_median_even() {
    TEST("Median kernel with even count");

    double data[] = {1.0, 2.0, 3.0, 4.0};
    int n = 4;

    double median_val;
    int status;

    hpcs_accel_median(data, n, &median_val, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Median calculation succeeded");
    ASSERT_NEAR(median_val, 2.5, 1e-9, "Median is 2.5");
}

/*
 * Test 9: Median Kernel - Large Dataset
 */
void test_accel_median_large() {
    TEST("Median kernel with large dataset (1000 elements)");

    const int n = 1000;
    double* data = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        data[i] = (double)(n - i);  // Descending order
    }

    double median_val;
    int status;

    hpcs_accel_median(data, n, &median_val, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Large median calculation succeeded");
    ASSERT_NEAR(median_val, 500.5, 1e-9, "Median is 500.5");

    free(data);
}

/*
 * Test 10: Median Kernel - Invalid Arguments
 */
void test_accel_median_invalid() {
    TEST("Median kernel with invalid arguments");

    double data[10];
    double result;
    int status;

    // Test with n <= 0
    hpcs_accel_median(data, 0, &result, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=0");

    hpcs_accel_median(data, -1, &result, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=-1");

    // Test with NULL pointer
    hpcs_accel_median(NULL, 10, &result, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected NULL device_ptr");
}

/*
 * Test 11: MAD Kernel - Known Values
 */
void test_accel_mad() {
    TEST("MAD kernel with known values");

    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int n = 5;

    double mad_val;
    int status;

    hpcs_accel_mad(data, n, &mad_val, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "MAD calculation succeeded");

    // Reference MAD calculation
    double expected = reference_mad(data, n);
    ASSERT_NEAR(mad_val, expected, 1e-9, "MAD matches reference");

    printf("  → MAD = %.6f\n", mad_val);
}

/*
 * Test 12: MAD Kernel - Large Dataset
 */
void test_accel_mad_large() {
    TEST("MAD kernel with large dataset (500 elements)");

    const int n = 500;
    double* data = (double*)malloc(n * sizeof(double));

    // Create data with known distribution
    for (int i = 0; i < n; i++) {
        data[i] = (double)i / 10.0;
    }

    double mad_val;
    int status;

    hpcs_accel_mad(data, n, &mad_val, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Large MAD calculation succeeded");

    double expected = reference_mad(data, n);
    ASSERT_NEAR(mad_val, expected, 1e-6, "MAD matches reference");

    free(data);
}

/*
 * Test 13: MAD Kernel - Invalid Arguments
 */
void test_accel_mad_invalid() {
    TEST("MAD kernel with invalid arguments");

    double data[10];
    double result;
    int status;

    hpcs_accel_mad(data, 0, &result, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=0");

    hpcs_accel_mad(NULL, 10, &result, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected NULL device_ptr");
}

/*
 * Test 14: Rolling Median Kernel - Small Window
 */
void test_accel_rolling_median() {
    TEST("Rolling median kernel with small window");

    double data[] = {1.0, 5.0, 2.0, 8.0, 3.0, 7.0};
    int n = 6;
    int window = 3;

    void* device_output = NULL;
    int status;

    hpcs_accel_rolling_median(data, n, window, &device_output, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Rolling median calculation succeeded");
    ASSERT(device_output != NULL, "Output pointer is not NULL");

    printf("  → Rolling median with window=%d computed\n", window);
}

/*
 * Test 15: Rolling Median Kernel - Invalid Arguments
 */
void test_accel_rolling_median_invalid() {
    TEST("Rolling median kernel with invalid arguments");

    double data[10];
    void* output;
    int status;

    hpcs_accel_rolling_median(data, 0, 3, &output, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=0");

    hpcs_accel_rolling_median(data, 10, 0, &output, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected window=0");

    hpcs_accel_rolling_median(NULL, 10, 3, &output, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected NULL device_ptr");
}

/* ============================================================================
 * Test Suite: Example Reduction Wrapper
 * ============================================================================ */

/*
 * Test 16: Reduce Sum - Simple Array
 */
void test_accel_reduce_sum() {
    TEST("Reduce sum wrapper with simple array");

    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int n = 5;

    double result;
    int status;

    hpcs_accel_reduce_sum(data, n, &result, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Reduce sum succeeded");
    ASSERT_NEAR(result, 15.0, 1e-9, "Sum is 15.0");
}

/*
 * Test 17: Reduce Sum - Large Array
 */
void test_accel_reduce_sum_large() {
    TEST("Reduce sum wrapper with large array (10000 elements)");

    const int n = 10000;
    double* data = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        data[i] = 1.0;
    }

    double result;
    int status;

    hpcs_accel_reduce_sum(data, n, &result, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Large reduce sum succeeded");
    ASSERT_NEAR(result, 10000.0, 1e-6, "Sum is 10000.0");

    free(data);
}

/*
 * Test 18: Reduce Sum - Invalid Arguments
 */
void test_accel_reduce_sum_invalid() {
    TEST("Reduce sum wrapper with invalid arguments");

    double data[10];
    double result;
    int status;

    hpcs_accel_reduce_sum(data, 0, &result, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected n=0");

    hpcs_accel_reduce_sum(NULL, 10, &result, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Rejected NULL device_ptr");
}

/* ============================================================================
 * Test Suite: Integration Tests
 * ============================================================================ */

/*
 * Test 19: Full Workflow - Init, Copy, Compute, Copy Back
 */
void test_full_workflow() {
    TEST("Full workflow: init → copy → compute → copy back");

    int status;

    // 1. Initialize backend
    hpcs_accel_init(&status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Backend init succeeded");

    // 2. Prepare data
    const int n = 100;
    double* host_data = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        host_data[i] = (double)(i + 1);
    }

    // 3. Copy to device
    void* device_ptr = NULL;
    hpcs_accel_copy_to_device(host_data, n, &device_ptr, &status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Copy to device succeeded");

    // 4. Compute median on device
    double median_val;
    hpcs_accel_median(device_ptr, n, &median_val, &status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Median computation succeeded");
    ASSERT_NEAR(median_val, 50.5, 1e-9, "Median is 50.5");

    // 5. Copy result back (in this case, median is already on host)
    printf("  → Full workflow completed successfully\n");

    free(host_data);
}

/*
 * Test 20: Policy Interaction - CPU_ONLY Mode
 */
void test_policy_cpu_only_interaction() {
    TEST("Policy interaction: CPU_ONLY mode with acceleration");

    int status;

    // Set CPU_ONLY policy
    hpcs_set_accel_policy(HPCS_CPU_ONLY, &status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Set CPU_ONLY policy");

    // Acceleration functions should still work (falling back to CPU)
    double data[] = {5.0, 3.0, 1.0, 4.0, 2.0};
    double median_val;

    hpcs_accel_median(data, 5, &median_val, &status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Median works in CPU_ONLY mode");
    ASSERT_NEAR(median_val, 3.0, 1e-9, "Median is correct");

    // Reset to default policy
    hpcs_set_accel_policy(HPCS_GPU_PREFERRED, &status);

    printf("  → Acceleration works with CPU_ONLY policy (CPU fallback)\n");
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main() {
    printf("============================================================================\n");
    printf("HPCSeries Core v0.4 - Phase 2 Acceleration Test Suite\n");
    printf("============================================================================\n");

    // Backend Initialization Tests
    test_backend_init();
    test_backend_init_idempotent();

    // Memory Management Tests
    test_copy_to_device_valid();
    test_copy_to_device_invalid();
    test_copy_from_device_valid();
    test_copy_from_device_invalid();

    // HIGH PRIORITY Kernel Wrapper Tests
    test_accel_median_odd();
    test_accel_median_even();
    test_accel_median_large();
    test_accel_median_invalid();
    test_accel_mad();
    test_accel_mad_large();
    test_accel_mad_invalid();
    test_accel_rolling_median();
    test_accel_rolling_median_invalid();

    // Example Reduction Tests
    test_accel_reduce_sum();
    test_accel_reduce_sum_large();
    test_accel_reduce_sum_invalid();

    // Integration Tests
    test_full_workflow();
    test_policy_cpu_only_interaction();

    // Summary
    printf("\n============================================================================\n");
    printf("Test Summary\n");
    printf("============================================================================\n");
    printf("Total tests: %d\n", test_count);
    printf("Assertions:  %d passed, %d failed\n", test_pass, test_fail);

    if (test_fail == 0) {
        printf("\n✓ All tests PASSED\n");
        printf("============================================================================\n");
        return 0;
    } else {
        printf("\n✗ Some tests FAILED\n");
        printf("============================================================================\n");
        return 1;
    }
}
