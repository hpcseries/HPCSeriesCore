/*
 * ============================================================================
 * HPCSeries Core v0.4 - GPU Infrastructure Test Suite (Phase 1)
 * ============================================================================
 *
 * Tests the GPU device detection, selection, and policy management APIs.
 *
 * Test Coverage:
 *   - Acceleration policy management (get/set)
 *   - Device count query
 *   - Device selection and validation
 *   - Error handling and edge cases
 *   - CPU-only build compatibility
 *
 * Build: cmake .. && make test_gpu_infrastructure
 * Run: ./test_gpu_infrastructure
 *
 * Expected Behavior:
 *   - CPU-only builds: All tests pass, device count = 0
 *   - GPU builds: Tests pass, device count >= 0
 *
 * Author: HPCSeries Core Team
 * Version: 0.4.0-phase1
 * Date: 2025-11-21
 *
 * ============================================================================
 */

#include "hpcs_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ============================================================================
 * Test Suite
 * ============================================================================ */

/*
 * Test 1: Basic Policy Management - Get Default Policy
 */
void test_get_default_policy() {
    TEST("Get default acceleration policy");

    int policy, status;
    hpcs_get_accel_policy(&policy, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Status is success");
    ASSERT_EQ(policy, HPCS_GPU_PREFERRED, "Default policy is GPU_PREFERRED");
}

/*
 * Test 2: Basic Policy Management - Set CPU_ONLY Policy
 */
void test_set_cpu_only_policy() {
    TEST("Set CPU_ONLY acceleration policy");

    int policy, status;
    hpcs_set_accel_policy(HPCS_CPU_ONLY, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Set policy succeeded");

    hpcs_get_accel_policy(&policy, &status);
    ASSERT_EQ(policy, HPCS_CPU_ONLY, "Policy is now CPU_ONLY");
}

/*
 * Test 3: Basic Policy Management - Set GPU_PREFERRED Policy
 */
void test_set_gpu_preferred_policy() {
    TEST("Set GPU_PREFERRED acceleration policy");

    int policy, status;
    hpcs_set_accel_policy(HPCS_GPU_PREFERRED, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Set policy succeeded");

    hpcs_get_accel_policy(&policy, &status);
    ASSERT_EQ(policy, HPCS_GPU_PREFERRED, "Policy is now GPU_PREFERRED");
}

/*
 * Test 4: Basic Policy Management - Set GPU_ONLY Policy
 */
void test_set_gpu_only_policy() {
    TEST("Set GPU_ONLY acceleration policy");

    int policy, status;
    hpcs_set_accel_policy(HPCS_GPU_ONLY, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Set policy succeeded");

    hpcs_get_accel_policy(&policy, &status);
    ASSERT_EQ(policy, HPCS_GPU_ONLY, "Policy is now GPU_ONLY");
}

/*
 * Test 5: Policy Error Handling - Invalid Policy Value
 */
void test_set_invalid_policy() {
    TEST("Set invalid acceleration policy (should fail)");

    int status;
    hpcs_set_accel_policy(999, &status);

    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Invalid policy rejected with ERR_INVALID_ARGS");

    // Verify policy unchanged
    int policy;
    hpcs_get_accel_policy(&policy, &status);
    ASSERT_EQ(policy, HPCS_GPU_ONLY, "Policy unchanged after invalid set");
}

/*
 * Test 6: Policy Error Handling - Negative Policy Value
 */
void test_set_negative_policy() {
    TEST("Set negative acceleration policy (should fail)");

    int status;
    hpcs_set_accel_policy(-1, &status);

    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Negative policy rejected");
}

/*
 * Test 7: Device Count Query - Basic Query
 */
void test_get_device_count() {
    TEST("Query GPU device count");

    int count, status;
    hpcs_get_device_count(&count, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Device count query succeeded");
    ASSERT(count >= 0, "Device count is non-negative");

    printf("  → Detected %d GPU device(s)\n", count);

    if (count == 0) {
        printf("  ℹ CPU-only build or no GPU hardware available\n");
    } else {
        printf("  ℹ GPU acceleration available\n");
    }
}

/*
 * Test 8: Device Count Query - Cached Result
 */
void test_device_count_caching() {
    TEST("Device count query caching");

    int count1, count2, status1, status2;

    hpcs_get_device_count(&count1, &status1);
    hpcs_get_device_count(&count2, &status2);

    ASSERT_EQ(count1, count2, "Device count consistent across queries");
    ASSERT_EQ(status1, HPCS_SUCCESS, "First query succeeded");
    ASSERT_EQ(status2, HPCS_SUCCESS, "Second query succeeded");
}

/*
 * Test 9: Device Selection - Get Default Device
 */
void test_get_default_device() {
    TEST("Get default device ID");

    int device_id, status;
    hpcs_get_device(&device_id, &status);

    ASSERT_EQ(status, HPCS_SUCCESS, "Get device succeeded");
    ASSERT_EQ(device_id, 0, "Default device is 0");
}

/*
 * Test 10: Device Selection - Set Device 0
 */
void test_set_device_0() {
    TEST("Set device to 0");

    int count, status;
    hpcs_get_device_count(&count, &status);

    if (count == 0) {
        printf("  ℹ Skipping: No GPU devices available\n");
        // In CPU-only mode, device 0 should still be accepted
        hpcs_set_device(0, &status);
        ASSERT_EQ(status, HPCS_SUCCESS, "Device 0 accepted in CPU-only mode");
    } else {
        hpcs_set_device(0, &status);
        ASSERT_EQ(status, HPCS_SUCCESS, "Set device 0 succeeded");

        int device_id;
        hpcs_get_device(&device_id, &status);
        ASSERT_EQ(device_id, 0, "Current device is 0");
    }
}

/*
 * Test 11: Device Selection - Invalid Negative Device ID
 */
void test_set_invalid_negative_device() {
    TEST("Set invalid negative device ID (should fail)");

    int status;
    hpcs_set_device(-1, &status);

    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Negative device ID rejected");
}

/*
 * Test 12: Device Selection - Invalid Device ID Beyond Count
 */
void test_set_invalid_device_beyond_count() {
    TEST("Set device ID beyond available count (should fail)");

    int count, status;
    hpcs_get_device_count(&count, &status);

    if (count == 0) {
        // Special case: CPU-only mode allows device_id=0
        printf("  ℹ CPU-only mode: device_id=0 is valid (special case)\n");
        hpcs_set_device(0, &status);
        ASSERT_EQ(status, HPCS_SUCCESS, "Device 0 valid in CPU-only mode");

        // Try device_id=1 which should definitely be invalid
        hpcs_set_device(1, &status);
        ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Device 1 invalid in CPU-only mode");
    } else {
        // Try to set device = count (which is out of range for 0-indexed)
        hpcs_set_device(count, &status);
        ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Out-of-range device ID rejected");
    }
}

/*
 * Test 13: Device Selection - Large Invalid Device ID
 */
void test_set_large_invalid_device() {
    TEST("Set large invalid device ID (should fail)");

    int status;
    hpcs_set_device(9999, &status);

    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Large invalid device ID rejected");
}

/*
 * Test 14: Multi-GPU Support - Test Multiple Devices (if available)
 */
void test_multiple_devices() {
    TEST("Multiple GPU device support");

    int count, status;
    hpcs_get_device_count(&count, &status);

    if (count < 2) {
        printf("  ℹ Skipping: Need at least 2 GPUs (found %d)\n", count);
        return;
    }

    // Try setting device 1
    hpcs_set_device(1, &status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Set device 1 succeeded");

    int device_id;
    hpcs_get_device(&device_id, &status);
    ASSERT_EQ(device_id, 1, "Current device is 1");

    // Switch back to device 0
    hpcs_set_device(0, &status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Set device 0 succeeded");

    hpcs_get_device(&device_id, &status);
    ASSERT_EQ(device_id, 0, "Current device is 0");
}

/*
 * Test 15: CPU-Only Mode - Device 0 Only
 */
void test_cpu_only_device_constraint() {
    TEST("CPU-only mode: Only device 0 valid");

    int count, status;
    hpcs_get_device_count(&count, &status);

    if (count > 0) {
        printf("  ℹ Skipping: GPU devices available (count=%d)\n", count);
        return;
    }

    // In CPU-only mode, only device 0 should be valid
    hpcs_set_device(0, &status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Device 0 valid in CPU-only mode");

    hpcs_set_device(1, &status);
    ASSERT_EQ(status, HPCS_ERR_INVALID_ARGS, "Device 1 invalid in CPU-only mode");
}

/*
 * Test 16: Policy and Device Interaction
 */
void test_policy_device_interaction() {
    TEST("Policy and device selection interaction");

    int status, policy, device_id, count;

    // Set CPU_ONLY policy
    hpcs_set_accel_policy(HPCS_CPU_ONLY, &status);
    ASSERT_EQ(status, HPCS_SUCCESS, "Set CPU_ONLY policy");

    // Device selection should still work
    hpcs_get_device_count(&count, &status);
    if (count > 0) {
        hpcs_set_device(0, &status);
        ASSERT_EQ(status, HPCS_SUCCESS, "Device selection works with CPU_ONLY policy");
    }

    // Verify policy unchanged
    hpcs_get_accel_policy(&policy, &status);
    ASSERT_EQ(policy, HPCS_CPU_ONLY, "Policy unchanged after device selection");

    // Reset to default policy
    hpcs_set_accel_policy(HPCS_GPU_PREFERRED, &status);
}

/*
 * Test 17: Stress Test - Rapid Policy Changes
 */
void test_rapid_policy_changes() {
    TEST("Stress test: Rapid policy changes");

    int status;
    int policies[] = {HPCS_CPU_ONLY, HPCS_GPU_PREFERRED, HPCS_GPU_ONLY};

    for (int i = 0; i < 1000; i++) {
        int policy = policies[i % 3];
        hpcs_set_accel_policy(policy, &status);
        ASSERT(status == HPCS_SUCCESS, "All rapid policy changes succeeded");
        if (status != HPCS_SUCCESS) break;
    }

    printf("  → Completed 1000 policy changes\n");
}

/*
 * Test 18: Stress Test - Rapid Device Queries
 */
void test_rapid_device_queries() {
    TEST("Stress test: Rapid device count queries");

    int status, count, first_count;

    hpcs_get_device_count(&first_count, &status);

    for (int i = 0; i < 1000; i++) {
        hpcs_get_device_count(&count, &status);
        ASSERT(status == HPCS_SUCCESS && count == first_count,
               "All rapid queries succeeded with consistent count");
        if (status != HPCS_SUCCESS || count != first_count) break;
    }

    printf("  → Completed 1000 device queries\n");
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main() {
    printf("============================================================================\n");
    printf("HPCSeries Core v0.4 - GPU Infrastructure Test Suite (Phase 1)\n");
    printf("============================================================================\n");

    // Policy Management Tests
    test_get_default_policy();
    test_set_cpu_only_policy();
    test_set_gpu_preferred_policy();
    test_set_gpu_only_policy();
    test_set_invalid_policy();
    test_set_negative_policy();

    // Device Detection Tests
    test_get_device_count();
    test_device_count_caching();

    // Device Selection Tests
    test_get_default_device();
    test_set_device_0();
    test_set_invalid_negative_device();
    test_set_invalid_device_beyond_count();
    test_set_large_invalid_device();

    // Multi-GPU and Mode Tests
    test_multiple_devices();
    test_cpu_only_device_constraint();
    test_policy_device_interaction();

    // Stress Tests
    test_rapid_policy_changes();
    test_rapid_device_queries();

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
