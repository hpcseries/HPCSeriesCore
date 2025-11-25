/**
 * Comprehensive CUDA Runtime Test
 *
 * Tests CUDA device detection and selection using the hpcs_cuda_runtime module.
 * Validates:
 *   - Backend initialization
 *   - Device count query
 *   - Device selection
 *   - Error handling
 *   - Status code consistency
 *
 * Expected behavior:
 *   - CPU-only build: 0 devices, all operations succeed
 *   - CUDA build with GPU: 1+ devices, device selection works
 *   - CUDA build without GPU: 0 devices, returns HPCS_SUCCESS with count=0
 */

#include <stdio.h>
#include <stdlib.h>

// HPCSeries status codes (must match hpcs_constants.f90)
#define HPCS_SUCCESS          0
#define HPCS_ERR_INVALID_ARGS 1
#define HPCS_ERR_NUMERIC_FAIL 2

// Forward declarations for Fortran functions
extern void hpcs_get_device_count(int* count, int* status);
extern void hpcs_set_device(int device_id, int* status);
extern void hpcs_get_device(int* device_id, int* status);
extern void hpcs_accel_init(int* status);

// Helper to print status code name
const char* status_name(int status) {
    switch(status) {
        case HPCS_SUCCESS: return "HPCS_SUCCESS";
        case HPCS_ERR_INVALID_ARGS: return "HPCS_ERR_INVALID_ARGS";
        case HPCS_ERR_NUMERIC_FAIL: return "HPCS_ERR_NUMERIC_FAIL";
        default: return "UNKNOWN";
    }
}

int main() {
    int count = -1;
    int status = -1;
    int device_id = -1;
    int test_failures = 0;

    printf("============================================================================\n");
    printf("HPCSeries CUDA Runtime Test Suite\n");
    printf("============================================================================\n");
    printf("\n");

    // Build configuration info
    printf("Build Configuration:\n");
#ifdef HPCS_USE_CUDA
    printf("  Backend: CUDA (HPCS_USE_CUDA defined)\n");
#elif defined(HPCS_USE_OPENMP_TARGET)
    printf("  Backend: OpenMP Target Offload\n");
#else
    printf("  Backend: CPU-only (no GPU support)\n");
#endif
    printf("\n");

    // Test 1: Backend initialization
    printf("============================================================================\n");
    printf("[TEST 1] Backend Initialization\n");
    printf("============================================================================\n");
    hpcs_accel_init(&status);
    printf("  hpcs_accel_init() returned: %d (%s)\n", status, status_name(status));
    if (status != HPCS_SUCCESS) {
        printf("  ❌ FAILED: Backend initialization returned %s\n", status_name(status));
        test_failures++;
    } else {
        printf("  ✓ PASSED: Backend initialized successfully\n");
    }
    printf("\n");

    // Test 2: Device count query
    printf("============================================================================\n");
    printf("[TEST 2] Device Count Query\n");
    printf("============================================================================\n");
    hpcs_get_device_count(&count, &status);
    printf("  hpcs_get_device_count() returned:\n");
    printf("    count  = %d\n", count);
    printf("    status = %d (%s)\n", status, status_name(status));

    if (status != HPCS_SUCCESS) {
        printf("  ❌ FAILED: Query returned %s\n", status_name(status));
        test_failures++;
    } else if (count < 0) {
        printf("  ❌ FAILED: Invalid device count (%d)\n", count);
        test_failures++;
    } else {
        printf("  ✓ PASSED: Device count query succeeded\n");
        if (count == 0) {
            printf("  ℹ  No GPU devices available (CPU-only mode)\n");
        } else {
            printf("  ✅ %d GPU device(s) detected\n", count);
        }
    }
    printf("\n");

    // Test 3: Device selection (only if devices available)
    if (count > 0) {
        printf("============================================================================\n");
        printf("[TEST 3] Device Selection\n");
        printf("============================================================================\n");

        // Test valid device selection (device 0)
        printf("  Selecting device 0...\n");
        hpcs_set_device(0, &status);
        printf("  hpcs_set_device(0) returned: %d (%s)\n", status, status_name(status));
        if (status != HPCS_SUCCESS) {
            printf("  ❌ FAILED: Could not select device 0\n");
            test_failures++;
        } else {
            printf("  ✓ PASSED: Device 0 selected\n");

            // Verify device selection
            hpcs_get_device(&device_id, &status);
            printf("  hpcs_get_device() returned: device_id=%d, status=%d\n", device_id, status);
            if (device_id == 0) {
                printf("  ✓ PASSED: Current device is 0\n");
            } else {
                printf("  ⚠️  WARNING: Expected device 0, got device %d\n", device_id);
            }
        }
        printf("\n");

        // Test 4: Invalid device selection
        printf("============================================================================\n");
        printf("[TEST 4] Error Handling (Invalid Device)\n");
        printf("============================================================================\n");
        int invalid_device = count + 10;
        printf("  Attempting to select invalid device %d (count=%d)...\n", invalid_device, count);
        hpcs_set_device(invalid_device, &status);
        printf("  hpcs_set_device(%d) returned: %d (%s)\n", invalid_device, status, status_name(status));
        if (status == HPCS_ERR_INVALID_ARGS) {
            printf("  ✓ PASSED: Correctly rejected invalid device\n");
        } else {
            printf("  ❌ FAILED: Expected HPCS_ERR_INVALID_ARGS, got %s\n", status_name(status));
            test_failures++;
        }
        printf("\n");
    } else {
        printf("============================================================================\n");
        printf("[TEST 3] Device Selection (SKIPPED - No devices available)\n");
        printf("============================================================================\n");
        printf("  ℹ  Skipping device selection tests (count=0)\n");
        printf("\n");
    }

    // Test 5: Repeated device count queries (caching)
    printf("============================================================================\n");
    printf("[TEST 5] Repeated Queries (Caching)\n");
    printf("============================================================================\n");
    int count2, status2;
    hpcs_get_device_count(&count2, &status2);
    printf("  Second hpcs_get_device_count() returned: count=%d, status=%d\n", count2, status2);
    if (count2 == count && status2 == HPCS_SUCCESS) {
        printf("  ✓ PASSED: Cached value matches initial query\n");
    } else {
        printf("  ❌ FAILED: Inconsistent results (first=%d, second=%d)\n", count, count2);
        test_failures++;
    }
    printf("\n");

    // Summary
    printf("============================================================================\n");
    printf("Test Summary\n");
    printf("============================================================================\n");
    if (test_failures == 0) {
        printf("✅ ALL TESTS PASSED\n");
        printf("\n");
        printf("Device Count: %d\n", count);
        if (count > 0) {
            printf("Status: GPU acceleration available\n");
            printf("Ready for GPU benchmarks!\n");
        } else {
            printf("Status: CPU-only mode\n");
#ifdef HPCS_USE_CUDA
            printf("Note: CUDA enabled but no GPU detected\n");
            printf("  - Check nvidia-smi output\n");
            printf("  - Verify CUDA driver installation\n");
#endif
        }
    } else {
        printf("❌ %d TEST(S) FAILED\n", test_failures);
    }
    printf("\n");

    return test_failures;
}
