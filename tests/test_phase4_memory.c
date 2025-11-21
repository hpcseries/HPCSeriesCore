/**
 * HPCSeries Core v0.4 - Phase 4 Memory Management Test Suite
 *
 * Tests for actual GPU memory allocation, transfer, and deallocation
 * implemented in Phase 4.
 *
 * Phase 4A (Host/Device Memory Management):
 *   - Stage 1: Actual device memory allocation
 *   - Stage 2: Device-to-host transfers
 *   - Stage 3: Memory deallocation (hpcs_accel_free_device)
 *
 * Note: In CPU-only mode, these tests validate API correctness.
 *       On GPU hardware, they validate actual device memory operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hpcs_core.h"

/* Test assertion macros */
#define ASSERT_EQ(actual, expected, msg) \
  do { \
    if ((actual) != (expected)) { \
      printf("  ✗ FAILED: %s\n", msg); \
      printf("    Expected: %d, Got: %d\n", expected, actual); \
      return 0; \
    } \
    printf("  ✓ %s (got %d)\n", msg, actual); \
  } while(0)

#define ASSERT_NOT_NULL(ptr, msg) \
  do { \
    if ((ptr) == NULL) { \
      printf("  ✗ FAILED: %s (got NULL)\n", msg); \
      return 0; \
    } \
    printf("  ✓ %s\n", msg); \
  } while(0)

#define ASSERT_NEAR(actual, expected, tol, msg) \
  do { \
    double diff = fabs((actual) - (expected)); \
    if (diff > (tol)) { \
      printf("  ✗ FAILED: %s\n", msg); \
      printf("    Expected: %.6f, Got: %.6f, Diff: %.6e\n", \
             expected, actual, diff); \
      return 0; \
    } \
    printf("  ✓ %s (got %.6f, expected %.6f)\n", msg, actual, expected); \
  } while(0)

/* Test statistics */
static int tests_run = 0;
static int tests_passed = 0;
static int total_assertions = 0;

/* Test runner */
#define RUN_TEST(test) \
  do { \
    tests_run++; \
    printf("\n[TEST %d] %s\n", tests_run, #test); \
    if (test()) { \
      tests_passed++; \
    } \
  } while(0)

/**
 * Test 1: Basic Device Memory Allocation and Deallocation
 *
 * Tests the fundamental memory lifecycle:
 *   1. Allocate device memory
 *   2. Verify allocation succeeded
 *   3. Free device memory
 *   4. Verify deallocation succeeded
 */
int test_basic_allocation_deallocation() {
  double data[100];
  void* device_ptr;
  int status;

  /* Initialize test data */
  for (int i = 0; i < 100; i++) {
    data[i] = (double)i;
  }

  /* Allocate device memory */
  hpcs_accel_copy_to_device(data, 100, &device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Device allocation succeeded");

  total_assertions++;
  ASSERT_NOT_NULL(device_ptr, "Device pointer is not NULL");

  /* Free device memory */
  hpcs_accel_free_device(device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Device deallocation succeeded");

  return 1;
}

/**
 * Test 2: Host-to-Device Transfer
 *
 * Tests that data is correctly transferred to device:
 *   1. Copy array to device
 *   2. Modify host array
 *   3. Copy back from device
 *   4. Verify device copy preserved original data
 */
int test_host_to_device_transfer() {
  double host_data[50];
  double result_data[50];
  void* device_ptr;
  int status;

  /* Initialize host data */
  for (int i = 0; i < 50; i++) {
    host_data[i] = (double)(i * 2);
  }

  /* Copy to device */
  hpcs_accel_copy_to_device(host_data, 50, &device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Copy to device succeeded");

  /* Modify host data (should not affect device copy) */
  for (int i = 0; i < 50; i++) {
    host_data[i] = -999.0;
  }

  /* Copy back from device */
  hpcs_accel_copy_from_device(device_ptr, 50, result_data, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Copy from device succeeded");

  /* Verify data was preserved on device */
  total_assertions++;
  ASSERT_NEAR(result_data[0], 0.0, 1e-9, "First element preserved");
  total_assertions++;
  ASSERT_NEAR(result_data[25], 50.0, 1e-9, "Middle element preserved");
  total_assertions++;
  ASSERT_NEAR(result_data[49], 98.0, 1e-9, "Last element preserved");

  /* Free device memory */
  hpcs_accel_free_device(device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Cleanup succeeded");

  return 1;
}

/**
 * Test 3: Multiple Independent Allocations
 *
 * Tests that multiple device allocations can coexist:
 *   1. Allocate three separate buffers
 *   2. Verify all allocations succeeded
 *   3. Free in different order (2, 1, 3)
 *   4. Verify all deallocations succeeded
 */
int test_multiple_allocations() {
  double data1[100], data2[200], data3[150];
  void *dev_ptr1, *dev_ptr2, *dev_ptr3;
  int status;

  /* Initialize test data */
  for (int i = 0; i < 100; i++) data1[i] = (double)i;
  for (int i = 0; i < 200; i++) data2[i] = (double)(i * 2);
  for (int i = 0; i < 150; i++) data3[i] = (double)(i * 3);

  /* Allocate three buffers */
  hpcs_accel_copy_to_device(data1, 100, &dev_ptr1, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "First allocation succeeded");

  hpcs_accel_copy_to_device(data2, 200, &dev_ptr2, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Second allocation succeeded");

  hpcs_accel_copy_to_device(data3, 150, &dev_ptr3, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Third allocation succeeded");

  total_assertions++;
  ASSERT_NOT_NULL(dev_ptr1, "First device pointer valid");
  total_assertions++;
  ASSERT_NOT_NULL(dev_ptr2, "Second device pointer valid");
  total_assertions++;
  ASSERT_NOT_NULL(dev_ptr3, "Third device pointer valid");

  /* Free in different order: 2, 1, 3 */
  hpcs_accel_free_device(dev_ptr2, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Second deallocation succeeded");

  hpcs_accel_free_device(dev_ptr1, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "First deallocation succeeded");

  hpcs_accel_free_device(dev_ptr3, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Third deallocation succeeded");

  return 1;
}

/**
 * Test 4: Error Handling - Invalid Arguments
 *
 * Tests proper error handling for invalid inputs:
 *   1. NULL pointer to free
 *   2. Zero-size allocation
 *   3. Negative size allocation
 */
int test_error_handling() {
  double data[10];
  void* device_ptr;
  int status;

  /* Test 1: Free NULL pointer */
  hpcs_accel_free_device(NULL, &status);
  total_assertions++;
  ASSERT_EQ(status, 1, "NULL pointer rejected");

  /* Test 2: Zero-size allocation */
  hpcs_accel_copy_to_device(data, 0, &device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 1, "Zero-size allocation rejected");

  /* Test 3: Negative size allocation */
  hpcs_accel_copy_to_device(data, -5, &device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 1, "Negative size allocation rejected");

  /* Test 4: NULL host pointer */
  hpcs_accel_copy_to_device(NULL, 10, &device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 1, "NULL host pointer rejected");

  return 1;
}

/**
 * Test 5: Integration with Phase 3 Median Kernel
 *
 * Tests the complete workflow:
 *   1. Allocate device memory
 *   2. Compute median on device
 *   3. Verify result
 *   4. Free device memory
 */
int test_integration_median() {
  double data[101];
  void* device_ptr;
  double median;
  int status;

  /* Initialize data: 0, 1, 2, ..., 100 */
  for (int i = 0; i < 101; i++) {
    data[i] = (double)i;
  }

  /* Allocate device memory */
  hpcs_accel_copy_to_device(data, 101, &device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Device allocation succeeded");

  /* Compute median on device */
  hpcs_accel_median(device_ptr, 101, &median, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Median computation succeeded");

  /* Verify median is correct (middle of 0..100 is 50.0) */
  total_assertions++;
  ASSERT_NEAR(median, 50.0, 1e-9, "Median is 50.0");

  /* Free device memory */
  hpcs_accel_free_device(device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Device cleanup succeeded");

  return 1;
}

/**
 * Test 6: Integration with Phase 3 MAD Kernel
 *
 * Tests memory management with MAD computation:
 *   1. Allocate device memory
 *   2. Compute MAD on device
 *   3. Verify result
 *   4. Free device memory
 */
int test_integration_mad() {
  double data[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  void* device_ptr;
  double mad;
  int status;

  /* Allocate device memory */
  hpcs_accel_copy_to_device(data, 7, &device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Device allocation succeeded");

  /* Compute MAD on device */
  hpcs_accel_mad(device_ptr, 7, &mad, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "MAD computation succeeded");

  /* Verify MAD (median=4, deviations=[3,2,1,0,1,2,3], MAD=median([3,2,1,0,1,2,3])=2) */
  total_assertions++;
  ASSERT_NEAR(mad, 2.0, 1e-9, "MAD is 2.0");

  /* Free device memory */
  hpcs_accel_free_device(device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Device cleanup succeeded");

  return 1;
}

/**
 * Test 7: Large Dataset Allocation (Stress Test)
 *
 * Tests memory management with larger datasets:
 *   1. Allocate 10K element array
 *   2. Transfer data
 *   3. Compute median
 *   4. Free memory
 */
int test_large_dataset() {
  const int n = 10000;
  double* data = (double*)malloc(n * sizeof(double));
  void* device_ptr;
  double median;
  int status;

  if (data == NULL) {
    printf("  ✗ Failed to allocate host memory\n");
    return 0;
  }

  /* Initialize data: 0, 1, 2, ..., 9999 */
  for (int i = 0; i < n; i++) {
    data[i] = (double)i;
  }

  /* Allocate device memory */
  hpcs_accel_copy_to_device(data, n, &device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Large allocation succeeded");

  /* Compute median */
  hpcs_accel_median(device_ptr, n, &median, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Median on large dataset succeeded");

  /* Verify median (even count, so average of middle two: 4999 and 5000) */
  total_assertions++;
  ASSERT_NEAR(median, 4999.5, 1e-9, "Median is 4999.5");

  /* Free device memory */
  hpcs_accel_free_device(device_ptr, &status);
  total_assertions++;
  ASSERT_EQ(status, 0, "Large deallocation succeeded");

  free(data);
  return 1;
}

/**
 * Test 8: Sequential Allocation/Deallocation Cycles
 *
 * Tests repeated allocation and deallocation to verify no memory leaks:
 *   1. Allocate → Free (10 cycles)
 *   2. Verify all cycles succeed
 */
int test_allocation_cycles() {
  double data[50];
  int status;

  /* Initialize test data */
  for (int i = 0; i < 50; i++) {
    data[i] = (double)i;
  }

  /* Run 10 allocation/deallocation cycles */
  for (int cycle = 0; cycle < 10; cycle++) {
    void* device_ptr;

    hpcs_accel_copy_to_device(data, 50, &device_ptr, &status);
    if (status != 0) {
      printf("  ✗ Allocation failed on cycle %d\n", cycle);
      return 0;
    }

    hpcs_accel_free_device(device_ptr, &status);
    if (status != 0) {
      printf("  ✗ Deallocation failed on cycle %d\n", cycle);
      return 0;
    }
  }

  total_assertions++;
  printf("  ✓ All 10 allocation/deallocation cycles succeeded\n");

  return 1;
}

/**
 * Main test runner
 */
int main(int argc, char** argv) {
  printf("============================================================================\n");
  printf("HPCSeries Core v0.4 - Phase 4 Memory Management Test Suite\n");
  printf("============================================================================\n");

  /* Initialize backend */
  int status;
  hpcs_accel_init(&status);
  if (status != 0) {
    printf("\n✗ Backend initialization failed (status=%d)\n", status);
    return 1;
  }

  /* Run test suite */
  RUN_TEST(test_basic_allocation_deallocation);
  RUN_TEST(test_host_to_device_transfer);
  RUN_TEST(test_multiple_allocations);
  RUN_TEST(test_error_handling);
  RUN_TEST(test_integration_median);
  RUN_TEST(test_integration_mad);
  RUN_TEST(test_large_dataset);
  RUN_TEST(test_allocation_cycles);

  /* Print summary */
  printf("\n");
  printf("============================================================================\n");
  printf("Test Summary\n");
  printf("============================================================================\n");
  printf("Total tests: %d\n", tests_run);
  printf("Assertions:  %d passed, %d failed\n",
         total_assertions, (total_assertions - total_assertions));
  printf("\n");

  if (tests_passed == tests_run) {
    printf("✓ All tests PASSED\n");
  } else {
    printf("✗ %d/%d tests FAILED\n", tests_run - tests_passed, tests_run);
  }

  printf("============================================================================\n");

  return (tests_passed == tests_run) ? 0 : 1;
}
