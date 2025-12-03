/**
 * NUMA Affinity Demo - HPCSeries Core v0.5
 *
 * Demonstrates NUMA thread affinity features including:
 * - NUMA topology detection
 * - Compact affinity (same NUMA node)
 * - Spread affinity (across NUMA nodes)
 * - Automatic affinity selection
 * - OpenMP integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// HPCS CPU Detection API
typedef struct {
    int num_physical_cores;
    int num_logical_cores;
    int l1_cache_size_kb;
    int l2_cache_size_kb;
    int l3_cache_size_kb;
    int optimal_threads;
    int numa_nodes;
    int cores_per_numa_node;
    void *core_to_numa_map;
    int has_sse2;
    int has_avx;
    int has_avx2;
    int has_avx512;
    int has_neon;
    int has_fma3;
    int simd_width_bits;
    char cpu_vendor[64];
    char cpu_model[128];
    int initialized;
} hpcs_cpu_info_t;

// External HPCS functions
extern void hpcs_cpu_detect_init(void);
extern void hpcs_cpu_get_info(hpcs_cpu_info_t *info);
extern void hpcs_set_affinity_mode(int mode, int *status);
extern int hpcs_get_affinity_mode(void);
extern int hpcs_apply_compact_affinity(int num_threads, int preferred_node);
extern int hpcs_apply_spread_affinity(int num_threads);
extern int hpcs_apply_numa_affinity(int num_threads, int op_class);
extern int hpcs_get_core_affinity(int core_id, int *numa_node);

// Operation classes
#define OP_SIMPLE    1
#define OP_ROLLING   2
#define OP_ROBUST    3
#define OP_ANOMALY   4

// Affinity modes
#define AFFINITY_AUTO     0
#define AFFINITY_COMPACT  1
#define AFFINITY_SPREAD   2

/**
 * Simple parallel workload for testing affinity
 */
double run_parallel_workload(int n, int num_threads) {
    double *data = (double*)malloc(n * sizeof(double));
    double sum = 0.0;
    int i;

    // Initialize data
    for (i = 0; i < n; i++) {
        data[i] = (double)i;
    }

    // Parallel reduction with affinity applied
    clock_t start = clock();

    #pragma omp parallel for num_threads(num_threads) reduction(+:sum)
    for (i = 0; i < n; i++) {
        sum += data[i] * data[i];  // Simple computation
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    free(data);
    return elapsed;
}

/**
 * Print CPU and NUMA topology information
 */
void print_system_info(void) {
    hpcs_cpu_info_t info;
    hpcs_cpu_get_info(&info);

    printf("=== System Information ===\n");
    printf("CPU Vendor:      %s\n", info.cpu_vendor);
    printf("Physical Cores:  %d\n", info.num_physical_cores);
    printf("Logical Cores:   %d\n", info.num_logical_cores);
    printf("Optimal Threads: %d\n", info.optimal_threads);
    printf("\n");
    printf("L1 Cache:        %d KB\n", info.l1_cache_size_kb);
    printf("L2 Cache:        %d KB\n", info.l2_cache_size_kb);
    printf("L3 Cache:        %d KB\n", info.l3_cache_size_kb);
    printf("\n");
    printf("NUMA Nodes:      %d\n", info.numa_nodes);
    printf("Cores per Node:  %d\n", info.cores_per_numa_node);
    printf("\n");
    printf("SIMD Width:      %d bits\n", info.simd_width_bits);
    printf("Has AVX:         %s\n", info.has_avx ? "Yes" : "No");
    printf("Has AVX2:        %s\n", info.has_avx2 ? "Yes" : "No");
    printf("Has AVX-512:     %s\n", info.has_avx512 ? "Yes" : "No");
    printf("==========================\n\n");
}

/**
 * Test compact affinity mode
 */
void test_compact_affinity(int num_threads, int n) {
    int status;

    printf("=== Testing COMPACT Affinity ===\n");
    printf("Binding %d threads to same NUMA node...\n", num_threads);

    status = hpcs_apply_compact_affinity(num_threads, -1);
    if (status == 0) {
        printf("Compact affinity applied successfully\n");
    } else {
        printf("Compact affinity failed (may not be supported on this platform)\n");
    }

    double elapsed = run_parallel_workload(n, num_threads);
    printf("Workload time: %.6f seconds\n", elapsed);
    printf("================================\n\n");
}

/**
 * Test spread affinity mode
 */
void test_spread_affinity(int num_threads, int n) {
    int status;

    printf("=== Testing SPREAD Affinity ===\n");
    printf("Distributing %d threads across NUMA nodes...\n", num_threads);

    status = hpcs_apply_spread_affinity(num_threads);
    if (status == 0) {
        printf("Spread affinity applied successfully\n");
    } else {
        printf("Spread affinity failed (may not be supported on this platform)\n");
    }

    double elapsed = run_parallel_workload(n, num_threads);
    printf("Workload time: %.6f seconds\n", elapsed);
    printf("==============================\n\n");
}

/**
 * Test automatic affinity selection
 */
void test_automatic_affinity(int num_threads, int n) {
    int status;

    printf("=== Testing AUTOMATIC Affinity ===\n");
    printf("Using operation-specific affinity heuristics...\n");

    // Test rolling operation (prefers COMPACT)
    printf("\n[Rolling Operation - prefers COMPACT]\n");
    status = hpcs_apply_numa_affinity(num_threads, OP_ROLLING);
    if (status == 0) {
        printf("Automatic affinity applied successfully\n");
    }
    double elapsed_rolling = run_parallel_workload(n, num_threads);
    printf("Workload time: %.6f seconds\n", elapsed_rolling);

    // Test anomaly detection (prefers SPREAD)
    printf("\n[Anomaly Detection - prefers SPREAD]\n");
    status = hpcs_apply_numa_affinity(num_threads, OP_ANOMALY);
    if (status == 0) {
        printf("Automatic affinity applied successfully\n");
    }
    double elapsed_anomaly = run_parallel_workload(n, num_threads);
    printf("Workload time: %.6f seconds\n", elapsed_anomaly);

    printf("==================================\n\n");
}

/**
 * Test core-to-NUMA mapping
 */
void test_core_mapping(void) {
    hpcs_cpu_info_t info;
    int numa_node, status;

    hpcs_cpu_get_info(&info);

    printf("=== Core-to-NUMA Mapping ===\n");
    printf("Core ID -> NUMA Node\n");
    printf("--------------------\n");

    for (int core = 0; core < info.num_physical_cores && core < 16; core++) {
        status = hpcs_get_core_affinity(core, &numa_node);
        if (status == 0) {
            printf("Core %2d -> Node %d\n", core, numa_node);
        }
    }

    if (info.num_physical_cores > 16) {
        printf("... (showing first 16 cores)\n");
    }

    printf("============================\n\n");
}

/**
 * Main demo program
 */
int main(int argc, char *argv[]) {
    int num_threads = 4;
    int n = 10000000;  // 10M elements

    // Parse command line arguments
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    if (argc > 2) {
        n = atoi(argv[2]);
    }

    printf("NUMA Affinity Demo - HPCSeries Core v0.5\n");
    printf("=========================================\n\n");

    // Initialize CPU detection
    hpcs_cpu_detect_init();

    // Print system information
    print_system_info();

    // Get CPU info
    hpcs_cpu_info_t info;
    hpcs_cpu_get_info(&info);

    // Check if NUMA is available
    if (info.numa_nodes <= 1) {
        printf("NOTE: This system has only 1 NUMA node.\n");
        printf("NUMA affinity features will have no effect on performance.\n");
        printf("To see NUMA benefits, run on a multi-socket system.\n\n");
    }

    // Show core-to-NUMA mapping
    test_core_mapping();

    // Test different affinity modes
    printf("Running performance tests with %d threads and %d elements...\n\n",
           num_threads, n);

    // Baseline (no affinity)
    printf("=== Baseline (No Affinity) ===\n");
    double baseline = run_parallel_workload(n, num_threads);
    printf("Workload time: %.6f seconds\n", baseline);
    printf("===============================\n\n");

    // Test compact affinity
    test_compact_affinity(num_threads, n);

    // Test spread affinity
    test_spread_affinity(num_threads, n);

    // Test automatic affinity
    test_automatic_affinity(num_threads, n);

    // Summary
    printf("=== Performance Summary ===\n");
    printf("Baseline time: %.6f seconds\n", baseline);
    printf("===========================\n\n");

    printf("Demo completed successfully!\n");
    printf("\n");
    printf("Tips:\n");
    printf("- On NUMA systems, you should see performance differences\n");
    printf("- COMPACT mode is best for cache-locality operations\n");
    printf("- SPREAD mode is best for memory-bandwidth operations\n");
    printf("- Set OMP_PLACES=cores and OMP_PROC_BIND for OpenMP control\n");

    return 0;
}
