/**
 * Adaptive Parallelization Demo
 *
 * Demonstrates hardware-aware adaptive parallelization system.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "hpcs_cpu_detect.h"

void print_separator(void) {
    printf("================================================================================\n");
}

int main(int argc, char **argv) {
    hpcs_cpu_info_t cpu_info;
    int threshold;

    print_separator();
    printf("HPCS Adaptive Parallelization Demo\n");
    print_separator();
    printf("\n");

    // Initialize CPU detection
    printf("Initializing CPU detection...\n\n");
    hpcs_cpu_init();

    // Get CPU information
    hpcs_cpu_get_info(&cpu_info);

    printf("=== Hardware Detection Results (v0.5) ===\n\n");

    printf("CPU Topology:\n");
    printf("  Physical cores:      %d\n", cpu_info.num_physical_cores);
    printf("  Logical cores:       %d (includes hyperthreading)\n", cpu_info.num_logical_cores);
    printf("  Optimal threads:     %d\n", cpu_info.optimal_threads);
    if (cpu_info.cpu_vendor[0] != '\0') {
        printf("  CPU vendor:          %.12s\n", cpu_info.cpu_vendor);
    }
    printf("\n");

    printf("Cache Hierarchy:\n");
    printf("  L1 cache per core:   %d KB\n", cpu_info.l1_cache_size_kb);
    printf("  L2 cache per core:   %d KB\n", cpu_info.l2_cache_size_kb);
    printf("  L3 cache total:      %d KB (%.1f MB)\n",
           cpu_info.l3_cache_size_kb,
           cpu_info.l3_cache_size_kb / 1024.0);
    printf("\n");

    printf("NUMA Topology:\n");
    printf("  NUMA nodes:          %d\n", cpu_info.numa_nodes);
    if (cpu_info.numa_nodes > 1) {
        printf("  Cores per node:      %d\n", cpu_info.cores_per_numa_node);
        printf("  Topology:            Multi-socket / NUMA system\n");
    } else {
        printf("  Topology:            Uniform memory access (UMA)\n");
    }
    printf("\n");

    printf("SIMD Capabilities:\n");
    printf("  Max SIMD width:      %d bits\n", cpu_info.simd_width_bits);
    printf("  SSE2 support:        %s\n", cpu_info.has_sse2 ? "Yes" : "No");
    printf("  AVX support:         %s\n", cpu_info.has_avx ? "Yes" : "No");
    printf("  AVX2 support:        %s\n", cpu_info.has_avx2 ? "Yes" : "No");
    printf("  AVX-512 support:     %s\n", cpu_info.has_avx512 ? "Yes" : "No");
    printf("  ARM NEON support:    %s\n", cpu_info.has_neon ? "Yes" : "No");
    printf("  FMA3 support:        %s\n", cpu_info.has_fma3 ? "Yes" : "No");
    printf("\n");

    // Show adaptive thresholds
    printf("=== Adaptive Thresholds ===\n\n");

    threshold = hpcs_cpu_get_threshold(HPCS_THRESHOLD_SIMPLE_REDUCE);
    printf("Simple reductions (sum, mean, min, max):\n");
    printf("  Threshold: %d elements\n", threshold);
    printf("  Arrays smaller than this run sequentially (overhead not worth it)\n\n");

    threshold = hpcs_cpu_get_threshold(HPCS_THRESHOLD_ROLLING_SIMPLE);
    printf("Rolling simple operations (rolling sum/mean):\n");
    printf("  Threshold: %d elements\n", threshold);
    printf("  Fast operations with marginal parallel benefit\n\n");

    threshold = hpcs_cpu_get_threshold(HPCS_THRESHOLD_COMPUTE_HEAVY);
    printf("Compute-heavy operations (median, MAD, quantile):\n");
    printf("  Threshold: %d elements\n", threshold);
    printf("  Good parallel scaling, moderate threshold\n\n");

    threshold = hpcs_cpu_get_threshold(HPCS_THRESHOLD_ANOMALY_DETECT);
    printf("Anomaly detection:\n");
    printf("  Threshold: %d elements\n", threshold);
    printf("  Excellent parallel scaling, aggressive parallelization\n\n");

    // Demonstrate usage
    print_separator();
    printf("Example: Auto-selecting threads for OpenMP\n");
    print_separator();
    printf("\n");

    int optimal_threads = hpcs_cpu_get_optimal_threads();
    printf("Setting OpenMP threads to: %d (physical cores)\n", optimal_threads);
    omp_set_num_threads(optimal_threads);

    printf("\nTest parallel region:\n");
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        #pragma omp single
        {
            printf("  Running with %d threads\n", nthreads);
        }
    }

    // Example threshold usage
    printf("\n");
    print_separator();
    printf("Example: Adaptive threshold usage\n");
    print_separator();
    printf("\n");

    int test_sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    threshold = hpcs_cpu_get_threshold(HPCS_THRESHOLD_COMPUTE_HEAVY);

    printf("For compute-heavy operations (threshold = %d):\n\n", threshold);
    printf("  Array Size    | Will Parallelize?\n");
    printf("  --------------|-------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        int size = test_sizes[i];
        const char *will_parallel = (size > threshold) ? "YES" : "NO";
        printf("  %-13d | %s\n", size, will_parallel);
    }

    printf("\n");
    print_separator();
    printf("Architecture-Specific Recommendations\n");
    print_separator();
    printf("\n");

    if (cpu_info.num_physical_cores <= 2) {
        printf("⚠ Low-core system detected (%d cores)\n", cpu_info.num_physical_cores);
        printf("  Recommendation: Parallelization disabled for most operations\n");
        printf("  Reason: OpenMP overhead exceeds benefit on 1-2 core systems\n");
    } else if (cpu_info.num_physical_cores <= 8) {
        printf("✓ Mid-range system detected (%d cores)\n", cpu_info.num_physical_cores);
        printf("  Recommendation: Standard thresholds, balanced parallelization\n");
        printf("  Expected speedup: 2-4x on compute-heavy operations\n");
    } else if (cpu_info.num_physical_cores <= 32) {
        printf("✓ High-end system detected (%d cores)\n", cpu_info.num_physical_cores);
        printf("  Recommendation: Lower thresholds, aggressive parallelization\n");
        printf("  Expected speedup: 4-16x on compute-heavy operations\n");
    } else {
        printf("✓ HPC system detected (%d cores)\n", cpu_info.num_physical_cores);
        printf("  Recommendation: Very low thresholds, maximum parallelization\n");
        printf("  Expected speedup: 16x+ on compute-heavy operations\n");
        printf("  Note: Consider NUMA-aware scheduling for best performance\n");
    }

    printf("\n");
    print_separator();
    printf("Deployment Notes\n");
    print_separator();
    printf("\n");

    printf("This same binary will automatically adapt to:\n");
    printf("  • AWS EC2 instances (t3, c6i, m6i, etc.)\n");
    printf("  • Azure VMs (D-series, F-series, etc.)\n");
    printf("  • Google Cloud instances (n2, c2, etc.)\n");
    printf("  • On-premise HPC clusters\n");
    printf("\n");
    printf("No recompilation or configuration needed!\n");
    printf("\n");

    // v0.5: SIMD optimization notes
    print_separator();
    printf("v0.5: SIMD Optimization Potential\n");
    print_separator();
    printf("\n");

    if (cpu_info.has_avx512) {
        printf("🚀 AVX-512 detected (%d-bit SIMD)\n", cpu_info.simd_width_bits);
        printf("   Future: v0.6+ can add AVX-512 kernels for 2-4x additional speedup\n");
        printf("   Benefit: Process 8 doubles or 16 floats per instruction\n");
    } else if (cpu_info.has_avx2) {
        printf("✓ AVX2 detected (%d-bit SIMD)\n", cpu_info.simd_width_bits);
        printf("   Future: v0.6+ can add AVX2 kernels for 2-3x additional speedup\n");
        printf("   Benefit: Process 4 doubles or 8 floats per instruction\n");
    } else if (cpu_info.has_avx) {
        printf("✓ AVX detected (%d-bit SIMD)\n", cpu_info.simd_width_bits);
        printf("   Future: v0.6+ can add AVX kernels for 2x additional speedup\n");
        printf("   Benefit: Process 4 doubles or 8 floats per instruction\n");
    } else if (cpu_info.has_sse2) {
        printf("○ SSE2 detected (%d-bit SIMD)\n", cpu_info.simd_width_bits);
        printf("   Current: Baseline x86-64 support\n");
        printf("   Benefit: Process 2 doubles or 4 floats per instruction\n");
    } else if (cpu_info.has_neon) {
        printf("○ ARM NEON detected (%d-bit SIMD)\n", cpu_info.simd_width_bits);
        printf("   Future: v0.6+ can add NEON kernels for 2x additional speedup\n");
        printf("   Benefit: Process 2 doubles or 4 floats per instruction\n");
    }

    printf("\n");
    printf("Note: v0.5 focuses on CPU auto-tuning. SIMD kernels planned for v0.6+.\n");
    printf("\n");

    // v0.5: Tuning Configuration System
    print_separator();
    printf("v0.5: Tuning Configuration System\n");
    print_separator();
    printf("\n");

    printf("The tuning system stores operation-specific parameters:\n");
    printf("  • Parallelization thresholds per operation class\n");
    printf("  • Optimal thread counts per operation class\n");
    printf("  • NUMA affinity modes per operation class\n");
    printf("\n");

    // Get and display current tuning
    hpcs_tuning_t tuning;
    hpcs_get_tuning(&tuning);

    printf("Current Configuration (hardware-based defaults):\n");
    printf("\n");
    hpcs_print_tuning(&tuning);

    printf("\nExample: Querying tuning parameters by operation class:\n");
    printf("\n");

    int threshold_robust = hpcs_get_tuning_threshold(HPCS_OP_CLASS_ROBUST);
    int threads_robust = hpcs_get_tuning_threads(HPCS_OP_CLASS_ROBUST);
    int numa_mode_robust = hpcs_get_tuning_numa_mode(HPCS_OP_CLASS_ROBUST);

    const char *numa_names[] = {"AUTO", "COMPACT", "SPREAD"};
    printf("  Robust statistics (median, MAD, quantile):\n");
    printf("    Threshold:  %d elements\n", threshold_robust);
    printf("    Threads:    %d\n", threads_robust);
    printf("    NUMA mode:  %s\n", numa_names[numa_mode_robust]);
    printf("\n");

    printf("This configuration is automatically optimized for your CPU!\n");
    printf("\n");

    return 0;
}
