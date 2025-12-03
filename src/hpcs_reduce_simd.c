/**
 * HPCS SIMD Reduction Kernels - v0.6
 *
 * OpenMP SIMD implementations of core reduction operations.
 * These kernels use compiler-guided vectorization for portability.
 *
 * Kernels:
 * - reduce_sum (OpenMP SIMD)
 * - reduce_mean (OpenMP SIMD)
 * - reduce_min / reduce_max (OpenMP SIMD)
 * - reduce_variance / reduce_std (OpenMP SIMD)
 *
 * Optional intrinsics paths for AVX2/AVX-512 will be added separately.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// SIMD dispatch from v0.6
typedef enum {
    SIMD_NONE = 0,
    SIMD_SSE2 = 1,
    SIMD_AVX = 2,
    SIMD_AVX2 = 3,
    SIMD_AVX512 = 4,
    SIMD_NEON = 5,
    SIMD_OPENMP = 6
} simd_isa_t;

extern void hpcs_register_reduce_sum_kernel(simd_isa_t isa, double (*func)(const double*, int));
extern void hpcs_register_reduce_mean_kernel(simd_isa_t isa, double (*func)(const double*, int));
extern int hpcs_get_simd_width_doubles(void);

// ============================================================================
// OpenMP SIMD Reduction Kernels (Portable, Compiler-Optimized)
// ============================================================================

/**
 * Sum reduction - OpenMP SIMD
 *
 * Uses compiler-guided vectorization with SIMD pragmas.
 * Works on all platforms (Intel, AMD, ARM).
 */
double reduce_sum_openmp_simd(const double *x, int n) {
    double sum = 0.0;

    // OpenMP SIMD reduction
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }

    return sum;
}

/**
 * Sum reduction - Parallel OpenMP + SIMD
 *
 * Combines multithreading with SIMD vectorization.
 * Use for large arrays (>100K elements).
 */
double reduce_sum_parallel_simd(const double *x, int n) {
    double sum = 0.0;

    #pragma omp parallel for simd reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }

    return sum;
}

/**
 * Mean reduction - OpenMP SIMD
 */
double reduce_mean_openmp_simd(const double *x, int n) {
    if (n == 0) return 0.0;

    double sum = reduce_sum_openmp_simd(x, n);
    return sum / (double)n;
}

/**
 * Mean reduction - Parallel OpenMP + SIMD
 */
double reduce_mean_parallel_simd(const double *x, int n) {
    if (n == 0) return 0.0;

    double sum = reduce_sum_parallel_simd(x, n);
    return sum / (double)n;
}

/**
 * Min reduction - OpenMP SIMD
 */
double reduce_min_openmp_simd(const double *x, int n) {
    if (n == 0) return 0.0;

    double min_val = x[0];

    #pragma omp simd reduction(min:min_val)
    for (int i = 1; i < n; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
        }
    }

    return min_val;
}

/**
 * Max reduction - OpenMP SIMD
 */
double reduce_max_openmp_simd(const double *x, int n) {
    if (n == 0) return 0.0;

    double max_val = x[0];

    #pragma omp simd reduction(max:max_val)
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    return max_val;
}

/**
 * Variance reduction - OpenMP SIMD (two-pass algorithm)
 *
 * Pass 1: Compute mean
 * Pass 2: Compute sum of squared deviations
 */
double reduce_variance_openmp_simd(const double *x, int n) {
    if (n <= 1) return 0.0;

    // Pass 1: Mean
    double mean = reduce_mean_openmp_simd(x, n);

    // Pass 2: Sum of squared deviations
    double sum_sq = 0.0;

    #pragma omp simd reduction(+:sum_sq)
    for (int i = 0; i < n; i++) {
        double dev = x[i] - mean;
        sum_sq += dev * dev;
    }

    return sum_sq / (double)(n - 1);  // Sample variance
}

/**
 * Standard deviation - OpenMP SIMD
 */
double reduce_std_openmp_simd(const double *x, int n) {
    double variance = reduce_variance_openmp_simd(x, n);
    return sqrt(variance);
}

// ============================================================================
// Parallel + SIMD Variants (for large arrays)
// ============================================================================

/**
 * Min reduction - Parallel + SIMD
 */
double reduce_min_parallel_simd(const double *x, int n) {
    if (n == 0) return 0.0;

    double min_val = x[0];

    #pragma omp parallel for simd reduction(min:min_val)
    for (int i = 1; i < n; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
        }
    }

    return min_val;
}

/**
 * Max reduction - Parallel + SIMD
 */
double reduce_max_parallel_simd(const double *x, int n) {
    if (n == 0) return 0.0;

    double max_val = x[0];

    #pragma omp parallel for simd reduction(max:max_val)
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    return max_val;
}

/**
 * Variance reduction - Parallel + SIMD
 */
double reduce_variance_parallel_simd(const double *x, int n) {
    if (n <= 1) return 0.0;

    // Pass 1: Mean (parallel)
    double mean = reduce_mean_parallel_simd(x, n);

    // Pass 2: Sum of squared deviations (parallel + SIMD)
    double sum_sq = 0.0;

    #pragma omp parallel for simd reduction(+:sum_sq)
    for (int i = 0; i < n; i++) {
        double dev = x[i] - mean;
        sum_sq += dev * dev;
    }

    return sum_sq / (double)(n - 1);
}

/**
 * Standard deviation - Parallel + SIMD
 */
double reduce_std_parallel_simd(const double *x, int n) {
    double variance = reduce_variance_parallel_simd(x, n);
    return sqrt(variance);
}

// ============================================================================
// Kernel Registration (called during module initialization)
// ============================================================================

/**
 * Register OpenMP SIMD kernels with dispatch system
 */
void hpcs_register_simd_reduction_kernels(void) {
    // Register OpenMP SIMD versions (work on all platforms)
    hpcs_register_reduce_sum_kernel(SIMD_OPENMP, reduce_sum_openmp_simd);
    hpcs_register_reduce_mean_kernel(SIMD_OPENMP, reduce_mean_openmp_simd);

    fprintf(stderr, "[SIMD] Registered OpenMP SIMD reduction kernels\n");
}

// ============================================================================
// Unified API with Auto-Tuning Integration (v0.5 + v0.6)
// ============================================================================

/**
 * Smart reduction dispatch - picks SIMD vs Parallel+SIMD based on size
 *
 * Integrates with v0.5 auto-tuning thresholds and v0.6 SIMD capabilities.
 *
 * Strategy:
 * - Small arrays (< threshold): Use SIMD-only (no thread overhead)
 * - Large arrays (>= threshold): Use Parallel + SIMD (max throughput)
 */
double hpcs_reduce_sum_auto(const double *x, int n, int threshold) {
    if (n < threshold) {
        // Small array: SIMD-only (no threading overhead)
        return reduce_sum_openmp_simd(x, n);
    } else {
        // Large array: Parallel + SIMD
        return reduce_sum_parallel_simd(x, n);
    }
}

double hpcs_reduce_mean_auto(const double *x, int n, int threshold) {
    if (n < threshold) {
        return reduce_mean_openmp_simd(x, n);
    } else {
        return reduce_mean_parallel_simd(x, n);
    }
}

double hpcs_reduce_min_auto(const double *x, int n, int threshold) {
    if (n < threshold) {
        return reduce_min_openmp_simd(x, n);
    } else {
        return reduce_min_parallel_simd(x, n);
    }
}

double hpcs_reduce_max_auto(const double *x, int n, int threshold) {
    if (n < threshold) {
        return reduce_max_openmp_simd(x, n);
    } else {
        return reduce_max_parallel_simd(x, n);
    }
}

double hpcs_reduce_variance_auto(const double *x, int n, int threshold) {
    if (n < threshold) {
        return reduce_variance_openmp_simd(x, n);
    } else {
        return reduce_variance_parallel_simd(x, n);
    }
}

double hpcs_reduce_std_auto(const double *x, int n, int threshold) {
    if (n < threshold) {
        return reduce_std_openmp_simd(x, n);
    } else {
        return reduce_std_parallel_simd(x, n);
    }
}

// ============================================================================
// Module Initialization
// ============================================================================

/**
 * Initialize SIMD reduction module
 *
 * Call this once at startup to register kernels with dispatch system.
 */
void hpcs_simd_reductions_init(void) {
    hpcs_register_simd_reduction_kernels();
}
