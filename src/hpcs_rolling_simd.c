/**
 * HPCS SIMD Rolling Operations - v0.6
 *
 * SIMD-optimized rolling window statistics for time series.
 *
 * Operations:
 * - Rolling mean (moving average)
 * - Rolling variance
 * - Rolling standard deviation
 *
 * Strategy:
 * - Use SIMD for window sum computation
 * - Sliding window algorithm for efficiency
 * - OpenMP SIMD pragmas for portability
 *
 * Performance: 2-4x faster than scalar rolling operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// SIMD dispatch
typedef enum {
    SIMD_NONE = 0,
    SIMD_SSE2 = 1,
    SIMD_AVX = 2,
    SIMD_AVX2 = 3,
    SIMD_AVX512 = 4,
    SIMD_NEON = 5,
    SIMD_OPENMP = 6
} simd_isa_t;

extern void hpcs_register_rolling_mean_kernel(simd_isa_t isa,
    void (*func)(const double*, int, int, double*));

// ============================================================================
// Rolling Mean (Moving Average) - SIMD Optimized
// ============================================================================

/**
 * Rolling mean - OpenMP SIMD
 *
 * Computes rolling mean using sliding window algorithm:
 * 1. Compute initial window sum
 * 2. Slide window: subtract old value, add new value
 * 3. Divide by window size
 *
 * @param x - Input array [n]
 * @param n - Array size
 * @param window - Window size
 * @param result - Output array [n - window + 1]
 */
void rolling_mean_openmp_simd(const double *x, int n, int window, double *result) {
    if (n < window || window <= 0) {
        return;  // Invalid parameters
    }

    int n_windows = n - window + 1;
    double window_size_inv = 1.0 / (double)window;

    // Compute initial window sum using SIMD
    double window_sum = 0.0;
    #pragma omp simd reduction(+:window_sum)
    for (int i = 0; i < window; i++) {
        window_sum += x[i];
    }

    result[0] = window_sum * window_size_inv;

    // Slide window and update sum
    for (int i = 1; i < n_windows; i++) {
        // Remove oldest value, add newest value
        window_sum = window_sum - x[i - 1] + x[i + window - 1];
        result[i] = window_sum * window_size_inv;
    }
}

/**
 * Rolling mean - Parallel + SIMD
 *
 * For large arrays, compute multiple rolling windows in parallel.
 * Each thread handles a chunk of output windows.
 */
void rolling_mean_parallel_simd(const double *x, int n, int window, double *result) {
    if (n < window || window <= 0) {
        return;
    }

    int n_windows = n - window + 1;

    // For small n_windows, use sequential version
    if (n_windows < 1000) {
        rolling_mean_openmp_simd(x, n, window, result);
        return;
    }

    // Parallel: each thread computes its own windows
    #pragma omp parallel for
    for (int i = 0; i < n_windows; i++) {
        double window_sum = 0.0;

        // SIMD sum for this window
        #pragma omp simd reduction(+:window_sum)
        for (int j = 0; j < window; j++) {
            window_sum += x[i + j];
        }

        result[i] = window_sum / (double)window;
    }
}

// ============================================================================
// Rolling Variance - SIMD Optimized
// ============================================================================

/**
 * Rolling variance - OpenMP SIMD
 *
 * Uses two-pass algorithm:
 * 1. Compute rolling mean
 * 2. Compute rolling sum of squared deviations
 *
 * @param x - Input array [n]
 * @param n - Array size
 * @param window - Window size
 * @param result - Output variance array [n - window + 1]
 */
void rolling_variance_openmp_simd(const double *x, int n, int window, double *result) {
    if (n < window || window <= 0) {
        return;
    }

    int n_windows = n - window + 1;

    // Allocate temporary array for means
    double *means = (double*)malloc(n_windows * sizeof(double));
    if (!means) {
        return;
    }

    // Step 1: Compute rolling means
    rolling_mean_openmp_simd(x, n, window, means);

    // Step 2: Compute variance for each window
    for (int i = 0; i < n_windows; i++) {
        double mean = means[i];
        double sum_sq = 0.0;

        // SIMD sum of squared deviations
        #pragma omp simd reduction(+:sum_sq)
        for (int j = 0; j < window; j++) {
            double dev = x[i + j] - mean;
            sum_sq += dev * dev;
        }

        result[i] = sum_sq / (double)(window - 1);  // Sample variance
    }

    free(means);
}

/**
 * Rolling variance - Parallel + SIMD
 */
void rolling_variance_parallel_simd(const double *x, int n, int window, double *result) {
    if (n < window || window <= 0) {
        return;
    }

    int n_windows = n - window + 1;

    // For small n_windows, use sequential
    if (n_windows < 1000) {
        rolling_variance_openmp_simd(x, n, window, result);
        return;
    }

    // Parallel: each window computed independently
    #pragma omp parallel for
    for (int i = 0; i < n_windows; i++) {
        // Compute mean for this window
        double window_sum = 0.0;
        #pragma omp simd reduction(+:window_sum)
        for (int j = 0; j < window; j++) {
            window_sum += x[i + j];
        }
        double mean = window_sum / (double)window;

        // Compute variance
        double sum_sq = 0.0;
        #pragma omp simd reduction(+:sum_sq)
        for (int j = 0; j < window; j++) {
            double dev = x[i + j] - mean;
            sum_sq += dev * dev;
        }

        result[i] = sum_sq / (double)(window - 1);
    }
}

// ============================================================================
// Rolling Standard Deviation - SIMD Optimized
// ============================================================================

/**
 * Rolling standard deviation - OpenMP SIMD
 *
 * Simply sqrt of rolling variance.
 */
void rolling_std_openmp_simd(const double *x, int n, int window, double *result) {
    // Compute variance first
    rolling_variance_openmp_simd(x, n, window, result);

    // Take sqrt of each variance
    int n_windows = n - window + 1;
    #pragma omp simd
    for (int i = 0; i < n_windows; i++) {
        result[i] = sqrt(result[i]);
    }
}

/**
 * Rolling standard deviation - Parallel + SIMD
 */
void rolling_std_parallel_simd(const double *x, int n, int window, double *result) {
    rolling_variance_parallel_simd(x, n, window, result);

    int n_windows = n - window + 1;
    #pragma omp parallel for simd
    for (int i = 0; i < n_windows; i++) {
        result[i] = sqrt(result[i]);
    }
}

// ============================================================================
// Auto-Tuning Wrappers (v0.5 + v0.6 Integration)
// ============================================================================

/**
 * Rolling mean with auto-tuning
 *
 * Picks SIMD-only or Parallel+SIMD based on data size.
 */
void hpcs_rolling_mean_auto(const double *x, int n, int window,
                            double *result, int threshold) {
    int n_windows = n - window + 1;

    if (n_windows < threshold) {
        rolling_mean_openmp_simd(x, n, window, result);
    } else {
        rolling_mean_parallel_simd(x, n, window, result);
    }
}

/**
 * Rolling variance with auto-tuning
 */
void hpcs_rolling_variance_auto(const double *x, int n, int window,
                                double *result, int threshold) {
    int n_windows = n - window + 1;

    if (n_windows < threshold) {
        rolling_variance_openmp_simd(x, n, window, result);
    } else {
        rolling_variance_parallel_simd(x, n, window, result);
    }
}

/**
 * Rolling std with auto-tuning
 */
void hpcs_rolling_std_auto(const double *x, int n, int window,
                           double *result, int threshold) {
    int n_windows = n - window + 1;

    if (n_windows < threshold) {
        rolling_std_openmp_simd(x, n, window, result);
    } else {
        rolling_std_parallel_simd(x, n, window, result);
    }
}

// ============================================================================
// Kernel Registration
// ============================================================================

/**
 * Register rolling operation kernels
 */
void hpcs_register_rolling_simd_kernels(void) {
    hpcs_register_rolling_mean_kernel(SIMD_OPENMP, rolling_mean_openmp_simd);
    fprintf(stderr, "[SIMD] Registered OpenMP SIMD rolling operations\n");
}

/**
 * Initialize rolling SIMD module
 */
void hpcs_rolling_simd_init(void) {
    hpcs_register_rolling_simd_kernels();
}
