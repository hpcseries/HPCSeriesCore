#ifndef HPCS_CORE_H
#define HPCS_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------- */
/* Status codes (must match hpcs_constants.f90)                          */
/* --------------------------------------------------------------------- */
enum {
    HPCS_SUCCESS          = 0,
    HPCS_ERR_INVALID_ARGS = 1,
    HPCS_ERR_NUMERIC_FAIL = 2
    /* add more here if you extend hpcs_constants later */
};

/* --------------------------------------------------------------------- */
/* 1D Kernels (hpcs_core_1d)                                             */
/* --------------------------------------------------------------------- */

/* rolling sum: y[i] = sum of last window values up to i */
void hpcs_rolling_sum(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* rolling mean: rolling sum divided by min(i, window) */
void hpcs_rolling_mean(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* rolling variance: population variance over rolling window (v0.2) */
void hpcs_rolling_variance(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* rolling std: standard deviation over rolling window (v0.2) */
void hpcs_rolling_std(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* z-score normalization (Welford-based) */
void hpcs_zscore(
    const double *x,
    int           n,
    double       *y,
    int          *status
);

/* --------------------------------------------------------------------- */
/* Reductions (hpcs_core_reductions)                                     */
/* --------------------------------------------------------------------- */

/* out = sum(x[:n]) */
void hpcs_reduce_sum(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* out = min(x[:n]); n==0 -> +huge sentinel */
void hpcs_reduce_min(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* out = max(x[:n]); n==0 -> -huge sentinel */
void hpcs_reduce_max(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* out = mean(x[:n]) (v0.2) */
void hpcs_reduce_mean(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* out = variance(x[:n]) - population variance using Welford's algorithm (v0.2) */
void hpcs_reduce_variance(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* out = std(x[:n]) - population standard deviation (v0.2) */
void hpcs_reduce_std(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* grouped sum: y[g] = sum of x[i] for all i with group_ids[i] == g */
void hpcs_group_reduce_sum(
    const double *x,
    int           n,
    const int    *group_ids,
    int           n_groups,
    double       *y,
    int          *status
);

/* grouped mean: y[g] = sum(x)/count; empty groups -> NaN */
void hpcs_group_reduce_mean(
    const double *x,
    int           n,
    const int    *group_ids,
    int           n_groups,
    double       *y,
    int          *status
);

/* grouped variance: population variance per group using Welford's algorithm (v0.2) */
void hpcs_group_reduce_variance(
    const double *x,
    int           n,
    const int    *group_ids,
    int           n_groups,
    double       *y,
    int          *status
);

/* --------------------------------------------------------------------- */
/* Array utilities (hpcs_core_utils)                                     */
/* --------------------------------------------------------------------- */

/* in-place missing fill (sentinel + optional NaN) */
void hpcs_fill_missing(
    double *x,
    int     n,
    double  missing_value,
    double  replacement,
    int     treat_nan_as_missing,
    int    *status
);

/* y[i] = mask[i] ? a[i] : b[i] */
void hpcs_where(
    const int    *mask,
    int           n,
    const double *a,
    const double *b,
    double       *y,
    int          *status
);

/* fill array with constant value */
void hpcs_fill_value(
    double *x,
    int     n,
    double  value,
    int    *status
);

/* copy src array to dst array */
void hpcs_copy(
    double       *dst,
    const double *src,
    int           n,
    int          *status
);

/* min-max normalization: y[i] = (x[i] - min) / (max - min) to [0, 1] range (v0.2) */
void hpcs_normalize_minmax(
    const double *x,
    int           n,
    double       *y,
    int          *status
);

/* forward fill: propagate last valid (non-NaN) value forward (v0.2) */
void hpcs_fill_forward(
    const double *x,
    int           n,
    double       *y,
    int          *status
);

/* backward fill: propagate next valid (non-NaN) value backward (v0.2) */
void hpcs_fill_backward(
    const double *x,
    int           n,
    double       *y,
    int          *status
);

/* detect anomalies: z-score based anomaly detection (v0.2) */
/* anomaly[i] = 1 if |z-score| > threshold, 0 otherwise */
void hpcs_detect_anomalies(
    const double *x,
    int           n,
    double        threshold,
    int          *anomaly,
    int          *status
);

/* --------------------------------------------------------------------- */
/* Parallel Kernels (hpcs_core_parallel) - v0.2                         */
/* --------------------------------------------------------------------- */

/* Parallel reduce sum (uses OpenMP for n >= 100000) */
void hpcs_reduce_sum_parallel(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* Parallel reduce min (uses OpenMP for n >= 100000) */
void hpcs_reduce_min_parallel(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* Parallel reduce max (uses OpenMP for n >= 100000) */
void hpcs_reduce_max_parallel(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* Parallel reduce mean (v0.2, uses OpenMP for n >= 100000) */
void hpcs_reduce_mean_parallel(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* Parallel reduce variance (v0.2, uses OpenMP for n >= 100000) */
void hpcs_reduce_variance_parallel(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* Parallel reduce std (v0.2, uses OpenMP for n >= 100000) */
void hpcs_reduce_std_parallel(
    const double *x,
    int           n,
    double       *out,
    int          *status
);

/* Parallel group reduce variance (v0.2)
 * NOTE: Internally delegates to serial version due to atomic contention.
 * Benchmarking showed 5-10x slowdown with parallel implementation.
 * Retained for API compatibility. Use hpcs_group_reduce_variance() directly. */
void hpcs_group_reduce_variance_parallel(
    const double *x,
    int           n,
    const int    *group_ids,
    int           n_groups,
    double       *y,
    int          *status
);

/* ============================================================================
 * v0.3: Robust Statistics & Data Quality
 * ============================================================================
 * These functions provide outlier-resistant statistical measures and data
 * quality operations. Includes median, MAD, quantiles, rolling robust stats,
 * clipping, winsorization, and robust z-scores.
 * ========================================================================== */

/* ----------------------------------------------------------------------------
 * A. Basic Robust Statistics (Global)
 * -------------------------------------------------------------------------- */

/* Median: Compute the median of a 1D array (average of two middle values for even n) */
void hpcs_median(
    const double *x,
    int           n,
    double       *median,
    int          *status
);

/* MAD: Median Absolute Deviation around the median (raw MAD, no scaling) */
void hpcs_mad(
    const double *x,
    int           n,
    double       *mad,
    int          *status
);

/* Quantile: Compute the q-th quantile (q in [0,1]) using Type 7 interpolation */
void hpcs_quantile(
    const double *x,
    int           n,
    double        q,
    double       *value,
    int          *status
);

/* ----------------------------------------------------------------------------
 * B. Rolling Robust Statistics
 * -------------------------------------------------------------------------- */

/* Rolling Median: Compute median within sliding window of specified length */
void hpcs_rolling_median(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* Rolling MAD: Compute MAD within sliding window */
void hpcs_rolling_mad(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* Rolling Z-Score: Compute rolling (x - mean) / std within sliding window (v0.7) */
void hpcs_rolling_zscore(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* Rolling Robust Z-Score: Compute rolling MAD-based z-score within sliding window (v0.7) */
void hpcs_rolling_robust_zscore(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* ----------------------------------------------------------------------------
 * C. Data Quality: Clipping and Winsorization
 * -------------------------------------------------------------------------- */

/* Clip: Clamp each element into [min_val, max_val] (in-place modification) */
void hpcs_clip(
    double       *x,
    int           n,
    double        min_val,
    double        max_val,
    int          *status
);

/* Winsorize by Quantiles: Clamp values to quantile-based bounds [q_low, q_high] */
void hpcs_winsorize_by_quantiles(
    double       *x,
    int           n,
    double        q_low,
    double        q_high,
    int          *status
);

/* ----------------------------------------------------------------------------
 * D. Robust Z-Score
 * -------------------------------------------------------------------------- */

/* Robust Z-Score: Compute (x - median) / (MAD * scale) for outlier detection
 * Default scale is 1.4826 to match normal distribution std. deviation.
 * Status = 2 if MAD is degenerate (≈0). */
void hpcs_robust_zscore(
    const double *x,
    int           n,
    double       *y,
    int          *status
);

/* ----------------------------------------------------------------------------
 * D2. Robust Anomaly Detection (v0.3+)
 * -------------------------------------------------------------------------- */

/* Robust Anomaly Detection: Detect outliers using median/MAD-based z-score
 * This method is resistant to outliers (avoids the "masking" problem).
 * Formula: z_robust = |x - median| / (MAD * 1.4826)
 * Returns: anomaly[i] = 1 if |z_robust| > threshold, 0 otherwise
 *
 * Recommended over classical z-score for real-world outlier detection. */
void hpcs_detect_anomalies_robust(
    const double *x,
    int           n,
    double        threshold,
    int          *anomaly,
    int          *status
);

/* Iterative Outlier Removal: Repeatedly detect and remove outliers
 * Iteratively applies robust anomaly detection until convergence or max_iter.
 * Returns cleaned data (non-outliers) and actual number of iterations.
 *
 * Arguments:
 *   x          - input array of length n
 *   n          - number of input elements
 *   threshold  - z-score threshold (e.g., 3.0)
 *   max_iter   - maximum iterations (e.g., 10)
 *   cleaned    - output: cleaned data (must be pre-allocated, size >= n)
 *   n_clean    - output: number of values in cleaned array
 *   iterations - output: actual iterations performed
 *   status     - output: status code */
void hpcs_remove_outliers_iterative(
    const double *x,
    int           n,
    double        threshold,
    int           max_iter,
    double       *cleaned,
    int          *n_clean,
    int          *iterations,
    int          *status
);

/* Rolling Anomaly Detection: Detect outliers using sliding window
 * For each position i >= window, computes robust z-score based on the
 * median and MAD of the window ending at i, then flags if |z| > threshold.
 * This provides adaptive outlier detection for non-stationary time series.
 *
 * For i < window, output is set to 0 (no detection possible).
 *
 * Arguments:
 *   x         - input time series of length n
 *   n         - number of elements
 *   window    - window length
 *   threshold - z-score threshold (e.g., 3.0)
 *   anomaly   - output: 1=anomaly, 0=normal
 *   status    - output: status code */
void hpcs_rolling_anomalies(
    const double *x,
    int           n,
    int           window,
    double        threshold,
    int          *anomaly,
    int          *status
);

/* ============================================================================
 * v0.3 OPTIMIZED: Parallel and Fast Implementations
 * ============================================================================
 * These optimized versions provide significant speedups for large arrays:
 * - Parallel versions: 3-4x faster for n >= 100K (OpenMP)
 * - Fast rolling: 20-40x faster rolling operations (C++ heap-based)
 * ========================================================================== */

/* ----------------------------------------------------------------------------
 * E. Parallel Robust Statistics (OpenMP)
 * -------------------------------------------------------------------------- */

/* Parallel Median: 3-4x faster for arrays >= 100K elements */
void hpcs_median_parallel(
    const double *x,
    int           n,
    double       *median,
    int          *status
);

/* Parallel MAD: 3-4x faster for arrays >= 100K elements */
void hpcs_mad_parallel(
    const double *x,
    int           n,
    double       *mad,
    int          *status
);

/* Parallel Quantile: 3-4x faster for arrays >= 100K elements */
void hpcs_quantile_parallel(
    const double *x,
    int           n,
    double        q,
    double       *value,
    int          *status
);

/* Parallel Robust Z-Score: 3-4x faster for arrays >= 100K elements */
void hpcs_robust_zscore_parallel(
    const double *x,
    int           n,
    double       *y,
    int          *status
);

/* ----------------------------------------------------------------------------
 * F. Fast Rolling Operations (C++ Heap-Based, O(n log w))
 * -------------------------------------------------------------------------- */

/* Fast Rolling Median: 20-40x faster than standard rolling_median
 * Uses balanced BST (std::multiset) for O(n log w) instead of O(n*w) */
void hpcs_rolling_median_fast(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* Fast Rolling MAD: 20-40x faster than standard rolling_mad
 * Uses balanced BST for efficient window management */
void hpcs_rolling_mad_fast(
    const double *x,
    int           n,
    int           window,
    double       *y,
    int          *status
);

/* ============================================================================
 * v0.4 GPU ACCELERATION CONTROL (Phase 1)
 * ============================================================================
 * GPU acceleration infrastructure for transparent hardware acceleration.
 *
 * Key Features:
 * - Optional GPU use (never breaks CPU-only workflows)
 * - Portable backend strategy (OpenMP target, CUDA, HIP)
 * - Automatic CPU fallback for small workloads
 * - Device detection and selection
 *
 * Acceleration Policies:
 *   HPCS_CPU_ONLY (0)       - Never use GPU, always execute on CPU
 *   HPCS_GPU_PREFERRED (1)  - Use GPU for large workloads, fallback to CPU (default)
 *   HPCS_GPU_ONLY (2)       - Only use GPU, fail if unavailable (status=2)
 *
 * Typical Usage:
 *   int count, status;
 *   hpcs_get_device_count(&count, &status);
 *   if (count > 0) {
 *       hpcs_set_device(0, &status);  // Select first GPU
 *       hpcs_set_accel_policy(HPCS_GPU_PREFERRED, &status);
 *   }
 *
 * Note: CPU-only builds expose these APIs but report 0 devices.
 * ========================================================================== */

/* ----------------------------------------------------------------------------
 * G. GPU Control APIs (v0.4 Phase 1)
 * -------------------------------------------------------------------------- */

/* Acceleration policy constants */
#define HPCS_CPU_ONLY       0
#define HPCS_GPU_PREFERRED  1
#define HPCS_GPU_ONLY       2

/* Set acceleration policy for GPU kernel execution
 *
 * Parameters:
 *   policy - HPCS_CPU_ONLY, HPCS_GPU_PREFERRED, or HPCS_GPU_ONLY
 *   status - 0=success, 1=invalid policy
 *
 * Thread Safety: Set once during initialization, not concurrently */
void hpcs_set_accel_policy(
    int  policy,
    int *status
);

/* Get current acceleration policy
 *
 * Parameters:
 *   policy - Output: current policy (0, 1, or 2)
 *   status - 0=success */
void hpcs_get_accel_policy(
    int *policy,
    int *status
);

/* Query number of available GPU devices
 *
 * Returns 0 for CPU-only builds or when no GPU hardware is available.
 * Uses OpenMP target, CUDA, or HIP runtime depending on compile flags.
 *
 * Parameters:
 *   count  - Output: number of GPU devices (0 if none)
 *   status - 0=success, 2=runtime error
 *
 * Performance: O(1) - Single runtime query, result cached */
void hpcs_get_device_count(
    int *count,
    int *status
);

/* Select a specific GPU device for kernel execution
 *
 * Device IDs are 0-indexed. Must be in range [0, count-1].
 * CPU-only builds only accept device_id=0.
 *
 * Parameters:
 *   device_id - Device to select (0 to count-1)
 *   status    - 0=success, 1=invalid device_id, 2=runtime error
 *
 * Thread Safety: Set once during initialization or use per-thread management */
void hpcs_set_device(
    int  device_id,
    int *status
);

/* Get currently selected GPU device ID
 *
 * Parameters:
 *   device_id - Output: current device ID (default=0)
 *   status    - 0=success */
void hpcs_get_device(
    int *device_id,
    int *status
);

/* ----------------------------------------------------------------------------
 * H. GPU Acceleration Internal APIs (v0.4 Phase 2)
 * --------------------------------------------------------------------------
 *
 * These functions provide low-level GPU acceleration infrastructure.
 * Most users should use the standard kernel APIs (hpcs_median, hpcs_mad, etc.)
 * which automatically dispatch to GPU when beneficial.
 *
 * Phase 2 Scope:
 *   - Backend initialization
 *   - Memory management (host-device transfers)
 *   - HIGH PRIORITY kernel wrappers (median, MAD, rolling_median)
 *   - Example reduction wrapper (reduce_sum)
 *
 * CPU-Only Behavior:
 *   - All functions succeed and delegate to CPU implementations
 *   - Memory copies are no-ops (device_ptr = host_ptr)
 * -------------------------------------------------------------------------- */

/* Initialize GPU backend for accelerated execution
 *
 * Must be called before any GPU kernel execution.
 * Idempotent: Multiple calls are safe (returns immediately if already initialized).
 *
 * Parameters:
 *   status - 0=success (backend ready or CPU-only), 2=runtime error
 *
 * Thread Safety: Call once during program initialization */
void hpcs_accel_init(
    int *status
);

/* Copy data from host to device memory
 *
 * Allocates device memory and copies data from host array to device.
 * CPU-only builds: Returns device_ptr = host_ptr (no actual copy).
 *
 * Parameters:
 *   host_ptr   - Source data on host
 *   n          - Number of elements
 *   device_ptr - Output: pointer to device memory
 *   status     - 0=success, 1=invalid args, 2=allocation/copy failed */
void hpcs_accel_copy_to_device(
    const double *host_ptr,
    int           n,
    void        **device_ptr,
    int          *status
);

/* Copy data from device to host memory
 *
 * Copies data from device array back to host.
 * CPU-only builds: No-op (data already on host).
 *
 * Parameters:
 *   device_ptr - Source data on device
 *   n          - Number of elements
 *   host_ptr   - Output: destination on host
 *   status     - 0=success, 1=invalid args, 2=copy failed */
void hpcs_accel_copy_from_device(
    const void   *device_ptr,
    int           n,
    double       *host_ptr,
    int          *status
);

/* Free device memory allocated by hpcs_accel_copy_to_device
 *
 * Deallocates device memory and removes allocation from tracking table.
 * Must be called for all allocations to prevent memory leaks.
 * CPU-only builds: No-op (memory managed by caller).
 *
 * Parameters:
 *   device_ptr - Device memory pointer to free
 *   status     - 0=success, 1=invalid args, 2=not found */
void hpcs_accel_free_device(
    void *device_ptr,
    int  *status
);

/* GPU-accelerated median computation (HIGH PRIORITY)
 *
 * Computes median of array on device.
 * Phase 2: Falls back to CPU (Phase 3 will add GPU kernel).
 * Benchmark: 366ms for 5M elements on CPU (18x slower than reductions).
 *
 * Parameters:
 *   device_ptr - Input array on device
 *   n          - Number of elements
 *   median_val - Output: median value
 *   status     - 0=success, 1=invalid args */
void hpcs_accel_median(
    const void *device_ptr,
    int         n,
    double     *median_val,
    int        *status
);

/* GPU-accelerated MAD computation (HIGH PRIORITY)
 *
 * Computes Median Absolute Deviation on device.
 * Phase 2: Falls back to CPU (Phase 3 will add GPU kernel).
 * Benchmark: Similar performance to median (slow on CPU).
 *
 * Parameters:
 *   device_ptr - Input array on device
 *   n          - Number of elements
 *   mad_val    - Output: MAD value
 *   status     - 0=success, 1=invalid args */
void hpcs_accel_mad(
    const void *device_ptr,
    int         n,
    double     *mad_val,
    int        *status
);

/* GPU-accelerated rolling median (HIGH PRIORITY)
 *
 * Computes rolling median with specified window size.
 * Phase 2: Falls back to CPU (Phase 3 will add GPU kernel).
 * Benchmark: 8.6s for 1M elements with window=200 (VERY EXPENSIVE).
 *
 * Parameters:
 *   device_ptr    - Input array on device
 *   n             - Number of elements
 *   window        - Window size
 *   device_output - Output: pointer to result array on device
 *   status        - 0=success, 1=invalid args */
void hpcs_accel_rolling_median(
    const void *device_ptr,
    int         n,
    int         window,
    void      **device_output,
    int        *status
);

/* GPU-accelerated reduction sum (example wrapper)
 *
 * Computes sum of array on device.
 * Included for spec compliance - reductions are already fast on CPU (20ms for 5M).
 *
 * Parameters:
 *   device_ptr - Input array on device
 *   n          - Number of elements
 *   result     - Output: sum value
 *   status     - 0=success, 1=invalid args */
void hpcs_accel_reduce_sum(
    const void *device_ptr,
    int         n,
    double     *result,
    int        *status
);

/* GPU-accelerated prefix sum (inclusive scan)
 *
 * Computes inclusive prefix sum of array on device.
 * Uses Blelloch scan algorithm for efficient parallel computation.
 *
 * Parameters:
 *   device_input_ptr  - Input array on device
 *   n                 - Number of elements
 *   device_output_ptr - Output array on device (prefix sum)
 *   status            - 0=success, 1=invalid args */
void hpcs_accel_prefix_sum(
    const void *device_input_ptr,
    int         n,
    void       *device_output_ptr,
    int        *status
);

/* ============================================================================
 * v0.6 SIMD INFORMATION & DISPATCH
 * ============================================================================
 * Runtime SIMD capability detection and instruction set information.
 * Functions to query the active SIMD ISA and vector width.
 * ========================================================================== */

/* Get name of active SIMD instruction set
 *
 * Returns a string like "AVX2", "AVX", "SSE2", "NEON", or "Scalar".
 *
 * Returns:
 *   Pointer to static string (do not free)
 */
const char* hpcs_get_simd_name(void);

/* Get SIMD vector width in bytes
 *
 * Returns:
 *   32 for AVX2, 16 for SSE2/NEON, 8 for scalar
 */
int hpcs_get_simd_width_bytes(void);

/* Get SIMD vector width in number of doubles
 *
 * Returns:
 *   4 for AVX2, 2 for SSE2/NEON, 1 for scalar
 */
int hpcs_get_simd_width_doubles(void);

/* Print current SIMD configuration to stdout
 *
 * Prints active ISA and vector width information
 */
void hpcs_print_simd_status(void);

/* Initialize SIMD dispatch system and register all SIMD kernels
 *
 * Must be called before using any SIMD-accelerated functions.
 * Safe to call multiple times (subsequent calls are ignored).
 */
void hpcs_simd_reductions_init(void);
void hpcs_rolling_simd_init(void);
void hpcs_zscore_simd_init(void);

/* ============================================================================
 * v0.7: 2D AXIS OPERATIONS (C SIMD)
 * ============================================================================
 * SIMD-accelerated 2D array operations along specified axes.
 * These complement the Fortran axis operations with C/SIMD implementations.
 * ========================================================================== */

/* Axis-0 Min: Compute minimum along rows (per column) of 2D array (v0.7)
 *
 * For a column-major 2D array x[n,m], computes y[j] = min(x[:,j]) for each column.
 *
 * Parameters:
 *   x      - Input 2D array in column-major order
 *   n      - Number of rows
 *   m      - Number of columns
 *   y      - Output: min per column [m]
 *   status - 0=success, 1=invalid args
 */
void hpcs_reduce_min_axis0_simd(
    const double *x,
    int           n,
    int           m,
    double       *y,
    int          *status
);

/* Axis-0 Max: Compute maximum along rows (per column) of 2D array (v0.7)
 *
 * For a column-major 2D array x[n,m], computes y[j] = max(x[:,j]) for each column.
 *
 * Parameters:
 *   x      - Input 2D array in column-major order
 *   n      - Number of rows
 *   m      - Number of columns
 *   y      - Output: max per column [m]
 *   status - 0=success, 1=invalid args
 */
void hpcs_reduce_max_axis0_simd(
    const double *x,
    int           n,
    int           m,
    double       *y,
    int          *status
);

/* ============================================================================
 * v0.7: AUTO-TUNING CALIBRATION (v0.5)
 * ============================================================================
 * Benchmark-based auto-tuning system for optimal parallelization thresholds.
 * Measures performance at various array sizes to find serial/parallel crossover
 * points for different operation classes.
 * ========================================================================== */

/* Run full calibration benchmark
 *
 * Performs comprehensive benchmarking across all operation classes:
 * - Simple reductions (sum, mean, min, max)
 * - Rolling window operations
 * - Robust statistics (median, MAD)
 * - Anomaly detection
 *
 * Determines optimal parallelization thresholds and stores in global config.
 * Takes 30-60 seconds depending on CPU.
 *
 * Parameters:
 *   status - 0=success, 1=error
 */
void hpcs_calibrate(int *status);

/* Run quick calibration benchmark (faster, less accurate)
 *
 * Performs reduced benchmarking with fewer sizes and trials.
 * Suitable for quick testing or resource-constrained environments.
 * Takes 5-10 seconds.
 *
 * Parameters:
 *   status - 0=success, 1=error
 */
void hpcs_calibrate_quick(int *status);

/* Save calibration configuration to file
 *
 * Exports current tuning thresholds to JSON configuration file.
 * Typical location: $HOME/.hpcs/config.json
 *
 * Parameters:
 *   path   - Configuration file path
 *   status - 0=success, 1=error (e.g., permission denied)
 */
void hpcs_save_config(const char *path, int *status);

/* Load calibration configuration from file
 *
 * Imports tuning thresholds from JSON configuration file.
 * Applies thresholds to all subsequent operations.
 *
 * Parameters:
 *   path   - Configuration file path
 *   status - 0=success, 1=error (e.g., file not found)
 */
void hpcs_load_config(const char *path, int *status);

#ifdef __cplusplus
}
#endif

#endif /* HPCS_CORE_H */
