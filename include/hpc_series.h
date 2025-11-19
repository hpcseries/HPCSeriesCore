#ifndef HPC_SERIES_H
#define HPC_SERIES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Simplified C API for benchmarking HPCSeries kernels
 *
 * These wrappers call the full hpcs_core Fortran API but provide simpler
 * function signatures for benchmarking purposes (no explicit status handling).
 * Errors are silently ignored to simplify benchmark code.
 */

/**
 * Rolling sum: y[i] = sum of last 'window' elements up to i
 * @param x Input array (length n)
 * @param y Output array (length n)
 * @param n Array length
 * @param window Window size
 */
void rolling_sum(const double *x, double *y, size_t n, size_t window);

/**
 * Rolling mean: y[i] = mean of last 'window' elements up to i
 * @param x Input array (length n)
 * @param y Output array (length n)
 * @param n Array length
 * @param window Window size
 */
void rolling_mean(const double *x, double *y, size_t n, size_t window);

/**
 * Reduce sum: compute total sum of array
 * @param x Input array (length n)
 * @param n Array length
 * @return Sum of all elements
 */
double reduce_sum(const double *x, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* HPC_SERIES_H */
