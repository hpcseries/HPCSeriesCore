/**
 * @file hpcs_core.h
 * @brief HPC Series Core Library - C API
 *
 * High-performance computational kernels for time series analysis
 * and statistical operations. Implemented in Fortran with C interoperability.
 *
 * @version 0.1.0
 */

#ifndef HPCS_CORE_H
#define HPCS_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* =============================================================================
 * Rolling Window Operations (1D)
 * ============================================================================= */

/**
 * @brief Compute rolling mean over a sliding window
 * @param input Input array
 * @param n Array length
 * @param window Window size
 * @param output Output array (same size as input)
 */
void hpcs_rolling_mean(const double* input, size_t n, size_t window, double* output);

/**
 * @brief Compute rolling standard deviation
 * @param input Input array
 * @param n Array length
 * @param window Window size
 * @param output Output array
 */
void hpcs_rolling_std(const double* input, size_t n, size_t window, double* output);

/**
 * @brief Compute rolling minimum
 * @param input Input array
 * @param n Array length
 * @param window Window size
 * @param output Output array
 */
void hpcs_rolling_min(const double* input, size_t n, size_t window, double* output);

/**
 * @brief Compute rolling maximum
 * @param input Input array
 * @param n Array length
 * @param window Window size
 * @param output Output array
 */
void hpcs_rolling_max(const double* input, size_t n, size_t window, double* output);

/**
 * @brief Compute rolling sum
 * @param input Input array
 * @param n Array length
 * @param window Window size
 * @param output Output array
 */
void hpcs_rolling_sum(const double* input, size_t n, size_t window, double* output);

/* =============================================================================
 * Statistical Transformations
 * ============================================================================= */

/**
 * @brief Compute z-score normalization
 * @param input Input array
 * @param n Array length
 * @param output Output array (normalized values)
 */
void hpcs_zscore(const double* input, size_t n, double* output);

/**
 * @brief Compute rank transformation
 * @param input Input array
 * @param n Array length
 * @param output Output array (ranks)
 */
void hpcs_rank(const double* input, size_t n, double* output);

/* =============================================================================
 * Reductions
 * ============================================================================= */

/**
 * @brief Compute simple mean
 * @param input Input array
 * @param n Array length
 * @return Mean value
 */
double hpcs_mean(const double* input, size_t n);

/**
 * @brief Compute simple sum
 * @param input Input array
 * @param n Array length
 * @return Sum value
 */
double hpcs_sum(const double* input, size_t n);

/**
 * @brief Compute standard deviation
 * @param input Input array
 * @param n Array length
 * @return Standard deviation
 */
double hpcs_std(const double* input, size_t n);

/**
 * @brief Group-based mean reduction
 * @param input Input array
 * @param n Array length
 * @param groups Group labels (integer array)
 * @param num_groups Number of unique groups
 * @param output Output array (size = num_groups)
 */
void hpcs_groupby_mean(const double* input, size_t n,
                       const int* groups, size_t num_groups, double* output);

/**
 * @brief Group-based sum reduction
 * @param input Input array
 * @param n Array length
 * @param groups Group labels (integer array)
 * @param num_groups Number of unique groups
 * @param output Output array (size = num_groups)
 */
void hpcs_groupby_sum(const double* input, size_t n,
                      const int* groups, size_t num_groups, double* output);

/* =============================================================================
 * Version Information
 * ============================================================================= */

/**
 * @brief Get library version string
 * @return Version string (e.g., "0.1.0")
 */
const char* hpcs_version(void);

#ifdef __cplusplus
}
#endif

#endif /* HPCS_CORE_H */
