#ifndef HPCS_CORE_H
#define HPCS_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * HPCSeries Core v0.1 – C Interface
 *
 * All kernels operate on 1D contiguous arrays of double.
 * - Arrays are passed as raw pointers.
 * - Lengths and group IDs are 32-bit ints (matching Fortran c_int).
 * - Kernels return an int status code:
 *
 *   0  = success
 *   1  = invalid arguments (e.g., n <= 0, window <= 0, n_groups <= 0)
 *   2  = numeric failure (e.g., zero std-dev in z-score)
 *  >=100 reserved for future use.
 */

/* Status code constants (optional convenience macros) */
#define HPCS_STATUS_SUCCESS        0
#define HPCS_STATUS_INVALID_ARGS   1
#define HPCS_STATUS_NUMERIC_FAIL   2

/* ------------------------- Rolling operations (1D) ------------------------- */

/*
 * Rolling sum over x[0..n-1] with window length `window`.
 *
 * Inputs:
 *   x      - pointer to input array (length >= n)
 *   n      - number of elements
 *   window - window length (> 0)
 *
 * Output:
 *   y      - pointer to output array (length >= n)
 *
 * Returns:
 *   int status code (0 on success).
 *
 * For i < window, the sum is over the truncated prefix.
 */
int hpcs_rolling_sum(const double *x, int n, int window, double *y);

/*
 * Rolling mean over x[0..n-1] with window length `window`.
 * Same semantics as hpcs_rolling_sum, but divides by the
 * effective window length (min(i+1, window)) at each position.
 */
int hpcs_rolling_mean(const double *x, int n, int window, double *y);

/* --------------------------- Grouped reductions --------------------------- */

/*
 * Grouped sum.
 *
 * Inputs:
 *   x         - input array (length >= n)
 *   n         - number of elements
 *   group_ids - group identifier per element (length >= n)
 *   n_groups  - number of distinct groups (>= 0)
 *
 * Output:
 *   y         - per-group sums (length >= n_groups)
 *
 * Notes:
 *   - Valid group IDs are in [0, n_groups-1]; others are ignored.
 *   - Groups with no elements remain 0.0.
 */
int hpcs_group_reduce_sum(const double *x,
                          int n,
                          const int *group_ids,
                          int n_groups,
                          double *y);

/*
 * Grouped mean.
 *
 * Same inputs as hpcs_group_reduce_sum.
 *
 * Output:
 *   y - per-group means (length >= n_groups)
 *
 * Notes:
 *   - Groups with no elements produce NaN (implementation-defined NaN).
 */
int hpcs_group_reduce_mean(const double *x,
                           int n,
                           const int *group_ids,
                           int n_groups,
                           double *y);

/* ---------------------------- Simple reductions --------------------------- */

/*
 * Sum of all elements in x[0..n-1].
 *
 * Input:
 *   x  - input array (length >= n)
 *   n  - number of elements
 *
 * Output:
 *   out - pointer to scalar result
 */
int hpcs_reduce_sum(const double *x, int n, double *out);

/*
 * Minimum of all elements in x[0..n-1].
 *
 * For n <= 0, returns a +inf sentinel (implementation-defined)
 * and may set a non-zero status code.
 */
int hpcs_reduce_min(const double *x, int n, double *out);

/*
 * Maximum of all elements in x[0..n-1].
 *
 * For n <= 0, returns a -inf sentinel (implementation-defined)
 * and may set a non-zero status code.
 */
int hpcs_reduce_max(const double *x, int n, double *out);

/* --------------------------- Statistical transforms ----------------------- */

/*
 * Z-score transform:
 *
 *   y[i] = (x[i] - mean(x)) / std(x)
 *
 * Inputs:
 *   x - input array (length >= n)
 *   n - number of elements
 *
 * Output:
 *   y - output array (length >= n)
 *
 * Notes:
 *   - If std == 0, status = HPCS_STATUS_NUMERIC_FAIL (2) and implementation
 *     may set all y[i] = 0.0.
 */
int hpcs_zscore(const double *x, int n, double *y);

/* ----------------------------- Array utilities ---------------------------- */

/*
 * In-place fill: x[i] = value for i=0..n-1.
 *
 * This routine does not return a status code (always succeeds
 * for valid pointers and n >= 0).
 */
void hpcs_fill_value(double *x, int n, double value);

/*
 * Copy array: y[i] = x[i] for i=0..n-1.
 *
 * Inputs:
 *   x - source array (length >= n)
 *   n - number of elements
 *
 * Output:
 *   y - destination array (length >= n)
 */
void hpcs_copy(const double *x, int n, double *y);

#ifdef __cplusplus
}
#endif

#endif /* HPCS_CORE_H */
