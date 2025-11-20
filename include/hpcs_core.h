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

/* Parallel group reduce variance (v0.2, uses OpenMP for n >= 100000) */
void hpcs_group_reduce_variance_parallel(
    const double *x,
    int           n,
    const int    *group_ids,
    int           n_groups,
    double       *y,
    int          *status
);

#ifdef __cplusplus
}
#endif

#endif /* HPCS_CORE_H */
