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

#ifdef __cplusplus
}
#endif

#endif /* HPCS_CORE_H */
