#include <stdio.h>
#include <math.h>
#include "../include/hpcs_core.h"

/* Simple helper to print an array */
static void print_array(const char *label, const double *x, int n) {
    printf("%s:", label);
    for (int i = 0; i < n; ++i) {
        printf(" %g", x[i]);
    }
    printf("\n");
}

/* Approximate check for doubles */
static int approx_equal(double a, double b, double tol) {
    double diff = a - b;
    if (diff < 0) diff = -diff;
    return diff <= tol;
}

int main(void) {
    int status;

    /* --------- Test data --------- */
    const int n = 5;
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[5];

    /* ----------------- Rolling sum & mean ----------------- */
    int window = 3;

    status = hpcs_rolling_sum(x, n, window, y);
    printf("hpcs_rolling_sum status = %d\n", status);
    print_array("rolling_sum", y, n);
    /* Expected: [1, 3, 6, 9, 12] */

    status = hpcs_rolling_mean(x, n, window, y);
    printf("hpcs_rolling_mean status = %d\n", status);
    print_array("rolling_mean", y, n);
    /* Expected: [1, 1.5, 2, 3, 4] */

    /* ----------------- Grouped reductions ----------------- */
    /* Two groups: 0 and 1 */
    int group_ids[5] = {0, 1, 0, 1, 0};
    int n_groups = 2;
    double gsum[2];
    double gmean[2];

    status = hpcs_group_reduce_sum(x, n, group_ids, n_groups, gsum);
    printf("hpcs_group_reduce_sum status = %d\n", status);
    print_array("group_sum", gsum, n_groups);
    /* group 0: 1+3+5 = 9, group 1: 2+4 = 6 */

    status = hpcs_group_reduce_mean(x, n, group_ids, n_groups, gmean);
    printf("hpcs_group_reduce_mean status = %d\n", status);
    print_array("group_mean", gmean, n_groups);
    /* group 0 mean: 9/3=3, group 1 mean: 6/2=3 */

    /* ----------------- Simple reductions ------------------ */
    double sum, minv, maxv;

    status = hpcs_reduce_sum(x, n, &sum);
    printf("hpcs_reduce_sum status = %d, sum = %g\n", status, sum);
    /* Expected: 15 */

    status = hpcs_reduce_min(x, n, &minv);
    printf("hpcs_reduce_min status = %d, min = %g\n", status, minv);
    /* Expected: 1 */

    status = hpcs_reduce_max(x, n, &maxv);
    printf("hpcs_reduce_max status = %d, max = %g\n", status, maxv);
    /* Expected: 5 */

    /* ----------------- Z-score transform ------------------ */
    status = hpcs_zscore(x, n, y);
    printf("hpcs_zscore status = %d\n", status);
    print_array("zscore", y, n);

    /* Quick sanity check: mean ~ 0, std ~ 1 */
    double zsum = 0.0;
    for (int i = 0; i < n; ++i) zsum += y[i];
    double zmean = zsum / n;

    double zvar = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = y[i] - zmean;
        zvar += d * d;
    }
    zvar /= n;
    double zstd = sqrt(zvar);

    printf("zscore mean ~ %g, std ~ %g\n", zmean, zstd);
    if (!approx_equal(zmean, 0.0, 1e-12) || !approx_equal(zstd, 1.0, 1e-12)) {
        printf("WARNING: z-score sanity check failed (tolerance 1e-12)\n");
    }

    /* ------------------ Array utilities ------------------- */
    double buf[4];
    hpcs_fill_value(buf, 4, 42.0);
    print_array("fill_value(42)", buf, 4);

    double src[3] = {10.0, 20.0, 30.0};
    double dst[3] = {0.0, 0.0, 0.0};
    hpcs_copy(src, 3, dst);
    print_array("copy src->dst", dst, 3);

    /* Basic success/failure exit code */
    if (status != HPCS_STATUS_SUCCESS) {
        /* Last status from zscore; real test harness could track all */
        return 1;
    }

    return 0;
}
