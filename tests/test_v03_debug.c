#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/hpcs_core.h"

int main() {
    printf("Debug: Testing hpcs_robust_zscore\n\n");

    /* Data with one clear outlier */
    double x[] = {10.0, 11.0, 10.5, 10.2, 9.8, 10.3, 9.9, 50.0};
    double y[8];
    int status;

    /* First, let's check median and MAD separately */
    double median, mad;
    int st;

    hpcs_median(x, 8, &median, &st);
    printf("Median: %.6f (status=%d)\n", median, st);

    hpcs_mad(x, 8, &mad, &st);
    printf("MAD: %.6f (status=%d)\n", mad, st);
    printf("MAD * 1.4826: %.6f\n\n", mad * 1.4826);

    /* Now compute robust z-scores */
    hpcs_robust_zscore(x, 8, y, &status);
    printf("Robust Z-score status: %d\n\n", status);

    printf("Data and Z-scores:\n");
    printf("Index  Value    Z-score    |Z|\n");
    printf("-----  -------  ---------  ------\n");
    for (int i = 0; i < 8; i++) {
        printf("  %d    %6.2f   %9.4f  %6.4f  %s\n",
               i, x[i], y[i], fabs(y[i]),
               (i < 7 && fabs(y[i]) >= 3.0) ? "<-- FAIL" : "");
    }

    printf("\nOutlier check (|z[7]| > 10.0): %s\n",
           fabs(y[7]) > 10.0 ? "PASS" : "FAIL");

    printf("Normal values check (|z[0..6]| < 3.0):\n");
    int failures = 0;
    for (int i = 0; i < 7; i++) {
        if (fabs(y[i]) >= 3.0) {
            printf("  FAIL at index %d: |z| = %.4f\n", i, fabs(y[i]));
            failures++;
        }
    }
    if (failures == 0) {
        printf("  All normal values PASS\n");
    }

    return failures > 0 ? 1 : 0;
}
