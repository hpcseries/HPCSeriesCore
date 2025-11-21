/*
 * Simple debug test for fill_forward
 */

#include "../include/hpcs_core.h"
#include <stdio.h>
#include <math.h>

int main(void)
{
    printf("Debug Test: fill_forward\n");
    printf("========================\n\n");

    double x[] = {1.0, NAN, NAN, 2.0, NAN, 3.0};
    double y[6];
    int status;
    int n = 6;

    printf("Input array:\n");
    for (int i = 0; i < n; i++) {
        if (isnan(x[i])) {
            printf("  x[%d] = NaN (isnan=true, x==x=%d)\n", i, x[i] == x[i]);
        } else {
            printf("  x[%d] = %.4f\n", i, x[i]);
        }
    }

    printf("\nCalling hpcs_fill_forward...\n");
    hpcs_fill_forward(x, n, y, &status);
    printf("Status: %d\n\n", status);

    printf("Output array:\n");
    for (int i = 0; i < n; i++) {
        if (isnan(y[i])) {
            printf("  y[%d] = NaN (isnan=true)\n", i);
        } else {
            printf("  y[%d] = %.4f\n", i, y[i]);
        }
    }

    printf("\nExpected: [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]\n");

    return 0;
}
