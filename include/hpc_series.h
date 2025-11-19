#ifndef HPC_SERIES_H
#define HPC_SERIES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void rolling_sum(const double *input, double *output, size_t n, size_t window);
void rolling_mean(const double *input, double *output, size_t n, size_t window);
double reduce_sum(const double *input, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* HPC_SERIES_H */
