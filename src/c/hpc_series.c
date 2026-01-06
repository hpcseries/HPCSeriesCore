/*
 * Implementation of simple rolling and reduction operations for
 * benchmarking purposes. These functions deliberately contain no
 * parallelisation so that their serial performance can be measured
 * against OpenMP implementations later on. The algorithms here are
 * straightforward and prioritise clarity over optimality.
 */

#include "../include/hpc_series.h"

double reduce_sum(const double *input, size_t n)
{
    /* Multi-accumulator strategy with 8 independent accumulators
     * to break data dependency chains and enable better CPU pipelining.
     * Each accumulator can progress independently, hiding memory latency
     * and allowing the compiler to vectorize more aggressively. */
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    double sum4 = 0.0, sum5 = 0.0, sum6 = 0.0, sum7 = 0.0;

    /* Process 8 elements per iteration for maximum instruction-level parallelism */
    size_t i = 0;
    size_t n_vec = (n / 8) * 8;  /* Round down to nearest multiple of 8 */

    for (i = 0; i < n_vec; i += 8)
    {
        sum0 += input[i + 0];
        sum1 += input[i + 1];
        sum2 += input[i + 2];
        sum3 += input[i + 3];
        sum4 += input[i + 4];
        sum5 += input[i + 5];
        sum6 += input[i + 6];
        sum7 += input[i + 7];
    }

    /* Handle remaining elements (0-7) that don't fit in groups of 8 */
    double remainder = 0.0;
    for (; i < n; ++i)
    {
        remainder += input[i];
    }

    /* Combine all accumulators - compiler can optimize this into a tree reduction */
    return (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7) + remainder;
}

void rolling_sum(const double *input, double *output, size_t n, size_t window)
{
    if (n == 0)
    {
        return;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += input[i];
        /* Once the window has reached its full size, drop the oldest element */
        if (i >= window)
        {
            sum -= input[i - window];
        }
        output[i] = sum;
    }
}

void rolling_mean(const double *input, double *output, size_t n, size_t window)
{
    if (n == 0)
    {
        return;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += input[i];
        if (i >= window)
        {
            sum -= input[i - window];
            /* For indices >= window the divisor is the full window */
            output[i] = sum / (double)window;
        }
        else
        {
            /* For initial positions use the number of elements seen so far */
            output[i] = sum / (double)(i + 1);
        }
    }
}
