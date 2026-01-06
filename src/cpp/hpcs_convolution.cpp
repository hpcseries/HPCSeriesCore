/*
 * Finite Impulse Response (FIR) Filters for HPCSeries Core v0.8.0
 *
 * This module provides optimized 1D convolution for small kernels:
 *   - hpcs_convolve_valid: Valid-mode convolution (no padding)
 *
 * Optimization strategy:
 *   - Direct convolution (no FFT) for small kernels (m = 3-15)
 *   - Template specializations for common kernel sizes (unrolled inner loops)
 *   - OpenMP parallelization over output indices
 *   - SIMD-friendly memory access patterns
 *   - Execution modes: SAFE (default), FAST (skip validation), DETERMINISTIC (no OpenMP)
 *
 * Algorithm:
 *   y[i] = Σ_{j=0}^{m-1} x[i+j] * k[j]
 *   Output length = n - m + 1
 *
 * References:
 *   - Smith, S. W. (1997). "The Scientist and Engineer's Guide to Digital Signal Processing"
 *   - Oppenheim, A. V. (1999). "Discrete-Time Signal Processing"
 *
 * Author: HPCSeries Core Library
 * Version: 0.8.0
 * Date: 2025-12-23
 */

#include <cstring>  // for memcpy
#include <cmath>    // for isnan
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// Execution mode constants (match hpcs_constants.f90)
#define HPCS_MODE_SAFE          0
#define HPCS_MODE_FAST          1
#define HPCS_MODE_DETERMINISTIC 2
#define HPCS_MODE_USE_GLOBAL   -1

// Error codes
#define HPCS_SUCCESS            0
#define HPCS_ERR_INVALID_ARGS   1

// Execution mode API (from hpcs_core_execution_mode.f90)
extern "C" {
    void hpcs_get_execution_mode(int* mode, int* status);
}

// Helper to get effective execution mode
namespace {
inline int get_effective_mode(int mode) {
    if (mode == HPCS_MODE_USE_GLOBAL) {
        int global_mode;
        int status;
        hpcs_get_execution_mode(&global_mode, &status);
        return (status == HPCS_SUCCESS) ? global_mode : HPCS_MODE_SAFE;
    }
    return mode;
}
} // namespace

// Anonymous namespace for template specializations
namespace {

/*
 * Generic convolution kernel (fallback for arbitrary sizes).
 * Uses straightforward dot product - compiler may auto-vectorize.
 */
inline double convolve_generic(const double* x, const double* k, int m) {
    double sum = 0.0;
    for (int j = 0; j < m; ++j) {
        sum += x[j] * k[j];
    }
    return sum;
}

/*
 * Template specializations for common kernel sizes.
 * Fully unrolled for compiler optimization (FMA instructions).
 */

// Size 3: common for simple smoothing (eg. [0.25, 0.5, 0.25])
template<int M>
inline double convolve_unrolled(const double* x, const double* k);

template<>
inline double convolve_unrolled<3>(const double* x, const double* k) {
    return x[0] * k[0] + x[1] * k[1] + x[2] * k[2];
}

// Size 5: common for Gaussian-like filters
template<>
inline double convolve_unrolled<5>(const double* x, const double* k) {
    return x[0] * k[0] + x[1] * k[1] + x[2] * k[2] + x[3] * k[3] + x[4] * k[4];
}

// Size 7
template<>
inline double convolve_unrolled<7>(const double* x, const double* k) {
    return x[0] * k[0] + x[1] * k[1] + x[2] * k[2] + x[3] * k[3] +
           x[4] * k[4] + x[5] * k[5] + x[6] * k[6];
}

// Size 9
template<>
inline double convolve_unrolled<9>(const double* x, const double* k) {
    return x[0] * k[0] + x[1] * k[1] + x[2] * k[2] + x[3] * k[3] +
           x[4] * k[4] + x[5] * k[5] + x[6] * k[6] + x[7] * k[7] +
           x[8] * k[8];
}

// Size 11
template<>
inline double convolve_unrolled<11>(const double* x, const double* k) {
    return x[0] * k[0] + x[1] * k[1] + x[2] * k[2] + x[3] * k[3] +
           x[4] * k[4] + x[5] * k[5] + x[6] * k[6] + x[7] * k[7] +
           x[8] * k[8] + x[9] * k[9] + x[10] * k[10];
}

// Size 13
template<>
inline double convolve_unrolled<13>(const double* x, const double* k) {
    return x[0] * k[0] + x[1] * k[1] + x[2] * k[2] + x[3] * k[3] +
           x[4] * k[4] + x[5] * k[5] + x[6] * k[6] + x[7] * k[7] +
           x[8] * k[8] + x[9] * k[9] + x[10] * k[10] + x[11] * k[11] +
           x[12] * k[12];
}

// Size 15
template<>
inline double convolve_unrolled<15>(const double* x, const double* k) {
    return x[0] * k[0] + x[1] * k[1] + x[2] * k[2] + x[3] * k[3] +
           x[4] * k[4] + x[5] * k[5] + x[6] * k[6] + x[7] * k[7] +
           x[8] * k[8] + x[9] * k[9] + x[10] * k[10] + x[11] * k[11] +
           x[12] * k[12] + x[13] * k[13] + x[14] * k[14];
}

/*
 * Dispatch function: selects unrolled version if available, else generic.
 */
inline double convolve_dispatch(const double* x, const double* k, int m) {
    switch (m) {
        case 3:  return convolve_unrolled<3>(x, k);
        case 5:  return convolve_unrolled<5>(x, k);
        case 7:  return convolve_unrolled<7>(x, k);
        case 9:  return convolve_unrolled<9>(x, k);
        case 11: return convolve_unrolled<11>(x, k);
        case 13: return convolve_unrolled<13>(x, k);
        case 15: return convolve_unrolled<15>(x, k);
        default: return convolve_generic(x, k, m);
    }
}

/*
 * hpcs_convolve_valid_safe - SAFE mode implementation
 *
 * Full validation and OpenMP parallelization (current behavior).
 */
void hpcs_convolve_valid_safe(
    const double* x,
    int n,
    const double* k,
    int m,
    double* y,
    int* status
) {
    // Validate arguments
    if (n <= 0 || m <= 0 || m > n ||
        x == nullptr || k == nullptr || y == nullptr || status == nullptr) {
        if (status != nullptr) *status = HPCS_ERR_INVALID_ARGS;
        return;
    }

    const int out_n = n - m + 1;

    // Parallel convolution over output indices
    // Each output y[i] is independent - trivially parallelizable
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(out_n >= 1000)
#endif
    for (int i = 0; i < out_n; ++i) {
        // Dispatch to optimized kernel (unrolled if m ∈ {3,5,7,9,11,13,15})
        y[i] = convolve_dispatch(x + i, k, m);
    }

    *status = HPCS_SUCCESS;
}

/*
 * hpcs_convolve_valid_fast - FAST mode implementation
 *
 * Minimal validation, skip null pointer checks (assumes valid inputs).
 * OpenMP parallelization enabled.
 */
void hpcs_convolve_valid_fast(
    const double* x,
    int n,
    const double* k,
    int m,
    double* y,
    int* status
) {
    // Minimal validation (skip null pointer checks)
    if (n <= 0 || m <= 0 || m > n) {
        if (status != nullptr) *status = HPCS_ERR_INVALID_ARGS;
        return;
    }

    const int out_n = n - m + 1;

    // Parallel convolution (same as SAFE mode)
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(out_n >= 1000)
#endif
    for (int i = 0; i < out_n; ++i) {
        y[i] = convolve_dispatch(x + i, k, m);
    }

    *status = HPCS_SUCCESS;
}

/*
 * hpcs_convolve_valid_deterministic - DETERMINISTIC mode implementation
 *
 * Full validation, disable OpenMP for deterministic execution.
 * Guarantees same results across runs (no thread-dependent ordering).
 */
void hpcs_convolve_valid_deterministic(
    const double* x,
    int n,
    const double* k,
    int m,
    double* y,
    int* status
) {
    // Full validation (same as SAFE)
    if (n <= 0 || m <= 0 || m > n ||
        x == nullptr || k == nullptr || y == nullptr || status == nullptr) {
        if (status != nullptr) *status = HPCS_ERR_INVALID_ARGS;
        return;
    }

    const int out_n = n - m + 1;

    // Sequential convolution (no OpenMP for deterministic behavior)
    for (int i = 0; i < out_n; ++i) {
        y[i] = convolve_dispatch(x + i, k, m);
    }

    *status = HPCS_SUCCESS;
}

} // anonymous namespace

extern "C" {

/*
 * hpcs_convolve_valid - Public API with execution mode support
 *
 * Compute valid-mode 1D convolution (no padding).
 * Optimized for small FIR kernels (m = 3-15).
 *
 * Formula:
 *   y[i] = Σ_{j=0}^{m-1} x[i+j] * k[j]  for i = 0 .. n-m
 *
 * Parameters:
 *   x      - Input signal [n]
 *   n      - Input length
 *   k      - Filter kernel [m]
 *   m      - Kernel length (should be ≪ n, typically 3-15)
 *   y      - Output signal [n-m+1] (valid region only)
 *   mode   - Execution mode: -1=use global, 0=SAFE, 1=FAST, 2=DETERMINISTIC
 *   status - Output: 0=success, 1=invalid args
 *
 * Output length: out_n = n - m + 1
 *
 * Execution Modes:
 *   - SAFE (0): Full validation, OpenMP enabled (default behavior)
 *   - FAST (1): Skip null pointer checks, OpenMP enabled (~5% faster)
 *   - DETERMINISTIC (2): Full validation, OpenMP disabled (reproducible)
 *
 * Parallelization (SAFE/FAST):
 *   - OpenMP parallelizes outer loop (output indices)
 *   - Each thread writes to independent output region
 *   - Static scheduling for load balance
 *
 * Example:
 *   double x[] = {1,2,3,4,5};        // Signal
 *   double k[] = {0.25, 0.5, 0.25};  // Smoothing kernel (size 3)
 *   double y[3];                     // Output size = 5-3+1 = 3
 *   int status;
 *   hpcs_convolve_valid(x, 5, k, 3, y, HPCS_MODE_SAFE, &status);
 *   // y = {1*0.25 + 2*0.5 + 3*0.25, 2*0.25 + 3*0.5 + 4*0.25, 3*0.25 + 4*0.5 + 5*0.25}
 *   //   = {2.0, 3.0, 4.0}
 */
void hpcs_convolve_valid(
    const double* x,
    int n,
    const double* k,
    int m,
    double* y,
    int mode,
    int* status
) {
    // Resolve execution mode
    int effective_mode = get_effective_mode(mode);

    // Dispatch to mode-specific implementation
    switch (effective_mode) {
        case HPCS_MODE_SAFE:
            hpcs_convolve_valid_safe(x, n, k, m, y, status);
            break;
        case HPCS_MODE_FAST:
            hpcs_convolve_valid_fast(x, n, k, m, y, status);
            break;
        case HPCS_MODE_DETERMINISTIC:
            hpcs_convolve_valid_deterministic(x, n, k, m, y, status);
            break;
        default:
            if (status != nullptr) *status = HPCS_ERR_INVALID_ARGS;
            break;
    }
}

/*
 * hpcs_convolve_valid_symmetric
 *
 * Optimized convolution for symmetric kernels (k[i] = k[m-1-i]).
 * Reduces FLOPs by ~2x using symmetry.
 *
 * Many FIR filters are symmetric (Gaussian, Hamming, etc.).
 * This function exploits symmetry to compute convolution faster.
 *
 * Formula (for symmetric k):
 *   y[i] = k[m/2] * x[i+m/2] +  Σ_{j=0}^{m/2-1} k[j] * (x[i+j] + x[i+m-1-j])
 *
 * Parameters:
 *   x      - Input signal [n]
 *   n      - Input length
 *   k      - Symmetric filter kernel [m] (assumed symmetric)
 *   m      - Kernel length (should be odd for exact center)
 *   y      - Output signal [n-m+1]
 *   status - Output: 0=success, 1=invalid args
 *
 * Note: Caller must ensure k is symmetric. No validation performed.
 */
void hpcs_convolve_valid_symmetric(
    const double* x,
    int n,
    const double* k,
    int m,
    double* y,
    int* status
) {
    // Validate arguments
    if (n <= 0 || m <= 0 || m > n ||
        x == nullptr || k == nullptr || y == nullptr || status == nullptr) {
        if (status != nullptr) *status = 1;  // HPCS_ERR_INVALID_ARGS
        return;
    }

    const int out_n = n - m + 1;
    const int half_m = m / 2;
    const bool odd_m = (m % 2 == 1);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(out_n >= 1000)
#endif
    for (int i = 0; i < out_n; ++i) {
        double sum = 0.0;

        // Exploit symmetry: add pairs
        for (int j = 0; j < half_m; ++j) {
            sum += k[j] * (x[i + j] + x[i + m - 1 - j]);
        }

        // Add center element if kernel is odd-sized
        if (odd_m) {
            sum += k[half_m] * x[i + half_m];
        }

        y[i] = sum;
    }

    *status = 0;  // HPCS_SUCCESS
}

} // extern "C"
