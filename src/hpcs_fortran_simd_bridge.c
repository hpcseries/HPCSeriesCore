/**
 * HPCS Fortran-SIMD Bridge - v0.6
 *
 * Provides C wrapper functions with Fortran-compatible signatures that
 * dispatch to optimized SIMD kernels. This allows Fortran code to
 * transparently benefit from AVX2/AVX/SSE2 intrinsics without modification.
 *
 * Architecture:
 * 1. Fortran calls C wrapper (e.g., hpcs_reduce_sum_simd)
 * 2. C wrapper validates inputs and calls SIMD dispatch
 * 3. SIMD dispatch selects optimal kernel (AVX2 -> OpenMP SIMD -> fallback)
 * 4. Result returned to Fortran with proper status codes
 *
 * Status Codes (from hpcs_constants.f90):
 * - HPCS_SUCCESS = 0
 * - HPCS_ERR_INVALID_ARGS = -1
 */

#include <stdio.h>
#include <math.h>

// SIMD dispatch functions from v0.6
extern double hpcs_dispatch_reduce_sum(const double *x, int n);
extern double hpcs_dispatch_reduce_mean(const double *x, int n);

// SIMD reduction kernels (for direct fallback)
extern double reduce_sum_openmp_simd(const double *x, int n);
extern double reduce_mean_openmp_simd(const double *x, int n);
extern double reduce_min_openmp_simd(const double *x, int n);
extern double reduce_max_openmp_simd(const double *x, int n);
extern double reduce_variance_openmp_simd(const double *x, int n);
extern double reduce_std_openmp_simd(const double *x, int n);

// Status codes (must match hpcs_constants.f90)
#define HPCS_SUCCESS 0
#define HPCS_ERR_INVALID_ARGS -1

// ============================================================================
// Fortran-Compatible C Wrappers (match Fortran signatures exactly)
// ============================================================================

/**
 * Sum reduction - SIMD-accelerated version for Fortran
 *
 * Fortran signature:
 *   subroutine hpcs_reduce_sum_simd(x, n, out, status) bind(C)
 */
void hpcs_reduce_sum_simd(const double *x, int n, double *out, int *status) {
    // Validate inputs
    if (n <= 0) {
        *status = HPCS_ERR_INVALID_ARGS;
        *out = 0.0;
        return;
    }

    // Dispatch to SIMD kernel
    *out = hpcs_dispatch_reduce_sum(x, n);
    *status = HPCS_SUCCESS;
}

/**
 * Mean reduction - SIMD-accelerated version for Fortran
 */
void hpcs_reduce_mean_simd(const double *x, int n, double *out, int *status) {
    if (n <= 0) {
        *status = HPCS_ERR_INVALID_ARGS;
        *out = 0.0;
        return;
    }

    *out = hpcs_dispatch_reduce_mean(x, n);
    *status = HPCS_SUCCESS;
}

/**
 * Min reduction - SIMD-accelerated version for Fortran
 */
void hpcs_reduce_min_simd(const double *x, int n, double *out, int *status) {
    if (n <= 0) {
        *status = HPCS_ERR_INVALID_ARGS;
        *out = 0.0;
        return;
    }

    // Direct call to OpenMP SIMD kernel (no dispatch table for min/max yet)
    *out = reduce_min_openmp_simd(x, n);
    *status = HPCS_SUCCESS;
}

/**
 * Max reduction - SIMD-accelerated version for Fortran
 */
void hpcs_reduce_max_simd(const double *x, int n, double *out, int *status) {
    if (n <= 0) {
        *status = HPCS_ERR_INVALID_ARGS;
        *out = 0.0;
        return;
    }

    *out = reduce_max_openmp_simd(x, n);
    *status = HPCS_SUCCESS;
}

/**
 * Variance reduction - SIMD-accelerated version for Fortran
 */
void hpcs_reduce_variance_simd(const double *x, int n, double *out, int *status) {
    if (n <= 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        *out = 0.0;
        return;
    }

    *out = reduce_variance_openmp_simd(x, n);
    *status = HPCS_SUCCESS;
}

/**
 * Standard deviation reduction - SIMD-accelerated version for Fortran
 */
void hpcs_reduce_std_simd(const double *x, int n, double *out, int *status) {
    if (n <= 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        *out = 0.0;
        return;
    }

    *out = reduce_std_openmp_simd(x, n);
    *status = HPCS_SUCCESS;
}

// ============================================================================
// Diagnostic Functions
// ============================================================================

/**
 * Check if SIMD is available and which ISA is being used
 */
void hpcs_simd_get_info(char *isa_name, int name_len, int *simd_width) {
    extern const char* hpcs_get_simd_name(void);
    extern int hpcs_get_simd_width_doubles(void);

    const char *name = hpcs_get_simd_name();
    int width = hpcs_get_simd_width_doubles();

    // Copy ISA name (safely)
    int i = 0;
    while (i < name_len - 1 && name[i] != '\0') {
        isa_name[i] = name[i];
        i++;
    }
    isa_name[i] = '\0';

    *simd_width = width;
}
