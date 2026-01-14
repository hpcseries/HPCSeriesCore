/*
 * HPCSeries Core - Version Query & Feature Discovery Functions
 *
 * Provides runtime version information for ABI compatibility checking.
 * Enables downstream consumers to verify library versions
 * at compile-time and runtime.
 *
 * Author: HPCSeries Core Team
 * Version: 0.8.0
 */

#include "../../include/hpcs_core.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* --------------------------------------------------------------------- */
/* Thread-local Error Buffer                                  */
/* --------------------------------------------------------------------- */

#define HPCS_ERROR_BUFFER_SIZE 256

static char g_last_error[HPCS_ERROR_BUFFER_SIZE] = "";

#ifdef _OPENMP
#pragma omp threadprivate(g_last_error)
#endif

/*
 * Get library version string
 *
 * Returns the semantic version string (MAJOR.MINOR.PATCH).
 * This allows runtime version checks and logging.
 *
 * Returns:
 *   Statically allocated version string (do not free)
 *
 * Example:
 *   const char* version = hpcs_get_version();
 *   printf("Using HPCSeries Core %s\n", version);
 */
const char* hpcs_get_version(void) {
    return HPCS_VERSION_STRING;
}

/*
 * Get ABI version number
 *
 * Returns the ABI compatibility version. Increment this on breaking changes.
 * Downstream consumers can verify ABI compatibility:
 *   - Same ABI version = compatible
 *   - Different ABI version = incompatible, rebuild required
 *
 * Returns:
 *   ABI version number
 *
 * Example:
 *   int abi = hpcs_get_abi_version();
 *   if (abi < MIN_REQUIRED_ABI) {
 *       fprintf(stderr, "Incompatible ABI version\n");
 *       exit(1);
 *   }
 */
int hpcs_get_abi_version(void) {
    return HPCS_ABI_VERSION;
}

/*
 * Get build feature bitmask
 *
 * Returns a bitmask indicating which optional features are compiled into
 * this build of the library. Use HPCS_FEAT_* macros to test features.
 *
 * Returns:
 *   Bitmask of enabled features
 */
uint64_t get_build_features(void) {
    uint64_t features = 0;

    /* OpenMP support */
    #ifdef _OPENMP
    features |= HPCS_FEAT_OPENMP;
    #endif

    /* SIMD support - detected via architecture macros */
    #if defined(__AVX2__)
    features |= HPCS_FEAT_SIMD_AVX2;
    #endif

    #if defined(__AVX512F__)
    features |= HPCS_FEAT_SIMD_AVX512;
    #endif

    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
    features |= HPCS_FEAT_SIMD_NEON;
    #endif

    /* Fast-math enabled */
    #if defined(__FAST_MATH__) || defined(HPCS_FAST_MATH)
    features |= HPCS_FEAT_FAST_MATH;
    #endif

    /* GPU offload support */
    #if defined(HPCS_GPU_OFFLOAD) || defined(_OPENMP_TARGET)
    features |= HPCS_FEAT_GPU_OFFLOAD;
    #endif

    return features;
}

/* --------------------------------------------------------------------- */
/* Thread-local Error Handling                                           */
/* --------------------------------------------------------------------- */

/*
 * Get last error message
 *
 * Returns the thread-local error message from the last failing call.
 * Empty string if no error or after successful calls.
 */
const char* get_last_error(void) {
    return g_last_error;
}

/*
 * Set last error message (internal use)
 *
 * Sets the thread-local error message. Called by functions on failure.
 */
void set_last_error(const char* msg) {
    if (msg != NULL) {
        strncpy(g_last_error, msg, HPCS_ERROR_BUFFER_SIZE - 1);
        g_last_error[HPCS_ERROR_BUFFER_SIZE - 1] = '\0';
    } else {
        g_last_error[0] = '\0';
    }
}

/*
 * Clear last error message (internal use)
 *
 * Called by functions on success to clear any previous error.
 */
void clear_last_error(void) {
    g_last_error[0] = '\0';
}
