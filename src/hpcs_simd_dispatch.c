/**
 * HPCS SIMD Dispatch - v0.6
 *
 * Runtime ISA selection for vectorized kernels.
 * Integrates with v0.5 CPU detection to pick optimal SIMD path.
 *
 * Dispatch priority:
 * 1. AVX-512 (512-bit, 8 doubles)
 * 2. AVX2 (256-bit, 4 doubles)
 * 3. AVX (256-bit, 4 doubles)
 * 4. SSE2 (128-bit, 2 doubles)
 * 5. OpenMP SIMD (compiler-optimized)
 * 6. Scalar fallback
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CPU detection from v0.5
extern void hpcs_cpu_detect_init(void);

// CPU info structure (must match hpcs_cpu_detect.c exactly!)
typedef struct {
    // Basic CPU topology
    int num_physical_cores;  // MUST be first
    int num_logical_cores;   // MUST be second
    int l1_cache_size_kb;
    int l2_cache_size_kb;
    int l3_cache_size_kb;
    int optimal_threads;

    // NUMA topology (v0.5)
    int numa_nodes;
    int cores_per_numa_node;
    int *core_to_numa_map;

    // SIMD/ISA capabilities (v0.5)
    int has_sse2;
    int has_avx;
    int has_avx2;
    int has_avx512;
    int has_neon;
    int has_fma3;
    int simd_width_bits;

    // CPU identification
    char cpu_vendor[64];
    char cpu_model[128];

    int initialized;
} hpcs_cpu_info_t;

extern void hpcs_cpu_detect_enhanced(hpcs_cpu_info_t *info);

// ============================================================================
// SIMD ISA Enumeration
// ============================================================================

typedef enum {
    SIMD_NONE = 0,      // Scalar fallback
    SIMD_SSE2 = 1,      // 128-bit (2 doubles)
    SIMD_AVX = 2,       // 256-bit (4 doubles)
    SIMD_AVX2 = 3,      // 256-bit (4 doubles) + FMA
    SIMD_AVX512 = 4,    // 512-bit (8 doubles)
    SIMD_NEON = 5,      // ARM NEON (2 doubles)
    SIMD_OPENMP = 6     // Compiler-optimized OpenMP SIMD
} simd_isa_t;

// Global SIMD capability (initialized on first call)
static simd_isa_t g_simd_isa = SIMD_NONE;
static int g_simd_initialized = 0;
static int g_simd_width_bytes = 0;  // 16, 32, 64

// ============================================================================
// SIMD Detection and Initialization
// ============================================================================

/**
 * Detect best available SIMD ISA
 */
static void detect_simd_isa(void) {
    if (g_simd_initialized) {
        return;
    }

    hpcs_cpu_info_t cpu_info;
    memset(&cpu_info, 0, sizeof(cpu_info));  // Zero-initialize to prevent garbage
    hpcs_cpu_detect_init();
    hpcs_cpu_detect_enhanced(&cpu_info);

    // Debug: Print detected SIMD capabilities
    fprintf(stderr, "[SIMD Debug] AVX-512=%d, AVX2=%d, AVX=%d, SSE2=%d\n",
            cpu_info.has_avx512, cpu_info.has_avx2, cpu_info.has_avx, cpu_info.has_sse2);

    // Prioritize widest available ISA
    // Note: Only use hardware ISA if actually available
    if (cpu_info.has_avx512 > 0) {
        g_simd_isa = SIMD_AVX512;
        g_simd_width_bytes = 64;
    } else if (cpu_info.has_avx2 > 0) {
        g_simd_isa = SIMD_AVX2;
        g_simd_width_bytes = 32;
    } else if (cpu_info.has_avx > 0) {
        g_simd_isa = SIMD_AVX;
        g_simd_width_bytes = 32;
    } else if (cpu_info.has_sse2 > 0) {
        g_simd_isa = SIMD_SSE2;
        g_simd_width_bytes = 16;
    } else if (cpu_info.has_neon > 0) {
        g_simd_isa = SIMD_NEON;
        g_simd_width_bytes = 16;
    } else {
        // No hardware SIMD - use OpenMP SIMD (compiler will auto-vectorize)
        g_simd_isa = SIMD_OPENMP;
        g_simd_width_bytes = 32;  // Assume AVX-capable compiler
    }

    g_simd_initialized = 1;

    fprintf(stderr, "[SIMD] Detected ISA: ");
    switch (g_simd_isa) {
        case SIMD_AVX512: fprintf(stderr, "AVX-512 (512-bit)\n"); break;
        case SIMD_AVX2:   fprintf(stderr, "AVX2 (256-bit)\n"); break;
        case SIMD_AVX:    fprintf(stderr, "AVX (256-bit)\n"); break;
        case SIMD_SSE2:   fprintf(stderr, "SSE2 (128-bit)\n"); break;
        case SIMD_NEON:   fprintf(stderr, "NEON (128-bit)\n"); break;
        case SIMD_OPENMP: fprintf(stderr, "OpenMP SIMD (compiler)\n"); break;
        default:          fprintf(stderr, "None (scalar)\n"); break;
    }
}

/**
 * Get current SIMD ISA
 */
simd_isa_t hpcs_get_simd_isa(void) {
    if (!g_simd_initialized) {
        detect_simd_isa();
    }
    return g_simd_isa;
}

/**
 * Get SIMD width in bytes (for alignment)
 */
int hpcs_get_simd_width_bytes(void) {
    if (!g_simd_initialized) {
        detect_simd_isa();
    }
    return g_simd_width_bytes;
}

/**
 * Get SIMD width in doubles (2, 4, or 8)
 */
int hpcs_get_simd_width_doubles(void) {
    if (!g_simd_initialized) {
        detect_simd_isa();
    }
    return g_simd_width_bytes / sizeof(double);
}

/**
 * Get SIMD ISA name (for diagnostics)
 */
const char* hpcs_get_simd_name(void) {
    if (!g_simd_initialized) {
        detect_simd_isa();
    }

    switch (g_simd_isa) {
        case SIMD_AVX512: return "AVX-512";
        case SIMD_AVX2:   return "AVX2";
        case SIMD_AVX:    return "AVX";
        case SIMD_SSE2:   return "SSE2";
        case SIMD_NEON:   return "NEON";
        case SIMD_OPENMP: return "OpenMP SIMD";
        default:          return "Scalar";
    }
}

/**
 * Get L1 cache size in KB
 */
int hpcs_get_l1_cache_size_kb(void) {
    hpcs_cpu_info_t cpu_info;
    memset(&cpu_info, 0, sizeof(cpu_info));
    hpcs_cpu_detect_init();
    hpcs_cpu_detect_enhanced(&cpu_info);
    return cpu_info.l1_cache_size_kb;
}

/**
 * Get L2 cache size in KB
 */
int hpcs_get_l2_cache_size_kb(void) {
    hpcs_cpu_info_t cpu_info;
    memset(&cpu_info, 0, sizeof(cpu_info));
    hpcs_cpu_detect_init();
    hpcs_cpu_detect_enhanced(&cpu_info);
    return cpu_info.l2_cache_size_kb;
}

/**
 * Get optimal number of threads
 */
int hpcs_get_optimal_threads(void) {
    hpcs_cpu_info_t cpu_info;
    memset(&cpu_info, 0, sizeof(cpu_info));
    hpcs_cpu_detect_init();
    hpcs_cpu_detect_enhanced(&cpu_info);
    return cpu_info.optimal_threads;
}

// ============================================================================
// SIMD Kernel Function Pointers (for runtime dispatch)
// ============================================================================

// Reduction kernels
typedef double (*reduce_sum_func_t)(const double *x, int n);
typedef double (*reduce_mean_func_t)(const double *x, int n);

// Rolling kernels
typedef void (*rolling_mean_func_t)(const double *x, int n, int window, double *result);

// Dispatch tables (populated by kernel registration)
static reduce_sum_func_t g_reduce_sum_kernels[7] = {NULL};
static reduce_mean_func_t g_reduce_mean_kernels[7] = {NULL};
static rolling_mean_func_t g_rolling_mean_kernels[7] = {NULL};

// ============================================================================
// Kernel Registration (called by kernel modules during initialization)
// ============================================================================

void hpcs_register_reduce_sum_kernel(simd_isa_t isa, reduce_sum_func_t func) {
    if (isa >= 0 && isa <= SIMD_OPENMP) {
        g_reduce_sum_kernels[isa] = func;
    }
}

void hpcs_register_reduce_mean_kernel(simd_isa_t isa, reduce_mean_func_t func) {
    if (isa >= 0 && isa <= SIMD_OPENMP) {
        g_reduce_mean_kernels[isa] = func;
    }
}

void hpcs_register_rolling_mean_kernel(simd_isa_t isa, rolling_mean_func_t func) {
    if (isa >= 0 && isa <= SIMD_OPENMP) {
        g_rolling_mean_kernels[isa] = func;
    }
}

// ============================================================================
// Dispatch Functions (select optimal kernel at runtime)
// ============================================================================

/**
 * Dispatch reduce_sum to optimal SIMD kernel
 */
double hpcs_dispatch_reduce_sum(const double *x, int n) {
    simd_isa_t isa = hpcs_get_simd_isa();

    // Try to use registered kernel for current ISA
    if (g_reduce_sum_kernels[isa] != NULL) {
        return g_reduce_sum_kernels[isa](x, n);
    }

    // Fallback: try next-best ISA (hardware-specific)
    for (int i = isa - 1; i >= 0; i--) {
        if (g_reduce_sum_kernels[i] != NULL) {
            return g_reduce_sum_kernels[i](x, n);
        }
    }

    // Universal fallback: try OpenMP SIMD (compiler-optimized)
    if (isa != SIMD_OPENMP && g_reduce_sum_kernels[SIMD_OPENMP] != NULL) {
        return g_reduce_sum_kernels[SIMD_OPENMP](x, n);
    }

    // No kernel registered - error
    fprintf(stderr, "[SIMD] ERROR: No reduce_sum kernel registered\n");
    return 0.0;
}

/**
 * Dispatch reduce_mean to optimal SIMD kernel
 */
double hpcs_dispatch_reduce_mean(const double *x, int n) {
    simd_isa_t isa = hpcs_get_simd_isa();

    if (g_reduce_mean_kernels[isa] != NULL) {
        return g_reduce_mean_kernels[isa](x, n);
    }

    // Fallback: try next-best ISA (hardware-specific)
    for (int i = isa - 1; i >= 0; i--) {
        if (g_reduce_mean_kernels[i] != NULL) {
            return g_reduce_mean_kernels[i](x, n);
        }
    }

    // Universal fallback: try OpenMP SIMD (compiler-optimized)
    if (isa != SIMD_OPENMP && g_reduce_mean_kernels[SIMD_OPENMP] != NULL) {
        return g_reduce_mean_kernels[SIMD_OPENMP](x, n);
    }

    fprintf(stderr, "[SIMD] ERROR: No reduce_mean kernel registered\n");
    return 0.0;
}

/**
 * Dispatch rolling_mean to optimal SIMD kernel
 */
void hpcs_dispatch_rolling_mean(const double *x, int n, int window, double *result) {
    simd_isa_t isa = hpcs_get_simd_isa();

    if (g_rolling_mean_kernels[isa] != NULL) {
        g_rolling_mean_kernels[isa](x, n, window, result);
        return;
    }

    // Fallback: try next-best ISA (hardware-specific)
    for (int i = isa - 1; i >= 0; i--) {
        if (g_rolling_mean_kernels[i] != NULL) {
            g_rolling_mean_kernels[i](x, n, window, result);
            return;
        }
    }

    // Universal fallback: try OpenMP SIMD (compiler-optimized)
    if (isa != SIMD_OPENMP && g_rolling_mean_kernels[SIMD_OPENMP] != NULL) {
        g_rolling_mean_kernels[SIMD_OPENMP](x, n, window, result);
        return;
    }

    fprintf(stderr, "[SIMD] ERROR: No rolling_mean kernel registered\n");
}

// ============================================================================
// SIMD Configuration Persistence (extends v0.5 config.json)
// ============================================================================

/**
 * Get SIMD configuration for saving to config.json
 */
void hpcs_get_simd_config(char *simd_name, int *simd_width) {
    if (!g_simd_initialized) {
        detect_simd_isa();
    }

    snprintf(simd_name, 32, "%s", hpcs_get_simd_name());
    *simd_width = g_simd_width_bytes;
}

/**
 * Print SIMD dispatch status (for diagnostics)
 */
void hpcs_print_simd_status(void) {
    if (!g_simd_initialized) {
        detect_simd_isa();
    }

    printf("=== SIMD Dispatch Status ===\n");
    printf("ISA:         %s\n", hpcs_get_simd_name());
    printf("Width:       %d bytes (%d doubles)\n",
           g_simd_width_bytes, hpcs_get_simd_width_doubles());
    printf("\n");

    printf("Registered Kernels:\n");
    // Safety check for array bounds
    int isa_index = (int)g_simd_isa;
    if (isa_index >= 0 && isa_index <= SIMD_OPENMP) {
        // Check hardware-specific kernel
        int has_reduce_sum = g_reduce_sum_kernels[isa_index] != NULL;
        int has_reduce_mean = g_reduce_mean_kernels[isa_index] != NULL;
        int has_rolling_mean = g_rolling_mean_kernels[isa_index] != NULL;

        // Check OpenMP SIMD fallback
        int has_openmp_sum = (isa_index != SIMD_OPENMP && g_reduce_sum_kernels[SIMD_OPENMP] != NULL);
        int has_openmp_mean = (isa_index != SIMD_OPENMP && g_reduce_mean_kernels[SIMD_OPENMP] != NULL);
        int has_openmp_rolling = (isa_index != SIMD_OPENMP && g_rolling_mean_kernels[SIMD_OPENMP] != NULL);

        printf("  reduce_sum:    %s%s\n",
               has_reduce_sum ? "✓" : (has_openmp_sum ? "✓ (OpenMP)" : "✗"),
               has_reduce_sum ? "" : "");
        printf("  reduce_mean:   %s%s\n",
               has_reduce_mean ? "✓" : (has_openmp_mean ? "✓ (OpenMP)" : "✗"),
               has_reduce_mean ? "" : "");
        printf("  rolling_mean:  %s%s\n",
               has_rolling_mean ? "✓" : (has_openmp_rolling ? "✓ (OpenMP)" : "✗"),
               has_rolling_mean ? "" : "");
    } else {
        printf("  ERROR: Invalid ISA index %d\n", isa_index);
    }
    printf("============================\n");
}

// ============================================================================
// SIMD-aware Threshold Adjustment (integrates with v0.5 auto-tuning)
// ============================================================================

/**
 * Adjust parallel threshold based on SIMD width
 *
 * SIMD reduces per-element cost, so parallelization is worthwhile at
 * smaller sizes. Adjust thresholds accordingly:
 *
 * - AVX-512 (8x width): threshold *= 0.5  (2x lower)
 * - AVX2 (4x width):    threshold *= 0.6  (1.6x lower)
 * - AVX (4x width):     threshold *= 0.7  (1.4x lower)
 * - SSE2 (2x width):    threshold *= 0.8  (1.25x lower)
 */
int hpcs_adjust_threshold_for_simd(int base_threshold) {
    if (!g_simd_initialized) {
        detect_simd_isa();
    }

    double scale = 1.0;

    switch (g_simd_isa) {
        case SIMD_AVX512:
            scale = 0.5;  // 50% reduction (8x SIMD width)
            break;
        case SIMD_AVX2:
            scale = 0.6;  // 40% reduction (4x SIMD width + FMA)
            break;
        case SIMD_AVX:
            scale = 0.7;  // 30% reduction (4x SIMD width)
            break;
        case SIMD_SSE2:
        case SIMD_NEON:
            scale = 0.8;  // 20% reduction (2x SIMD width)
            break;
        default:
            scale = 1.0;  // No adjustment
            break;
    }

    int adjusted = (int)(base_threshold * scale);

    // Ensure minimum threshold (avoid over-parallelizing tiny arrays)
    if (adjusted < 1000) {
        adjusted = 1000;
    }

    return adjusted;
}
