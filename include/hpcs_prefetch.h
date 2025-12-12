/**
 * HPCS Prefetch Hints - v0.6 Microarchitecture Optimization
 * ===========================================================
 *
 * Portable prefetch macros for explicit cache control.
 *
 * Prefetching brings data into cache before it's needed, reducing memory
 * latency for predictable access patterns (e.g., sequential scans, rolling windows).
 *
 * Performance Impact:
 * - Large arrays (>1MB): 10-20% speedup
 * - Medium arrays (100K-1M): 5-15% speedup
 * - Small arrays (<100K): Minimal/negative (cache pollution)
 *
 * Compiler Support:
 * - GCC/Clang: __builtin_prefetch
 * - MSVC: _mm_prefetch
 * - Intel ICC: __builtin_prefetch
 * - ARM: __builtin_prefetch (NEON)
 *
 * Usage:
 *   for (int i = 0; i < n; i++) {
 *       HPCS_PREFETCH_READ(&x[i + 64]);  // Prefetch 64 elements ahead
 *       result[i] = compute(x[i]);
 *   }
 */

#ifndef HPCS_PREFETCH_H
#define HPCS_PREFETCH_H

// ============================================================================
// Compiler Detection
// ============================================================================

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
    #define HPCS_HAS_BUILTIN_PREFETCH 1
#elif defined(_MSC_VER)
    #include <xmmintrin.h>  // For _mm_prefetch
    #define HPCS_HAS_MSVC_PREFETCH 1
#else
    #define HPCS_HAS_NO_PREFETCH 1
#endif

// ============================================================================
// Prefetch Macros
// ============================================================================

/**
 * HPCS_PREFETCH_READ(addr)
 *
 * Prefetch data for reading (load into cache with no intent to modify).
 *
 * Locality hints:
 * - 3: High temporal locality (data will be reused soon)
 * - 2: Moderate temporal locality
 * - 1: Low temporal locality
 * - 0: No temporal locality (streaming, single use)
 *
 * For most HPCSeries workloads, use locality = 3 (default).
 */
#if defined(HPCS_HAS_BUILTIN_PREFETCH)
    // GCC/Clang/ICC: __builtin_prefetch(addr, rw, locality)
    // rw: 0 = read, 1 = write
    // locality: 0-3 (0 = no temporal locality, 3 = high temporal locality)
    #define HPCS_PREFETCH_READ(addr)        __builtin_prefetch((addr), 0, 3)
    #define HPCS_PREFETCH_READ_LOW(addr)    __builtin_prefetch((addr), 0, 1)
    #define HPCS_PREFETCH_WRITE(addr)       __builtin_prefetch((addr), 1, 3)

#elif defined(HPCS_HAS_MSVC_PREFETCH)
    // MSVC: _mm_prefetch(addr, hint)
    // _MM_HINT_T0: prefetch to L1 cache (high temporal locality)
    // _MM_HINT_T1: prefetch to L2 cache
    // _MM_HINT_T2: prefetch to L3 cache
    // _MM_HINT_NTA: non-temporal (no cache pollution)
    #define HPCS_PREFETCH_READ(addr)        _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define HPCS_PREFETCH_READ_LOW(addr)    _mm_prefetch((const char*)(addr), _MM_HINT_T2)
    #define HPCS_PREFETCH_WRITE(addr)       _mm_prefetch((const char*)(addr), _MM_HINT_T0)

#else
    // No prefetch support - macros become no-ops
    #define HPCS_PREFETCH_READ(addr)        ((void)0)
    #define HPCS_PREFETCH_READ_LOW(addr)    ((void)0)
    #define HPCS_PREFETCH_WRITE(addr)       ((void)0)
#endif

// ============================================================================
// Prefetch Distance Tuning
// ============================================================================

/**
 * Optimal prefetch distance depends on:
 * - Memory latency (DDR4: ~100ns, DDR5: ~80ns)
 * - CPU frequency
 * - Cache line size (typically 64 bytes = 8 doubles)
 * - Loop iteration time
 *
 * Rule of thumb:
 * - Fast loops: prefetch_distance = 64-128 elements
 * - Slow loops (heavy compute): prefetch_distance = 16-32 elements
 * - Very slow loops: prefetch_distance = 8-16 elements
 *
 * These can be tuned via v0.5 calibration system.
 */

// Default prefetch distances (elements, not bytes)
#define HPCS_PREFETCH_DIST_REDUCTION     64   // For simple reductions (sum, mean)
#define HPCS_PREFETCH_DIST_ROLLING       32   // For rolling operations
#define HPCS_PREFETCH_DIST_AXIS          64   // For 2D axis operations
#define HPCS_PREFETCH_DIST_HEAVY         16   // For compute-heavy loops

// Cache line size (bytes)
#define HPCS_CACHE_LINE_SIZE             64

// Number of doubles per cache line
#define HPCS_DOUBLES_PER_CACHE_LINE      (HPCS_CACHE_LINE_SIZE / sizeof(double))

// ============================================================================
// Prefetch Patterns
// ============================================================================

/**
 * Pattern 1: Sequential Scan with Prefetch
 *
 * Use for: reductions, transformations, scans
 *
 * for (int i = 0; i < n; i++) {
 *     if (i + HPCS_PREFETCH_DIST_REDUCTION < n) {
 *         HPCS_PREFETCH_READ(&x[i + HPCS_PREFETCH_DIST_REDUCTION]);
 *     }
 *     result += x[i];
 * }
 */

/**
 * Pattern 2: Strided Access with Prefetch
 *
 * Use for: 2D matrix column access, batched operations
 *
 * for (int i = 0; i < n; i++) {
 *     if (i + stride < n) {
 *         HPCS_PREFETCH_READ(&matrix[i + stride]);
 *     }
 *     result += matrix[i];
 * }
 */

/**
 * Pattern 3: Rolling Window with Prefetch
 *
 * Use for: rolling operations (mean, median, etc.)
 *
 * for (int i = window; i < n; i++) {
 *     // Prefetch next window's new element
 *     if (i + HPCS_PREFETCH_DIST_ROLLING < n) {
 *         HPCS_PREFETCH_READ(&x[i + HPCS_PREFETCH_DIST_ROLLING]);
 *     }
 *     window_sum = window_sum - x[i - window] + x[i];
 *     result[i] = window_sum / window;
 * }
 */

/**
 * Pattern 4: Write Prefetch for Output Arrays
 *
 * Use for: large output buffers
 *
 * for (int i = 0; i < n; i++) {
 *     if (i + 32 < n) {
 *         HPCS_PREFETCH_WRITE(&output[i + 32]);
 *     }
 *     output[i] = compute(input[i]);
 * }
 */

// ============================================================================
// Prefetch Utilities
// ============================================================================

/**
 * Prefetch an entire cache line starting at address.
 */
static inline void hpcs_prefetch_cache_line(const void *addr) {
    HPCS_PREFETCH_READ(addr);
}

/**
 * Prefetch N consecutive cache lines starting at address.
 * Use for large data structures that span multiple cache lines.
 */
static inline void hpcs_prefetch_n_lines(const void *addr, int n_lines) {
    const char *ptr = (const char *)addr;
    for (int i = 0; i < n_lines; i++) {
        HPCS_PREFETCH_READ(ptr + i * HPCS_CACHE_LINE_SIZE);
    }
}

/**
 * Prefetch array region [start, start+count) for reading.
 * Prefetches one cache line per HPCS_CACHE_LINE_SIZE bytes.
 */
static inline void hpcs_prefetch_region(const double *start, int count) {
    const char *ptr = (const char *)start;
    int bytes = count * sizeof(double);
    int n_lines = (bytes + HPCS_CACHE_LINE_SIZE - 1) / HPCS_CACHE_LINE_SIZE;
    hpcs_prefetch_n_lines(ptr, n_lines);
}

// ============================================================================
// Performance Notes
// ============================================================================

/*
 * When to Use Prefetching:
 * ========================
 *
 * ✅ GOOD USE CASES:
 * - Large sequential scans (>100K elements)
 * - Predictable access patterns
 * - Memory-bound operations
 * - Rolling windows with known stride
 *
 * ❌ BAD USE CASES:
 * - Random access patterns
 * - Small arrays (<10K elements)
 * - Data already in L1 cache
 * - Compute-bound operations (CPU saturated)
 *
 * Tuning Guidelines:
 * ==================
 *
 * 1. Start with default distances
 * 2. Benchmark with/without prefetch
 * 3. If no speedup, disable prefetch
 * 4. If speedup, try doubling distance
 * 5. Find optimal distance via binary search
 *
 * Expected Speedups:
 * ==================
 *
 * Array Size    | Expected Gain | Notes
 * --------------|---------------|---------------------------
 * < 10K         | 0% (negative) | Already in cache
 * 10K - 100K    | 0-5%          | Marginal improvement
 * 100K - 1M     | 5-15%         | Good improvement
 * 1M - 10M      | 10-20%        | Significant improvement
 * > 10M         | 15-25%        | Best improvement
 *
 * Cache Hierarchy Impact:
 * =======================
 *
 * L1 Cache (~32 KB):   ~4K doubles  → prefetch not helpful
 * L2 Cache (~256 KB):  ~32K doubles → prefetch marginally helpful
 * L3 Cache (~4 MB):    ~512K doubles → prefetch helpful
 * Main Memory (GB):    Everything else → prefetch very helpful
 */

#endif // HPCS_PREFETCH_H
