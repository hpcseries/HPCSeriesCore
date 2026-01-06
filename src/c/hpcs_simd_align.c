/**
 * HPCS SIMD Alignment Utilities - v0.6
 *
 * Memory alignment helpers for optimal SIMD performance.
 *
 * Alignment requirements:
 * - SSE2/NEON:  16-byte (128-bit)
 * - AVX/AVX2:   32-byte (256-bit)
 * - AVX-512:    64-byte (512-bit)
 *
 * Provides:
 * - Aligned memory allocation
 * - Alignment checking
 * - Pointer alignment utilities
 * - Tail-loop helpers for unaligned remainders
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

// SIMD dispatch from v0.6
extern int hpcs_get_simd_width_bytes(void);

// ============================================================================
// Alignment Constants
// ============================================================================

#define ALIGN_SSE2     16  // 128-bit
#define ALIGN_AVX      32  // 256-bit
#define ALIGN_AVX512   64  // 512-bit

// ============================================================================
// Aligned Memory Allocation
// ============================================================================

/**
 * Allocate aligned memory
 *
 * @param size - Size in bytes
 * @param alignment - Alignment requirement (16, 32, or 64)
 * @return Aligned pointer, or NULL on failure
 *
 * Must be freed with hpcs_aligned_free()
 */
void* hpcs_aligned_alloc(size_t size, size_t alignment) {
    if (size == 0) {
        return NULL;
    }

#ifdef _WIN32
    // Windows: _aligned_malloc
    return _aligned_malloc(size, alignment);
#else
    // POSIX: posix_memalign
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

/**
 * Free aligned memory
 */
void hpcs_aligned_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }

#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * Allocate double array with SIMD alignment
 *
 * Automatically selects alignment based on detected SIMD width.
 *
 * @param n - Number of doubles
 * @return Aligned double array, or NULL on failure
 */
double* hpcs_alloc_doubles_aligned(int n) {
    if (n <= 0) {
        return NULL;
    }

    size_t alignment = (size_t)hpcs_get_simd_width_bytes();
    if (alignment < ALIGN_SSE2) {
        alignment = ALIGN_SSE2;  // Minimum 16-byte alignment
    }

    size_t size = n * sizeof(double);
    return (double*)hpcs_aligned_alloc(size, alignment);
}

// ============================================================================
// Alignment Checking
// ============================================================================

/**
 * Check if pointer is aligned to specified boundary
 *
 * @param ptr - Pointer to check
 * @param alignment - Required alignment (16, 32, 64)
 * @return 1 if aligned, 0 otherwise
 */
int hpcs_is_aligned(const void *ptr, size_t alignment) {
    uintptr_t addr = (uintptr_t)ptr;
    return (addr % alignment) == 0;
}

/**
 * Check if pointer is SIMD-aligned for current ISA
 */
int hpcs_is_simd_aligned(const void *ptr) {
    size_t alignment = (size_t)hpcs_get_simd_width_bytes();
    return hpcs_is_aligned(ptr, alignment);
}

/**
 * Get misalignment offset
 *
 * Returns number of bytes to skip to reach aligned address.
 *
 * @param ptr - Pointer to check
 * @param alignment - Required alignment
 * @return Offset in bytes (0 if already aligned)
 */
size_t hpcs_alignment_offset(const void *ptr, size_t alignment) {
    uintptr_t addr = (uintptr_t)ptr;
    size_t remainder = addr % alignment;

    if (remainder == 0) {
        return 0;
    }

    return alignment - remainder;
}

/**
 * Get number of elements to skip for alignment (for double arrays)
 */
int hpcs_alignment_offset_doubles(const double *ptr, size_t alignment) {
    size_t byte_offset = hpcs_alignment_offset(ptr, alignment);
    return (int)(byte_offset / sizeof(double));
}

// ============================================================================
// Tail Loop Utilities
// ============================================================================

/**
 * Compute aligned vector count and tail count
 *
 * For a loop over n elements with SIMD width w:
 * - aligned_count = number of full SIMD vectors
 * - tail_count = remaining scalar elements
 *
 * Example: n=100, width=4 (AVX)
 * - aligned_count = 25 vectors (100 elements)
 * - tail_count = 0
 *
 * Example: n=102, width=4
 * - aligned_count = 25 vectors (100 elements)
 * - tail_count = 2
 */
void hpcs_compute_simd_loop_counts(int n, int simd_width,
                                   int *aligned_count, int *tail_count) {
    *aligned_count = n / simd_width;
    *tail_count = n % simd_width;
}

/**
 * Compute alignment preamble and main loop counts
 *
 * For unaligned data, process:
 * 1. Preamble: scalar elements until aligned
 * 2. Main loop: aligned SIMD vectors
 * 3. Tail: remaining scalar elements
 */
void hpcs_compute_aligned_loop_counts(const double *ptr, int n,
                                      int *preamble_count,
                                      int *aligned_count,
                                      int *tail_count) {
    int simd_width_bytes = hpcs_get_simd_width_bytes();
    int simd_width_doubles = simd_width_bytes / sizeof(double);

    // Check if already aligned
    if (hpcs_is_aligned(ptr, simd_width_bytes)) {
        *preamble_count = 0;
        hpcs_compute_simd_loop_counts(n, simd_width_doubles,
                                      aligned_count, tail_count);
        return;
    }

    // Compute preamble size
    int offset_doubles = hpcs_alignment_offset_doubles(ptr, simd_width_bytes);
    *preamble_count = (offset_doubles < n) ? offset_doubles : n;

    // Remaining elements after preamble
    int remaining = n - *preamble_count;
    if (remaining <= 0) {
        *aligned_count = 0;
        *tail_count = 0;
        return;
    }

    // Compute aligned and tail counts
    hpcs_compute_simd_loop_counts(remaining, simd_width_doubles,
                                  aligned_count, tail_count);
}

// ============================================================================
// Alignment Diagnostics
// ============================================================================

/**
 * Print alignment information for a pointer
 */
void hpcs_print_alignment_info(const char *label, const void *ptr, int n) {
    printf("[Alignment] %s:\n", label);
    printf("  Address:      %p\n", ptr);
    printf("  Elements:     %d\n", n);
    printf("  Size:         %zu bytes\n", n * sizeof(double));
    printf("  16-byte:      %s\n", hpcs_is_aligned(ptr, 16) ? "✓" : "✗");
    printf("  32-byte:      %s\n", hpcs_is_aligned(ptr, 32) ? "✓" : "✗");
    printf("  64-byte:      %s\n", hpcs_is_aligned(ptr, 64) ? "✓" : "✗");
    printf("  SIMD-aligned: %s\n", hpcs_is_simd_aligned(ptr) ? "✓" : "✗");

    int simd_width = hpcs_get_simd_width_bytes() / sizeof(double);
    int preamble, aligned, tail;
    hpcs_compute_aligned_loop_counts((const double*)ptr, n,
                                     &preamble, &aligned, &tail);

    printf("  Loop structure (width=%d):\n", simd_width);
    printf("    Preamble:   %d elements\n", preamble);
    printf("    Aligned:    %d vectors (%d elements)\n",
           aligned, aligned * simd_width);
    printf("    Tail:       %d elements\n", tail);
}

// ============================================================================
// Zero-Copy Wrappers
// ============================================================================

/**
 * Copy to aligned buffer if needed
 *
 * If input is not aligned, copies to aligned temporary buffer.
 * Otherwise, returns original pointer.
 *
 * @param input - Input array (may be unaligned)
 * @param n - Number of elements
 * @param aligned_buffer - Output: aligned buffer (caller must free if non-NULL)
 * @return Pointer to use (either input or aligned_buffer)
 */
const double* hpcs_ensure_aligned(const double *input, int n,
                                  double **aligned_buffer) {
    *aligned_buffer = NULL;

    // Check if already aligned
    if (hpcs_is_simd_aligned(input)) {
        return input;  // No copy needed
    }

    // Allocate aligned buffer and copy
    *aligned_buffer = hpcs_alloc_doubles_aligned(n);
    if (*aligned_buffer == NULL) {
        return input;  // Allocation failed, use original
    }

    memcpy(*aligned_buffer, input, n * sizeof(double));
    return *aligned_buffer;
}

// ============================================================================
// Alignment Hints for Compiler
// ============================================================================

/**
 * Assume pointer is aligned (compiler hint)
 *
 * This tells the compiler it can assume alignment for optimization.
 * WARNING: Undefined behavior if pointer is not actually aligned!
 */
#define HPCS_ASSUME_ALIGNED(ptr, alignment) \
    __builtin_assume_aligned(ptr, alignment)

// For GCC/Clang: use __builtin_assume_aligned
// For MSVC: use __assume((((uintptr_t)(ptr)) % (alignment)) == 0)

#ifdef _MSC_VER
#define HPCS_ALIGNED_HINT(ptr, alignment) \
    __assume((((uintptr_t)(ptr)) % (alignment)) == 0)
#else
#define HPCS_ALIGNED_HINT(ptr, alignment) \
    (ptr) = (typeof(ptr))__builtin_assume_aligned((ptr), (alignment))
#endif

// ============================================================================
// Testing/Validation
// ============================================================================

/**
 * Test alignment utilities (for unit tests)
 */
int hpcs_test_alignment_utils(void) {
    int passed = 1;

    // Test 1: Aligned allocation
    double *aligned = hpcs_alloc_doubles_aligned(100);
    if (aligned == NULL) {
        printf("[Test] FAIL: Could not allocate aligned memory\n");
        passed = 0;
    } else {
        if (!hpcs_is_simd_aligned(aligned)) {
            printf("[Test] FAIL: Allocated memory is not SIMD-aligned\n");
            passed = 0;
        }
        hpcs_aligned_free(aligned);
    }

    // Test 2: Alignment checking
    double stack_array[100];
    int is_aligned = hpcs_is_aligned(stack_array, 16);
    printf("[Test] Stack array 16-byte aligned: %s\n", is_aligned ? "yes" : "no");

    // Test 3: Loop count computation
    int preamble, aligned_count, tail;
    hpcs_compute_aligned_loop_counts(stack_array, 100,
                                     &preamble, &aligned_count, &tail);
    printf("[Test] Loop counts: preamble=%d, aligned=%d, tail=%d\n",
           preamble, aligned_count, tail);

    if (passed) {
        printf("[Test] All alignment tests PASSED\n");
    }

    return passed;
}
