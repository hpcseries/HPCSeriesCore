/**
 * HPCS CPU Detection API
 *
 * Provides runtime CPU capability detection and adaptive parallelization.
 */

#ifndef HPCS_CPU_DETECT_H
#define HPCS_CPU_DETECT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CPU information structure (v0.5 extended)
 */
typedef struct {
    // Basic CPU topology
    int num_physical_cores;     // Physical cores (not hyperthreads)
    int num_logical_cores;      // Logical cores (including HT)
    int l1_cache_size_kb;       // L1 cache per core (KB)
    int l2_cache_size_kb;       // L2 cache per core (KB)
    int l3_cache_size_kb;       // L3 cache total (KB)
    int optimal_threads;        // Recommended thread count

    // NUMA topology (v0.5)
    int numa_nodes;             // Number of NUMA nodes
    int cores_per_numa_node;    // Average cores per NUMA node
    int *core_to_numa_map;      // Maps each core to its NUMA node (array)

    // SIMD/ISA capabilities (v0.5)
    int has_sse2;               // SSE2 support (baseline x86-64)
    int has_avx;                // AVX support
    int has_avx2;               // AVX2 support
    int has_avx512;             // AVX-512 support
    int has_neon;               // ARM NEON support
    int has_fma3;               // FMA3 support
    int simd_width_bits;        // Max SIMD width (128, 256, 512)

    // CPU identification
    char cpu_vendor[64];        // CPU vendor string (GenuineIntel, AuthenticAMD, etc.)
    char cpu_model[128];        // CPU model string

    int initialized;            // Has detection run?
} hpcs_cpu_info_t;

/**
 * Threshold operation types
 */
#define HPCS_THRESHOLD_SIMPLE_REDUCE   1  // sum, mean, min, max
#define HPCS_THRESHOLD_ROLLING_SIMPLE  2  // rolling sum/mean
#define HPCS_THRESHOLD_COMPUTE_HEAVY   3  // median, MAD, quantile
#define HPCS_THRESHOLD_ANOMALY_DETECT  4  // anomaly detection

/**
 * Initialize CPU detection
 * Call once at program startup
 */
void hpcs_cpu_detect_init(void);

/**
 * Get CPU information structure
 */
void hpcs_cpu_get_info(hpcs_cpu_info_t *info);

/**
 * Get optimal thread count for this hardware
 */
int hpcs_cpu_get_optimal_threads(void);

/**
 * Get adaptive threshold for operation type
 *
 * Returns element count threshold - arrays smaller than this
 * will not use OpenMP parallelization.
 *
 * Threshold adapts to:
 * - CPU core count (more cores = lower threshold)
 * - Cache size (larger cache = lower threshold)
 * - Operation complexity
 */
int hpcs_cpu_get_threshold(int operation_type);

/**
 * Enhanced CPU detection using C system calls
 * (Internal - called from Fortran)
 */
void hpcs_cpu_detect_enhanced(hpcs_cpu_info_t *info);

/**
 * Initialize CPU detection from C
 */
void hpcs_cpu_init(void);

/**
 * v0.5 API: NUMA-aware affinity modes
 */
#define HPCS_AFFINITY_AUTO     0  // Auto-select based on operation
#define HPCS_AFFINITY_COMPACT  1  // Pack threads on same NUMA node
#define HPCS_AFFINITY_SPREAD   2  // Distribute across NUMA nodes

/**
 * v0.5 API: Operation classes for tuning
 */
#define HPCS_OP_CLASS_SIMPLE    1  // Simple reductions (sum, mean, min, max)
#define HPCS_OP_CLASS_ROLLING   2  // Rolling operations
#define HPCS_OP_CLASS_ROBUST    3  // Robust statistics (median, MAD, quantile)
#define HPCS_OP_CLASS_ANOMALY   4  // Anomaly detection

/**
 * Tuning configuration structure (v0.5)
 * Stores optimized parameters for each operation class
 */
typedef struct {
    // Parallelization thresholds per operation class
    int threshold_simple;       // Simple reductions
    int threshold_rolling;      // Rolling operations
    int threshold_robust;       // Robust statistics
    int threshold_anomaly;      // Anomaly detection

    // Optimal thread counts per operation class
    int threads_simple;
    int threads_rolling;
    int threads_robust;
    int threads_anomaly;

    // NUMA affinity modes per operation class
    int numa_mode_simple;       // 0=auto, 1=compact, 2=spread
    int numa_mode_rolling;
    int numa_mode_robust;
    int numa_mode_anomaly;

    // CPU identification for cache validation
    char cpu_id[256];           // CPU identifier string
    long long timestamp;        // Configuration timestamp

    int valid;                  // Is this configuration valid?
} hpcs_tuning_t;

/**
 * Set NUMA affinity mode for thread placement
 */
void hpcs_set_affinity_mode(int mode, int *status);

/**
 * Get current affinity mode
 */
int hpcs_get_affinity_mode(void);

/**
 * Detect NUMA topology (v0.5)
 */
int hpcs_detect_numa_topology(hpcs_cpu_info_t *info);

/**
 * Detect SIMD capabilities using CPUID (v0.5)
 */
int hpcs_detect_simd_capabilities(hpcs_cpu_info_t *info);

/**
 * Get SIMD width in bits (128, 256, 512)
 */
int hpcs_get_simd_width(void);

/**
 * Check if specific SIMD feature is available
 */
int hpcs_has_avx(void);
int hpcs_has_avx2(void);
int hpcs_has_avx512(void);

/**
 * v0.5 Tuning Configuration API
 */

/**
 * Initialize tuning configuration with default values based on CPU
 */
void hpcs_tuning_init(hpcs_tuning_t *tuning);

/**
 * Get current tuning configuration
 */
void hpcs_get_tuning(hpcs_tuning_t *tuning);

/**
 * Set tuning configuration (validates parameters)
 */
int hpcs_set_tuning(const hpcs_tuning_t *tuning);

/**
 * Validate tuning configuration
 * Returns 0 if valid, error code otherwise
 */
int hpcs_validate_tuning(const hpcs_tuning_t *tuning);

/**
 * Get tuning parameter for specific operation class
 */
int hpcs_get_tuning_threshold(int op_class);
int hpcs_get_tuning_threads(int op_class);
int hpcs_get_tuning_numa_mode(int op_class);

/**
 * Set tuning parameter for specific operation class
 */
int hpcs_set_tuning_threshold(int op_class, int threshold);
int hpcs_set_tuning_threads(int op_class, int threads);
int hpcs_set_tuning_numa_mode(int op_class, int numa_mode);

/**
 * Reset tuning to hardware-based defaults
 */
void hpcs_reset_tuning(void);

/**
 * Print current tuning configuration (for debugging)
 */
void hpcs_print_tuning(const hpcs_tuning_t *tuning);

#ifdef __cplusplus
}
#endif

#endif // HPCS_CPU_DETECT_H
