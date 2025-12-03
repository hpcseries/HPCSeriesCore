/**
 * HPCS CPU Detection - C Implementation (v0.5)
 *
 * Provides detailed CPU capability detection including:
 * - NUMA topology
 * - SIMD/AVX capabilities
 * - CPU vendor/model identification
 */

// Define _GNU_SOURCE for Linux-specific extensions (pthread affinity, CPU_SET)
#ifndef _WIN32
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

// CPUID support for x86/x86_64
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define HPCS_HAS_CPUID 1
#ifdef __GNUC__
#include <cpuid.h>
#endif
#endif

// ARM NEON detection
#if defined(__ARM_NEON) || defined(__aarch64__)
#define HPCS_HAS_NEON 1
#endif

// External Fortran functions
extern void hpcs_cpu_detect_init(void);
extern int hpcs_cpu_get_threshold(int operation_type);

// Global affinity mode
static int g_affinity_mode = 0;  // 0 = auto, 1 = compact, 2 = spread

// Global tuning configuration (v0.5)
typedef struct {
    int threshold_simple;
    int threshold_rolling;
    int threshold_robust;
    int threshold_anomaly;
    int threads_simple;
    int threads_rolling;
    int threads_robust;
    int threads_anomaly;
    int numa_mode_simple;
    int numa_mode_rolling;
    int numa_mode_robust;
    int numa_mode_anomaly;
    char cpu_id[256];
    long long timestamp;
    int valid;
} hpcs_tuning_t;

static hpcs_tuning_t g_tuning = {0};
static int g_tuning_initialized = 0;

/**
 * CPU information structure (must match header definition - v0.5)
 */
typedef struct {
    // Basic CPU topology
    int num_physical_cores;
    int num_logical_cores;
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

/**
 * Execute CPUID instruction (x86/x86_64)
 */
#ifdef HPCS_HAS_CPUID
static void run_cpuid(unsigned int eax, unsigned int ecx, unsigned int *regs) {
#ifdef _WIN32
    __cpuidex((int*)regs, eax, ecx);
#else
    __cpuid_count(eax, ecx, regs[0], regs[1], regs[2], regs[3]);
#endif
}
#endif

/**
 * Detect SIMD capabilities using CPUID (v0.5)
 */
int hpcs_detect_simd_capabilities(hpcs_cpu_info_t *info) {
    // Initialize all to 0
    info->has_sse2 = 0;
    info->has_avx = 0;
    info->has_avx2 = 0;
    info->has_avx512 = 0;
    info->has_neon = 0;
    info->has_fma3 = 0;
    info->simd_width_bits = 64;  // Baseline scalar

#ifdef HPCS_HAS_CPUID
    unsigned int regs[4];

    // Get vendor string
    run_cpuid(0, 0, regs);
    memcpy(info->cpu_vendor, &regs[1], 4);
    memcpy(info->cpu_vendor + 4, &regs[3], 4);
    memcpy(info->cpu_vendor + 8, &regs[2], 4);
    info->cpu_vendor[12] = '\0';

    // Check basic features (EAX=1)
    run_cpuid(1, 0, regs);

    // SSE2 (bit 26 of EDX)
    info->has_sse2 = (regs[3] & (1 << 26)) ? 1 : 0;
    if (info->has_sse2) {
        info->simd_width_bits = 128;
    }

    // AVX (bit 28 of ECX), requires OS support via OSXSAVE (bit 27)
    int has_osxsave = (regs[2] & (1 << 27)) ? 1 : 0;
    info->has_avx = ((regs[2] & (1 << 28)) && has_osxsave) ? 1 : 0;
    if (info->has_avx) {
        info->simd_width_bits = 256;
    }

    // FMA3 (bit 12 of ECX)
    info->has_fma3 = (regs[2] & (1 << 12)) ? 1 : 0;

    // Check extended features (EAX=7, ECX=0)
    run_cpuid(7, 0, regs);

    // AVX2 (bit 5 of EBX)
    info->has_avx2 = ((regs[1] & (1 << 5)) && info->has_avx) ? 1 : 0;
    if (info->has_avx2) {
        info->simd_width_bits = 256;
    }

    // AVX-512F (bit 16 of EBX)
    info->has_avx512 = ((regs[1] & (1 << 16)) && info->has_avx) ? 1 : 0;
    if (info->has_avx512) {
        info->simd_width_bits = 512;
    }
#endif

#ifdef HPCS_HAS_NEON
    // ARM NEON is compile-time detected
    info->has_neon = 1;
    info->simd_width_bits = 128;
    strncpy(info->cpu_vendor, "ARM", 63);
#endif

    return 0;
}

/**
 * Detect NUMA topology (v0.5)
 */
int hpcs_detect_numa_topology(hpcs_cpu_info_t *info) {
    info->numa_nodes = 1;  // Default: single NUMA node
    info->cores_per_numa_node = info->num_physical_cores;
    info->core_to_numa_map = NULL;

#ifdef __linux__
    // Count NUMA nodes in /sys/devices/system/node/
    DIR *dir = opendir("/sys/devices/system/node");
    if (dir) {
        struct dirent *entry;
        int node_count = 0;

        while ((entry = readdir(dir)) != NULL) {
            if (strncmp(entry->d_name, "node", 4) == 0 &&
                entry->d_name[4] >= '0' && entry->d_name[4] <= '9') {
                node_count++;
            }
        }
        closedir(dir);

        if (node_count > 0) {
            info->numa_nodes = node_count;
            info->cores_per_numa_node = info->num_physical_cores / node_count;
        }
    }

    // Allocate and build core-to-NUMA map
    if (info->numa_nodes > 1 && info->num_physical_cores > 0) {
        info->core_to_numa_map = (int*)malloc(info->num_physical_cores * sizeof(int));
        if (info->core_to_numa_map) {
            // Simple distribution: assume cores are evenly distributed
            for (int i = 0; i < info->num_physical_cores; i++) {
                info->core_to_numa_map[i] = i / info->cores_per_numa_node;
            }
        }
    }
#endif

#ifdef _WIN32
    // Windows NUMA detection
    ULONG highest_node;
    if (GetNumaHighestNodeNumber(&highest_node)) {
        info->numa_nodes = highest_node + 1;
        info->cores_per_numa_node = info->num_physical_cores / info->numa_nodes;
    }
#endif

    return 0;
}

/**
 * Detect number of CPU cores
 */
static int detect_num_cores(int *physical, int *logical) {
    *physical = 1;
    *logical = 1;

#ifdef __linux__
    // Try to read from /proc/cpuinfo
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        int processor_count = 0;
        int core_id_set[1024] = {0};
        int unique_cores = 0;

        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "processor", 9) == 0) {
                processor_count++;
            }
            if (strncmp(line, "core id", 7) == 0) {
                int core_id;
                if (sscanf(line, "core id : %d", &core_id) == 1) {
                    if (core_id < 1024 && !core_id_set[core_id]) {
                        core_id_set[core_id] = 1;
                        unique_cores++;
                    }
                }
            }
        }
        fclose(fp);

        *logical = processor_count;
        *physical = (unique_cores > 0) ? unique_cores : processor_count;
    }

    // Fallback to sysconf
    if (*logical <= 1) {
        *logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
        *physical = *logical;
    }
#endif

#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    *logical = sysinfo.dwNumberOfProcessors;
    *physical = sysinfo.dwNumberOfProcessors / 2; // Estimate
#endif

#ifdef __APPLE__
    size_t len = sizeof(int);
    sysctlbyname("hw.ncpu", logical, &len, NULL, 0);
    sysctlbyname("hw.physicalcpu", physical, &len, NULL, 0);
#endif

    return 0;
}

/**
 * Detect cache sizes (in KB)
 */
static int detect_cache_sizes(int *l1, int *l2, int *l3) {
    *l1 = 32;    // Default
    *l2 = 256;   // Default
    *l3 = 8192;  // Default

#ifdef __linux__
    // Try to read from sysfs
    FILE *fp;

    fp = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
    if (fp) {
        char size_str[32];
        if (fgets(size_str, sizeof(size_str), fp)) {
            sscanf(size_str, "%dK", l1);
        }
        fclose(fp);
    }

    fp = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
    if (fp) {
        char size_str[32];
        if (fgets(size_str, sizeof(size_str), fp)) {
            sscanf(size_str, "%dK", l2);
        }
        fclose(fp);
    }

    fp = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (fp) {
        char size_str[32];
        if (fgets(size_str, sizeof(size_str), fp)) {
            sscanf(size_str, "%dK", l3);
        }
        fclose(fp);
    }
#endif

#ifdef __APPLE__
    size_t len;
    long long cache_size;

    len = sizeof(cache_size);
    if (sysctlbyname("hw.l1dcachesize", &cache_size, &len, NULL, 0) == 0) {
        *l1 = (int)(cache_size / 1024);
    }

    len = sizeof(cache_size);
    if (sysctlbyname("hw.l2cachesize", &cache_size, &len, NULL, 0) == 0) {
        *l2 = (int)(cache_size / 1024);
    }

    len = sizeof(cache_size);
    if (sysctlbyname("hw.l3cachesize", &cache_size, &len, NULL, 0) == 0) {
        *l3 = (int)(cache_size / 1024);
    }
#endif

    return 0;
}

/**
 * Enhanced CPU detection with C system calls (v0.5)
 * Called from Fortran to get detailed info
 */
void hpcs_cpu_detect_enhanced(hpcs_cpu_info_t *info) {
    int physical_cores = 1;
    int logical_cores = 1;
    int l1_kb = 32;
    int l2_kb = 256;
    int l3_kb = 8192;

    // Initialize structure
    memset(info, 0, sizeof(hpcs_cpu_info_t));

    // Detect cores
    detect_num_cores(&physical_cores, &logical_cores);

    // Detect cache sizes
    detect_cache_sizes(&l1_kb, &l2_kb, &l3_kb);

    // Fill basic structure
    info->num_physical_cores = physical_cores;
    info->num_logical_cores = logical_cores;
    info->l1_cache_size_kb = l1_kb;
    info->l2_cache_size_kb = l2_kb;
    info->l3_cache_size_kb = l3_kb;

    // Set optimal threads (use physical cores)
    info->optimal_threads = physical_cores;

    // If user set OMP_NUM_THREADS, respect it
    const char *omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        int user_threads = atoi(omp_threads);
        if (user_threads > 0 && user_threads <= logical_cores) {
            info->optimal_threads = user_threads;
        }
    }

    // v0.5: Detect SIMD capabilities
    hpcs_detect_simd_capabilities(info);

    // v0.5: Detect NUMA topology (must come after core detection)
    hpcs_detect_numa_topology(info);

    info->initialized = 1;
}

/**
 * Initialize CPU detection (called from C programs)
 */
void hpcs_cpu_init(void) {
    hpcs_cpu_detect_init();
}

// ============================================================================
// v0.5 API Functions
// ============================================================================

/**
 * Set NUMA affinity mode
 */
void hpcs_set_affinity_mode(int mode, int *status) {
    *status = 0;
    if (mode < 0 || mode > 2) {
        *status = -1;
        return;
    }
    g_affinity_mode = mode;
}

/**
 * Get current affinity mode
 */
int hpcs_get_affinity_mode(void) {
    return g_affinity_mode;
}

/**
 * Get SIMD width in bits
 */
int hpcs_get_simd_width(void) {
    hpcs_cpu_info_t info;
    hpcs_cpu_detect_enhanced(&info);
    return info.simd_width_bits;
}

/**
 * Check if AVX is available
 */
int hpcs_has_avx(void) {
    hpcs_cpu_info_t info;
    hpcs_cpu_detect_enhanced(&info);
    return info.has_avx;
}

/**
 * Check if AVX2 is available
 */
int hpcs_has_avx2(void) {
    hpcs_cpu_info_t info;
    hpcs_cpu_detect_enhanced(&info);
    return info.has_avx2;
}

/**
 * Check if AVX-512 is available
 */
int hpcs_has_avx512(void) {
    hpcs_cpu_info_t info;
    hpcs_cpu_detect_enhanced(&info);
    return info.has_avx512;
}

// ============================================================================
// v0.5 Tuning Configuration System
// ============================================================================

/**
 * Generate CPU ID string for cache validation
 */
static void generate_cpu_id(char *cpu_id, size_t size, const hpcs_cpu_info_t *info) {
    snprintf(cpu_id, size, "%s_%dcores_%dL3",
             info->cpu_vendor,
             info->num_physical_cores,
             info->l3_cache_size_kb);
}

/**
 * Get current timestamp (seconds since epoch)
 */
static long long get_timestamp(void) {
    return (long long)time(NULL);
}

/**
 * Initialize tuning configuration with hardware-based defaults
 */
void hpcs_tuning_init(hpcs_tuning_t *tuning) {
    hpcs_cpu_info_t cpu_info;

    // Get CPU information
    hpcs_cpu_detect_enhanced(&cpu_info);

    // Initialize thresholds using existing adaptive logic
    tuning->threshold_simple = hpcs_cpu_get_threshold(1);   // THRESHOLD_SIMPLE_REDUCE
    tuning->threshold_rolling = hpcs_cpu_get_threshold(2);  // THRESHOLD_ROLLING_SIMPLE
    tuning->threshold_robust = hpcs_cpu_get_threshold(3);   // THRESHOLD_COMPUTE_HEAVY
    tuning->threshold_anomaly = hpcs_cpu_get_threshold(4);  // THRESHOLD_ANOMALY_DETECT

    // Initialize thread counts (use optimal threads for all)
    tuning->threads_simple = cpu_info.optimal_threads;
    tuning->threads_rolling = cpu_info.optimal_threads;
    tuning->threads_robust = cpu_info.optimal_threads;
    tuning->threads_anomaly = cpu_info.optimal_threads;

    // Initialize NUMA modes (auto-select by default)
    tuning->numa_mode_simple = 0;   // AUTO
    tuning->numa_mode_rolling = 1;  // COMPACT (good for rolling ops - cache locality)
    tuning->numa_mode_robust = 0;   // AUTO
    tuning->numa_mode_anomaly = 2;  // SPREAD (good for anomaly detection - bandwidth)

    // Generate CPU ID for cache validation
    generate_cpu_id(tuning->cpu_id, sizeof(tuning->cpu_id), &cpu_info);

    // Set timestamp
    tuning->timestamp = get_timestamp();

    // Mark as valid
    tuning->valid = 1;
}

/**
 * Validate tuning configuration
 */
int hpcs_validate_tuning(const hpcs_tuning_t *tuning) {
    if (!tuning->valid) {
        return -1;  // Invalid flag
    }

    // Validate thresholds (must be positive, reasonable range)
    if (tuning->threshold_simple < 1000 || tuning->threshold_simple > 10000000) {
        return -2;  // Invalid threshold_simple
    }
    if (tuning->threshold_rolling < 1000 || tuning->threshold_rolling > 10000000) {
        return -3;  // Invalid threshold_rolling
    }
    if (tuning->threshold_robust < 1000 || tuning->threshold_robust > 10000000) {
        return -4;  // Invalid threshold_robust
    }
    if (tuning->threshold_anomaly < 1000 || tuning->threshold_anomaly > 10000000) {
        return -5;  // Invalid threshold_anomaly
    }

    // Validate thread counts (must be positive, not exceed 1024)
    if (tuning->threads_simple < 1 || tuning->threads_simple > 1024) {
        return -6;  // Invalid threads_simple
    }
    if (tuning->threads_rolling < 1 || tuning->threads_rolling > 1024) {
        return -7;  // Invalid threads_rolling
    }
    if (tuning->threads_robust < 1 || tuning->threads_robust > 1024) {
        return -8;  // Invalid threads_robust
    }
    if (tuning->threads_anomaly < 1 || tuning->threads_anomaly > 1024) {
        return -9;  // Invalid threads_anomaly
    }

    // Validate NUMA modes (must be 0, 1, or 2)
    if (tuning->numa_mode_simple < 0 || tuning->numa_mode_simple > 2) {
        return -10; // Invalid numa_mode_simple
    }
    if (tuning->numa_mode_rolling < 0 || tuning->numa_mode_rolling > 2) {
        return -11; // Invalid numa_mode_rolling
    }
    if (tuning->numa_mode_robust < 0 || tuning->numa_mode_robust > 2) {
        return -12; // Invalid numa_mode_robust
    }
    if (tuning->numa_mode_anomaly < 0 || tuning->numa_mode_anomaly > 2) {
        return -13; // Invalid numa_mode_anomaly
    }

    return 0;  // Valid
}

/**
 * Get current tuning configuration
 */
void hpcs_get_tuning(hpcs_tuning_t *tuning) {
    if (!g_tuning_initialized) {
        hpcs_tuning_init(&g_tuning);
        g_tuning_initialized = 1;
    }

    *tuning = g_tuning;
}

/**
 * Set tuning configuration (validates first)
 */
int hpcs_set_tuning(const hpcs_tuning_t *tuning) {
    int status = hpcs_validate_tuning(tuning);
    if (status != 0) {
        return status;  // Validation failed
    }

    g_tuning = *tuning;
    g_tuning_initialized = 1;

    return 0;  // Success
}

/**
 * Get tuning parameter for specific operation class
 */
int hpcs_get_tuning_threshold(int op_class) {
    if (!g_tuning_initialized) {
        hpcs_tuning_init(&g_tuning);
        g_tuning_initialized = 1;
    }

    switch (op_class) {
        case 1: return g_tuning.threshold_simple;
        case 2: return g_tuning.threshold_rolling;
        case 3: return g_tuning.threshold_robust;
        case 4: return g_tuning.threshold_anomaly;
        default: return g_tuning.threshold_robust;  // Default
    }
}

int hpcs_get_tuning_threads(int op_class) {
    if (!g_tuning_initialized) {
        hpcs_tuning_init(&g_tuning);
        g_tuning_initialized = 1;
    }

    switch (op_class) {
        case 1: return g_tuning.threads_simple;
        case 2: return g_tuning.threads_rolling;
        case 3: return g_tuning.threads_robust;
        case 4: return g_tuning.threads_anomaly;
        default: return g_tuning.threads_robust;  // Default
    }
}

int hpcs_get_tuning_numa_mode(int op_class) {
    if (!g_tuning_initialized) {
        hpcs_tuning_init(&g_tuning);
        g_tuning_initialized = 1;
    }

    switch (op_class) {
        case 1: return g_tuning.numa_mode_simple;
        case 2: return g_tuning.numa_mode_rolling;
        case 3: return g_tuning.numa_mode_robust;
        case 4: return g_tuning.numa_mode_anomaly;
        default: return 0;  // AUTO
    }
}

/**
 * Set tuning parameter for specific operation class
 */
int hpcs_set_tuning_threshold(int op_class, int threshold) {
    if (!g_tuning_initialized) {
        hpcs_tuning_init(&g_tuning);
        g_tuning_initialized = 1;
    }

    // Validate threshold
    if (threshold < 1000 || threshold > 10000000) {
        return -1;  // Invalid threshold
    }

    switch (op_class) {
        case 1: g_tuning.threshold_simple = threshold; break;
        case 2: g_tuning.threshold_rolling = threshold; break;
        case 3: g_tuning.threshold_robust = threshold; break;
        case 4: g_tuning.threshold_anomaly = threshold; break;
        default: return -2;  // Invalid op_class
    }

    return 0;  // Success
}

int hpcs_set_tuning_threads(int op_class, int threads) {
    if (!g_tuning_initialized) {
        hpcs_tuning_init(&g_tuning);
        g_tuning_initialized = 1;
    }

    // Validate threads
    if (threads < 1 || threads > 1024) {
        return -1;  // Invalid thread count
    }

    switch (op_class) {
        case 1: g_tuning.threads_simple = threads; break;
        case 2: g_tuning.threads_rolling = threads; break;
        case 3: g_tuning.threads_robust = threads; break;
        case 4: g_tuning.threads_anomaly = threads; break;
        default: return -2;  // Invalid op_class
    }

    return 0;  // Success
}

int hpcs_set_tuning_numa_mode(int op_class, int numa_mode) {
    if (!g_tuning_initialized) {
        hpcs_tuning_init(&g_tuning);
        g_tuning_initialized = 1;
    }

    // Validate NUMA mode
    if (numa_mode < 0 || numa_mode > 2) {
        return -1;  // Invalid NUMA mode
    }

    switch (op_class) {
        case 1: g_tuning.numa_mode_simple = numa_mode; break;
        case 2: g_tuning.numa_mode_rolling = numa_mode; break;
        case 3: g_tuning.numa_mode_robust = numa_mode; break;
        case 4: g_tuning.numa_mode_anomaly = numa_mode; break;
        default: return -2;  // Invalid op_class
    }

    return 0;  // Success
}

/**
 * Reset tuning to hardware-based defaults
 */
void hpcs_reset_tuning(void) {
    hpcs_tuning_init(&g_tuning);
    g_tuning_initialized = 1;
}

/**
 * Print current tuning configuration
 */
void hpcs_print_tuning(const hpcs_tuning_t *tuning) {
    printf("=== HPCS Tuning Configuration ===\n");
    printf("\n");
    printf("CPU ID: %s\n", tuning->cpu_id);
    printf("Timestamp: %lld\n", tuning->timestamp);
    printf("Valid: %s\n", tuning->valid ? "Yes" : "No");
    printf("\n");

    printf("Thresholds:\n");
    printf("  Simple:    %d elements\n", tuning->threshold_simple);
    printf("  Rolling:   %d elements\n", tuning->threshold_rolling);
    printf("  Robust:    %d elements\n", tuning->threshold_robust);
    printf("  Anomaly:   %d elements\n", tuning->threshold_anomaly);
    printf("\n");

    printf("Thread Counts:\n");
    printf("  Simple:    %d threads\n", tuning->threads_simple);
    printf("  Rolling:   %d threads\n", tuning->threads_rolling);
    printf("  Robust:    %d threads\n", tuning->threads_robust);
    printf("  Anomaly:   %d threads\n", tuning->threads_anomaly);
    printf("\n");

    printf("NUMA Modes:\n");
    const char *mode_names[] = {"AUTO", "COMPACT", "SPREAD"};
    printf("  Simple:    %s\n", mode_names[tuning->numa_mode_simple]);
    printf("  Rolling:   %s\n", mode_names[tuning->numa_mode_rolling]);
    printf("  Robust:    %s\n", mode_names[tuning->numa_mode_robust]);
    printf("  Anomaly:   %s\n", mode_names[tuning->numa_mode_anomaly]);
    printf("\n");
    printf("=================================\n");
}

// ============================================================================
// v0.5 NUMA Affinity Application Functions (Linux pthread)
// ============================================================================

#ifdef __linux__

/**
 * Apply compact thread affinity - pin threads to cores on same NUMA node
 *
 * Strategy: Keep all threads on the same NUMA node for best cache locality
 * and minimal cross-node memory traffic. Good for operations with high
 * data reuse (e.g., rolling operations).
 *
 * @param num_threads - Number of threads to configure
 * @param preferred_node - Preferred NUMA node (or -1 for auto-select)
 * @return 0 on success, -1 on error
 */
int hpcs_apply_compact_affinity(int num_threads, int preferred_node) {
    hpcs_cpu_info_t cpu_info;
    cpu_set_t cpuset;
    int status = 0;

    // Get CPU information
    hpcs_cpu_detect_enhanced(&cpu_info);

    // If single NUMA node, nothing to do
    if (cpu_info.numa_nodes <= 1) {
        return 0;  // Success - no NUMA to optimize
    }

    // Auto-select node 0 if not specified
    if (preferred_node < 0) {
        preferred_node = 0;
    }

    // Validate preferred node
    if (preferred_node >= cpu_info.numa_nodes) {
        preferred_node = 0;  // Fallback to node 0
    }

    // Calculate core range for this NUMA node
    int cores_per_node = cpu_info.cores_per_numa_node;
    int start_core = preferred_node * cores_per_node;
    int end_core = start_core + cores_per_node;

    // Limit to available cores
    if (end_core > cpu_info.num_physical_cores) {
        end_core = cpu_info.num_physical_cores;
    }

    // Build CPU set for the target NUMA node
    CPU_ZERO(&cpuset);
    for (int core = start_core; core < end_core; core++) {
        CPU_SET(core, &cpuset);
    }

    // Apply affinity to current thread (master thread)
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        status = -1;  // Failed to set affinity
    }

    return status;
}

/**
 * Apply spread thread affinity - distribute threads across NUMA nodes
 *
 * Strategy: Distribute threads evenly across NUMA nodes to maximize
 * memory bandwidth. Good for operations with low data reuse and high
 * bandwidth requirements (e.g., anomaly detection).
 *
 * @param num_threads - Number of threads to configure
 * @return 0 on success, -1 on error
 */
int hpcs_apply_spread_affinity(int num_threads) {
    hpcs_cpu_info_t cpu_info;
    cpu_set_t cpuset;
    int status = 0;

    // Get CPU information
    hpcs_cpu_detect_enhanced(&cpu_info);

    // If single NUMA node, nothing to do
    if (cpu_info.numa_nodes <= 1) {
        return 0;  // Success - no NUMA to optimize
    }

    // Build CPU set spanning all NUMA nodes
    CPU_ZERO(&cpuset);

    // Distribute threads across nodes in round-robin fashion
    int cores_per_node = cpu_info.cores_per_numa_node;
    int threads_per_node = (num_threads + cpu_info.numa_nodes - 1) / cpu_info.numa_nodes;

    for (int node = 0; node < cpu_info.numa_nodes; node++) {
        int start_core = node * cores_per_node;
        int node_threads = (node < cpu_info.numa_nodes - 1) ? threads_per_node :
                          (num_threads - node * threads_per_node);

        // Add cores from this node
        for (int i = 0; i < node_threads && (start_core + i) < cpu_info.num_physical_cores; i++) {
            CPU_SET(start_core + i, &cpuset);
        }
    }

    // Apply affinity to current thread (master thread)
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        status = -1;  // Failed to set affinity
    }

    return status;
}

/**
 * Apply NUMA affinity based on current affinity mode and operation type
 *
 * Automatically selects the best affinity strategy based on:
 * - Global affinity mode (AUTO, COMPACT, SPREAD)
 * - Operation class (simple, rolling, robust, anomaly)
 *
 * @param num_threads - Number of threads to configure
 * @param op_class - Operation class (1=simple, 2=rolling, 3=robust, 4=anomaly)
 * @return 0 on success, -1 on error
 */
int hpcs_apply_numa_affinity(int num_threads, int op_class) {
    int affinity_mode = g_affinity_mode;

    // Auto-select mode if needed
    if (affinity_mode == 0) {  // AFFINITY_AUTO
        int numa_mode = hpcs_get_tuning_numa_mode(op_class);

        switch (numa_mode) {
            case 1:  // COMPACT
                return hpcs_apply_compact_affinity(num_threads, -1);
            case 2:  // SPREAD
                return hpcs_apply_spread_affinity(num_threads);
            default:  // AUTO - use heuristic
                // Rolling ops benefit from compact (cache locality)
                // Anomaly detection benefits from spread (bandwidth)
                if (op_class == 2) {  // Rolling
                    return hpcs_apply_compact_affinity(num_threads, -1);
                } else if (op_class == 4) {  // Anomaly
                    return hpcs_apply_spread_affinity(num_threads);
                }
                // Default: no affinity change
                return 0;
        }
    }

    // Apply explicit mode
    switch (affinity_mode) {
        case 1:  // COMPACT
            return hpcs_apply_compact_affinity(num_threads, -1);
        case 2:  // SPREAD
            return hpcs_apply_spread_affinity(num_threads);
        default:
            return 0;  // No affinity
    }
}

/**
 * Get affinity for specific core (for debugging/testing)
 *
 * @param core_id - Core ID to query
 * @param numa_node - Output: NUMA node for this core
 * @return 0 on success, -1 on error
 */
int hpcs_get_core_affinity(int core_id, int *numa_node) {
    hpcs_cpu_info_t cpu_info;

    hpcs_cpu_detect_enhanced(&cpu_info);

    if (core_id < 0 || core_id >= cpu_info.num_physical_cores) {
        return -1;  // Invalid core ID
    }

    // Calculate NUMA node (simple distribution)
    if (cpu_info.numa_nodes > 1) {
        *numa_node = core_id / cpu_info.cores_per_numa_node;
    } else {
        *numa_node = 0;
    }

    return 0;
}

#else
// Non-Linux platforms: stub implementations

int hpcs_apply_compact_affinity(int num_threads, int preferred_node) {
    (void)num_threads;
    (void)preferred_node;
    return 0;  // No-op on non-Linux
}

int hpcs_apply_spread_affinity(int num_threads) {
    (void)num_threads;
    return 0;  // No-op on non-Linux
}

int hpcs_apply_numa_affinity(int num_threads, int op_class) {
    (void)num_threads;
    (void)op_class;
    return 0;  // No-op on non-Linux
}

int hpcs_get_core_affinity(int core_id, int *numa_node) {
    (void)core_id;
    *numa_node = 0;
    return 0;  // Single node on non-Linux
}

#endif  // __linux__
