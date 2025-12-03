# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""
HPCSeries Core v0.7 - SIMD & CPU Information
=============================================

Provides runtime information about SIMD capabilities and CPU topology.
"""

cdef extern from "hpcs_cpu_detect.h":
    ctypedef struct hpcs_cpu_info_t:
        int num_physical_cores
        int num_logical_cores
        int l1_cache_size_kb
        int l2_cache_size_kb
        int l3_cache_size_kb
        int optimal_threads
        int numa_nodes
        int cores_per_numa_node
        int has_sse2
        int has_avx
        int has_avx2
        int has_avx512
        int has_neon
        int has_fma3
        int simd_width_bits
        char cpu_vendor[64]
        char cpu_model[128]
        int initialized

    void hpcs_cpu_detect_init()
    void hpcs_cpu_detect_enhanced(hpcs_cpu_info_t *info)

cdef extern from "hpcs_simd_dispatch.h":
    const char* hpcs_get_simd_name()
    int hpcs_get_simd_width_doubles()
    int hpcs_get_simd_width_bytes()
    void hpcs_print_simd_status()

def simd_info():
    """
    Get SIMD instruction set information.

    Returns
    -------
    info : dict
        Dictionary containing:
        - 'isa': SIMD ISA name (e.g., 'AVX2', 'SSE2', 'NEON')
        - 'width_bytes': SIMD vector width in bytes
        - 'width_doubles': Number of doubles per SIMD vector

    Examples
    --------
    >>> import hpcs
    >>> info = hpcs.simd_info()
    >>> print(info['isa'])
    'AVX2'
    >>> print(info['width_doubles'])
    4
    """
    cdef const char* isa_name = hpcs_get_simd_name()
    cdef int width_bytes = hpcs_get_simd_width_bytes()
    cdef int width_doubles = hpcs_get_simd_width_doubles()

    return {
        'isa': isa_name.decode('utf-8'),
        'width_bytes': width_bytes,
        'width_doubles': width_doubles,
    }

def get_simd_width():
    """
    Get SIMD vector width in number of doubles.

    Returns
    -------
    width : int
        Number of double-precision values per SIMD vector (2, 4, or 8)

    Examples
    --------
    >>> import hpcs
    >>> hpcs.get_simd_width()
    4  # AVX2 processes 4 doubles at once
    """
    return hpcs_get_simd_width_doubles()

def get_cpu_info():
    """
    Get comprehensive CPU topology and capabilities information.

    Returns
    -------
    info : dict
        Dictionary containing:
        - 'physical_cores': Number of physical CPU cores
        - 'logical_cores': Number of logical cores (with hyperthreading)
        - 'optimal_threads': Recommended thread count for OpenMP
        - 'l1_cache_kb': L1 cache size in KB
        - 'l2_cache_kb': L2 cache size in KB
        - 'l3_cache_kb': L3 cache size in KB
        - 'numa_nodes': Number of NUMA nodes
        - 'cores_per_numa': Cores per NUMA node
        - 'cpu_vendor': CPU vendor string
        - 'simd_width_bits': SIMD vector width in bits
        - 'has_sse2': Boolean - SSE2 support
        - 'has_avx': Boolean - AVX support
        - 'has_avx2': Boolean - AVX2 support
        - 'has_avx512': Boolean - AVX-512 support
        - 'has_neon': Boolean - ARM NEON support
        - 'has_fma3': Boolean - FMA3 support

    Examples
    --------
    >>> import hpcs
    >>> info = hpcs.get_cpu_info()
    >>> print(f"CPU: {info['cpu_vendor']}")
    >>> print(f"Cores: {info['physical_cores']} ({info['logical_cores']} threads)")
    >>> print(f"SIMD: {info['simd_width_bits']}-bit")
    """
    cdef hpcs_cpu_info_t cpu_info

    # Initialize CPU detection
    hpcs_cpu_detect_init()
    hpcs_cpu_detect_enhanced(&cpu_info)

    return {
        'physical_cores': cpu_info.num_physical_cores,
        'logical_cores': cpu_info.num_logical_cores,
        'optimal_threads': cpu_info.optimal_threads,
        'l1_cache_kb': cpu_info.l1_cache_size_kb,
        'l2_cache_kb': cpu_info.l2_cache_size_kb,
        'l3_cache_kb': cpu_info.l3_cache_size_kb,
        'numa_nodes': cpu_info.numa_nodes,
        'cores_per_numa': cpu_info.cores_per_numa_node,
        'cpu_vendor': cpu_info.cpu_vendor.decode('utf-8'),
        'simd_width_bits': cpu_info.simd_width_bits,
        'has_sse2': bool(cpu_info.has_sse2),
        'has_avx': bool(cpu_info.has_avx),
        'has_avx2': bool(cpu_info.has_avx2),
        'has_avx512': bool(cpu_info.has_avx512),
        'has_neon': bool(cpu_info.has_neon),
        'has_fma3': bool(cpu_info.has_fma3),
    }
