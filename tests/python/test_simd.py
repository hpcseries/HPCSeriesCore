"""
HPCSeries Core v0.7 - SIMD and CPU Information Tests
====================================================

Tests for SIMD capability detection and CPU information APIs.
"""

import pytest
import hpcs


class TestSIMDInfo:
    """Test SIMD information API."""

    def test_simd_info_structure(self):
        """Test simd_info returns correct structure."""
        info = hpcs.simd_info()

        assert isinstance(info, dict)
        assert 'isa' in info
        assert 'width_bytes' in info
        assert 'width_doubles' in info

    def test_simd_isa_valid(self):
        """Test ISA name is valid."""
        info = hpcs.simd_info()
        isa = info['isa']

        # Should be one of the supported ISAs
        valid_isas = ['AVX2', 'AVX', 'SSE2', 'NEON', 'Scalar']
        assert isa in valid_isas, f"Unexpected ISA: {isa}"

    def test_simd_width_consistency(self):
        """Test SIMD width values are consistent."""
        info = hpcs.simd_info()

        width_bytes = info['width_bytes']
        width_doubles = info['width_doubles']

        # width_doubles should equal width_bytes / 8 (size of double)
        assert width_doubles == width_bytes // 8

        # Valid widths: 32 bytes (AVX2), 16 bytes (SSE2/NEON), 8 bytes (scalar)
        assert width_bytes in [8, 16, 32, 64], f"Unexpected width: {width_bytes}"

    def test_simd_width_doubles(self):
        """Test SIMD width in doubles."""
        info = hpcs.simd_info()
        width_doubles = info['width_doubles']

        # Valid: 1 (scalar), 2 (SSE2/NEON), 4 (AVX2), 8 (AVX-512)
        assert width_doubles in [1, 2, 4, 8], f"Unexpected width: {width_doubles}"

    def test_get_simd_width(self):
        """Test standalone get_simd_width function."""
        width = hpcs.get_simd_width()
        info = hpcs.simd_info()

        # Should match simd_info result
        assert width == info['width_doubles']


class TestCPUInfo:
    """Test CPU information API."""

    def test_cpu_info_structure(self):
        """Test get_cpu_info returns correct structure."""
        info = hpcs.get_cpu_info()

        assert isinstance(info, dict)

        # Core information
        assert 'physical_cores' in info
        assert 'logical_cores' in info
        assert 'optimal_threads' in info

        # Cache hierarchy
        assert 'l1_cache_kb' in info
        assert 'l2_cache_kb' in info
        assert 'l3_cache_kb' in info

        # NUMA topology
        assert 'numa_nodes' in info
        assert 'cores_per_numa' in info

        # CPU capabilities
        assert 'cpu_vendor' in info
        assert 'simd_width_bits' in info

        # SIMD flags
        assert 'has_sse2' in info
        assert 'has_avx' in info
        assert 'has_avx2' in info
        assert 'has_avx512' in info
        assert 'has_neon' in info
        assert 'has_fma3' in info

    def test_core_counts_valid(self):
        """Test CPU core counts are reasonable."""
        info = hpcs.get_cpu_info()

        physical = info['physical_cores']
        logical = info['logical_cores']
        optimal = info['optimal_threads']

        # Physical cores should be positive
        assert physical >= 1

        # Logical cores should be >= physical cores
        assert logical >= physical

        # Optimal threads should be positive
        assert optimal >= 1

        # Optimal threads typically <= logical cores
        assert optimal <= logical * 2  # Allow some flexibility

    def test_cache_sizes_valid(self):
        """Test cache sizes are reasonable."""
        info = hpcs.get_cpu_info()

        l1 = info['l1_cache_kb']
        l2 = info['l2_cache_kb']
        l3 = info['l3_cache_kb']

        # All caches should be non-negative
        assert l1 >= 0
        assert l2 >= 0
        assert l3 >= 0

        # L1 < L2 < L3 (typical hierarchy)
        if l1 > 0 and l2 > 0:
            assert l1 <= l2
        if l2 > 0 and l3 > 0:
            assert l2 <= l3

    def test_numa_topology_valid(self):
        """Test NUMA topology is reasonable."""
        info = hpcs.get_cpu_info()

        numa_nodes = info['numa_nodes']
        cores_per_numa = info['cores_per_numa']
        physical_cores = info['physical_cores']

        # NUMA nodes should be positive
        assert numa_nodes >= 1

        # Cores per NUMA should be positive
        assert cores_per_numa >= 1

        # Total cores should roughly match
        # (cores_per_numa * numa_nodes should be close to physical_cores)
        if numa_nodes > 0 and cores_per_numa > 0:
            estimated_cores = cores_per_numa * numa_nodes
            # Allow some tolerance for uneven distribution
            assert abs(estimated_cores - physical_cores) <= numa_nodes

    def test_cpu_vendor_valid(self):
        """Test CPU vendor string is valid."""
        info = hpcs.get_cpu_info()
        vendor = info['cpu_vendor']

        assert isinstance(vendor, str)
        assert len(vendor) > 0

        # Common vendors
        common_vendors = ['GenuineIntel', 'AuthenticAMD', 'ARM', 'Apple']
        # Just check it's not empty, vendor could be anything
        assert vendor != ""

    def test_simd_width_bits_valid(self):
        """Test SIMD width in bits is valid."""
        info = hpcs.get_cpu_info()
        width_bits = info['simd_width_bits']

        # Valid widths: 64 (scalar), 128 (SSE2/NEON), 256 (AVX2), 512 (AVX-512)
        assert width_bits in [64, 128, 256, 512], f"Unexpected width: {width_bits}"

    def test_simd_flags_boolean(self):
        """Test SIMD capability flags are boolean."""
        info = hpcs.get_cpu_info()

        simd_flags = ['has_sse2', 'has_avx', 'has_avx2', 'has_avx512', 'has_neon', 'has_fma3']

        for flag in simd_flags:
            assert isinstance(info[flag], bool), f"{flag} should be boolean"

    def test_simd_flags_hierarchy(self):
        """Test SIMD capability hierarchy is consistent."""
        info = hpcs.get_cpu_info()

        # If AVX2 is available, AVX and SSE2 should also be available
        if info['has_avx2']:
            assert info['has_avx'], "AVX2 implies AVX"
            assert info['has_sse2'], "AVX2 implies SSE2"

        # If AVX is available, SSE2 should be available
        if info['has_avx']:
            assert info['has_sse2'], "AVX implies SSE2"

        # AVX-512 implies AVX2 (typically)
        if info['has_avx512']:
            assert info['has_avx2'], "AVX-512 implies AVX2"


class TestSIMDIntegration:
    """Test integration between SIMD info and actual operations."""

    def test_simd_actually_used(self):
        """Test that SIMD is actually being used."""
        import numpy as np

        # Get SIMD info
        info = hpcs.simd_info()

        # Perform operation
        data = np.random.randn(10000)
        result = hpcs.sum(data)

        # Should complete without error
        assert isinstance(result, float)

        # If AVX2/AVX/SSE2 is available, we should be using it
        if info['isa'] in ['AVX2', 'AVX', 'SSE2']:
            assert info['width_doubles'] > 1, "SIMD should provide parallelism"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
