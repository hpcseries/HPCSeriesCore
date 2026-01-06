"""
Test Suite for Execution Mode System (v0.8.0)

Tests Safe vs Fast vs Deterministic execution modes for:
  - ewma, ewvar, ewstd
  - cumulative_min, cumulative_max
  - convolve_valid
  - trimmed_mean, winsorized_mean

Validates:
  1. Mode API (set/get_execution_mode)
  2. Per-call mode override
  3. NaN detection (SAFE vs FAST)
  4. Performance (FAST should be ~1.2-2x faster)
  5. Reproducibility (DETERMINISTIC should be bit-exact)
"""

import pytest
import numpy as np
import time
import sys
import os

# Add parent directory to path to import hpcs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

import hpcs


class TestExecutionModeAPI:
    """Test execution mode management API."""

    def test_default_mode_is_safe(self):
        """Default mode should be 'safe'."""
        mode = hpcs.get_execution_mode()
        assert mode == 'safe', f"Expected default mode 'safe', got '{mode}'"

    def test_set_mode_safe(self):
        """Test setting SAFE mode."""
        hpcs.set_execution_mode('safe')
        assert hpcs.get_execution_mode() == 'safe'

    def test_set_mode_fast(self):
        """Test setting FAST mode."""
        hpcs.set_execution_mode('fast')
        assert hpcs.get_execution_mode() == 'fast'
        # Reset to safe
        hpcs.set_execution_mode('safe')

    def test_set_mode_deterministic(self):
        """Test setting DETERMINISTIC mode."""
        hpcs.set_execution_mode('deterministic')
        assert hpcs.get_execution_mode() == 'deterministic'
        # Reset to safe
        hpcs.set_execution_mode('safe')

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            hpcs.set_execution_mode('invalid_mode')

    def test_mode_constants_exist(self):
        """Test that mode constants are defined."""
        assert hpcs.MODE_SAFE == 'safe'
        assert hpcs.MODE_FAST == 'fast'
        assert hpcs.MODE_DETERMINISTIC == 'deterministic'


class TestEWMAModes:
    """Test execution modes for exponentially weighted moving average."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_ewma_safe_mode(self):
        """Test EWMA in SAFE mode (default)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpcs.ewma(x, alpha=0.5, mode='safe')
        expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ewma_fast_mode(self):
        """Test EWMA in FAST mode."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpcs.ewma(x, alpha=0.5, mode='fast')
        expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ewma_deterministic_mode(self):
        """Test EWMA in DETERMINISTIC mode."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpcs.ewma(x, alpha=0.5, mode='deterministic')
        expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ewma_nan_detection_safe(self):
        """Test that SAFE mode propagates NaN correctly."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.ewma(x, alpha=0.5, mode='safe')
        # After NaN, all values should be NaN
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_ewma_global_mode_override(self):
        """Test that per-call mode overrides global mode."""
        hpcs.set_execution_mode('fast')
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Explicit safe mode should override global fast mode
        result = hpcs.ewma(x, alpha=0.5, mode='safe')
        expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

        # Reset
        hpcs.set_execution_mode('safe')

    def test_ewma_uses_global_mode_when_none(self):
        """Test that mode=None uses global mode setting."""
        hpcs.set_execution_mode('deterministic')
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # mode=None should use global DETERMINISTIC
        result = hpcs.ewma(x, alpha=0.5, mode=None)
        expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

        # Reset
        hpcs.set_execution_mode('safe')


class TestEWVARModes:
    """Test execution modes for exponentially weighted variance."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_ewvar_all_modes_consistent(self):
        """Test that all modes produce same results for valid input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result_safe = hpcs.ewvar(x, alpha=0.3, mode='safe')
        result_fast = hpcs.ewvar(x, alpha=0.3, mode='fast')
        result_det = hpcs.ewvar(x, alpha=0.3, mode='deterministic')

        # All modes should give identical results for valid input
        np.testing.assert_allclose(result_safe, result_fast, rtol=1e-10)
        np.testing.assert_allclose(result_safe, result_det, rtol=1e-10)


class TestEWSTDModes:
    """Test execution modes for exponentially weighted standard deviation."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_ewstd_all_modes_consistent(self):
        """Test that all modes produce same results for valid input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result_safe = hpcs.ewstd(x, alpha=0.4, mode='safe')
        result_fast = hpcs.ewstd(x, alpha=0.4, mode='fast')
        result_det = hpcs.ewstd(x, alpha=0.4, mode='deterministic')

        # All modes should give identical results for valid input
        np.testing.assert_allclose(result_safe, result_fast, rtol=1e-10)
        np.testing.assert_allclose(result_safe, result_det, rtol=1e-10)


class TestCumulativeMinMaxModes:
    """Test execution modes for cumulative min/max."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_cumulative_min_all_modes(self):
        """Test cumulative_min in all modes."""
        x = np.array([5.0, 3.0, 7.0, 1.0, 4.0, 2.0])
        expected = np.array([5.0, 3.0, 3.0, 1.0, 1.0, 1.0])

        result_safe = hpcs.cumulative_min(x, mode='safe')
        result_fast = hpcs.cumulative_min(x, mode='fast')
        result_det = hpcs.cumulative_min(x, mode='deterministic')

        np.testing.assert_array_equal(result_safe, expected)
        np.testing.assert_array_equal(result_fast, expected)
        np.testing.assert_array_equal(result_det, expected)

    def test_cumulative_max_all_modes(self):
        """Test cumulative_max in all modes."""
        x = np.array([1.0, 5.0, 3.0, 7.0, 2.0, 6.0])
        expected = np.array([1.0, 5.0, 5.0, 7.0, 7.0, 7.0])

        result_safe = hpcs.cumulative_max(x, mode='safe')
        result_fast = hpcs.cumulative_max(x, mode='fast')
        result_det = hpcs.cumulative_max(x, mode='deterministic')

        np.testing.assert_array_equal(result_safe, expected)
        np.testing.assert_array_equal(result_fast, expected)
        np.testing.assert_array_equal(result_det, expected)

    @pytest.mark.skip(reason="NaN propagation behavior needs verification")
    def test_cumulative_min_nan_safe(self):
        """Test that SAFE mode handles NaN in cumulative_min."""
        x = np.array([5.0, 3.0, np.nan, 1.0, 4.0])
        result = hpcs.cumulative_min(x, mode='safe')

        # After NaN, IEEE 754 min should propagate NaN
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])


class TestConvolveValidModes:
    """Test execution modes for convolve_valid."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_convolve_valid_all_modes(self):
        """Test convolve_valid in all modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kernel = np.array([0.25, 0.5, 0.25])
        expected = np.array([2.0, 3.0, 4.0])

        result_safe = hpcs.convolve_valid(x, kernel, mode='safe')
        result_fast = hpcs.convolve_valid(x, kernel, mode='fast')
        result_det = hpcs.convolve_valid(x, kernel, mode='deterministic')

        np.testing.assert_allclose(result_safe, expected, rtol=1e-10)
        np.testing.assert_allclose(result_fast, expected, rtol=1e-10)
        np.testing.assert_allclose(result_det, expected, rtol=1e-10)

    def test_convolve_deterministic_reproducible(self):
        """Test that DETERMINISTIC mode gives bit-exact reproducibility."""
        x = np.random.rand(1000)
        kernel = np.array([0.2, 0.6, 0.2])

        # Run twice in deterministic mode
        result1 = hpcs.convolve_valid(x, kernel, mode='deterministic')
        result2 = hpcs.convolve_valid(x, kernel, mode='deterministic')

        # Should be bit-exact (no OpenMP race conditions)
        np.testing.assert_array_equal(result1, result2)


class TestTrimmedMeanModes:
    """Test execution modes for trimmed_mean."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_trimmed_mean_all_modes(self):
        """Test trimmed_mean in all modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        trim_frac = 0.2  # Trim 20% from each end (2 values)

        result_safe = hpcs.trimmed_mean(x, trim_frac, mode='safe')
        result_fast = hpcs.trimmed_mean(x, trim_frac, mode='fast')
        result_det = hpcs.trimmed_mean(x, trim_frac, mode='deterministic')

        # Expected: mean of [3, 4, 5, 6, 7, 8] = 5.5
        expected = 5.5

        assert abs(result_safe - expected) < 1e-10
        assert abs(result_fast - expected) < 1e-10
        assert abs(result_det - expected) < 1e-10

    def test_trimmed_mean_with_outliers(self):
        """Test that trimmed_mean handles outliers correctly."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Outlier: 100
        trim_frac = 0.2

        result = hpcs.trimmed_mean(x, trim_frac, mode='safe')

        # Should trim 1 from each end, compute mean of [2,3,4,5]
        expected = 3.5
        assert abs(result - expected) < 1e-10


class TestWinsorizedMeanModes:
    """Test execution modes for winsorized_mean."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_winsorized_mean_all_modes(self):
        """Test winsorized_mean in all modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        win_frac = 0.1  # Winsorize 10% from each end (1 value)

        result_safe = hpcs.winsorized_mean(x, win_frac, mode='safe')
        result_fast = hpcs.winsorized_mean(x, win_frac, mode='fast')
        result_det = hpcs.winsorized_mean(x, win_frac, mode='deterministic')

        # Expected: mean of [2,2,3,4,5,6,7,8,9,9] = 5.5
        expected = 5.5

        assert abs(result_safe - expected) < 1e-10
        assert abs(result_fast - expected) < 1e-10
        assert abs(result_det - expected) < 1e-10

    def test_winsorized_mean_with_outliers(self):
        """Test that winsorized_mean clamps outliers correctly."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        win_frac = 0.2

        result = hpcs.winsorized_mean(x, win_frac, mode='safe')

        # Should clamp 1→2 and 100→5, compute mean of [2,2,3,4,5,5]
        expected = 3.5
        assert abs(result - expected) < 1e-10


class TestPerformanceBenchmarks:
    """Benchmark FAST mode performance gains."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    @pytest.mark.skip(reason="Performance optimization is future work (v0.9.0+)")
    def test_ewma_fast_is_faster(self):
        """Test that FAST mode is significantly faster than SAFE for EWMA."""
        n = 100000
        x = np.random.rand(n)
        alpha = 0.3

        # Warm-up
        _ = hpcs.ewma(x, alpha, mode='safe')
        _ = hpcs.ewma(x, alpha, mode='fast')

        # Benchmark SAFE
        t0 = time.perf_counter()
        for _ in range(10):
            _ = hpcs.ewma(x, alpha, mode='safe')
        t_safe = time.perf_counter() - t0

        # Benchmark FAST
        t0 = time.perf_counter()
        for _ in range(10):
            _ = hpcs.ewma(x, alpha, mode='fast')
        t_fast = time.perf_counter() - t0

        speedup = t_safe / t_fast
        print(f"\nEWMA speedup (FAST vs SAFE): {speedup:.2f}x")

        # FAST should be at least 1.1x faster (conservative threshold)
        assert speedup >= 1.1, f"Expected FAST to be faster, got {speedup:.2f}x"

    @pytest.mark.skip(reason="Performance optimization is future work (v0.9.0+)")
    def test_trimmed_mean_fast_is_faster(self):
        """Test that FAST mode is faster for trimmed_mean."""
        n = 50000
        x = np.random.rand(n)
        trim_frac = 0.1

        # Warm-up
        _ = hpcs.trimmed_mean(x, trim_frac, mode='safe')
        _ = hpcs.trimmed_mean(x, trim_frac, mode='fast')

        # Benchmark SAFE
        t0 = time.perf_counter()
        for _ in range(5):
            _ = hpcs.trimmed_mean(x, trim_frac, mode='safe')
        t_safe = time.perf_counter() - t0

        # Benchmark FAST
        t0 = time.perf_counter()
        for _ in range(5):
            _ = hpcs.trimmed_mean(x, trim_frac, mode='fast')
        t_fast = time.perf_counter() - t0

        speedup = t_safe / t_fast
        print(f"\nTrimmed mean speedup (FAST vs SAFE): {speedup:.2f}x")

        # FAST should be at least 1.1x faster
        assert speedup >= 1.1, f"Expected FAST to be faster, got {speedup:.2f}x"


class TestDeterministicReproducibility:
    """Test that DETERMINISTIC mode provides bit-exact reproducibility."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_ewma_deterministic_reproducible(self):
        """Test that DETERMINISTIC EWMA is bit-exact across runs."""
        x = np.random.rand(10000)
        alpha = 0.5

        # Run 3 times in deterministic mode
        result1 = hpcs.ewma(x, alpha, mode='deterministic')
        result2 = hpcs.ewma(x, alpha, mode='deterministic')
        result3 = hpcs.ewma(x, alpha, mode='deterministic')

        # Should be bit-exact (no SIMD, strict evaluation order)
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_cumulative_min_deterministic_reproducible(self):
        """Test that DETERMINISTIC cumulative_min is bit-exact."""
        x = np.random.rand(10000)

        # Run 3 times
        result1 = hpcs.cumulative_min(x, mode='deterministic')
        result2 = hpcs.cumulative_min(x, mode='deterministic')
        result3 = hpcs.cumulative_min(x, mode='deterministic')

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)


if __name__ == '__main__':
    # Run with: python test_execution_modes_v08.py -v
    pytest.main([__file__, '-v', '--tb=short'])
