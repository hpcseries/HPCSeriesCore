"""
Test Suite for Phase 2 Robust Statistics with Execution Modes (v0.8.0)

Tests Safe vs Fast vs Deterministic execution modes for 6 robust statistics functions:
  - median, mad, quantile
  - rolling_median, rolling_mad, rolling_robust_zscore

Validates:
  1. Mode API (set/get_execution_mode)
  2. Per-call mode override
  3. NaN detection (SAFE vs FAST)
  4. Correctness across all modes
  5. Rolling operations with mode support
  6. Invalid argument handling
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import hpcs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

import hpcs


class TestBasicRobustStatsModes:
    """Test execution modes for basic robust statistics functions."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_median_all_modes(self):
        """Test median with all execution modes."""
        x = np.array([5.0, 3.0, 7.0, 1.0, 9.0])
        expected = 5.0

        # Test SAFE mode
        result = hpcs.median(x, mode='safe')
        assert np.isclose(result, expected), f"median SAFE: expected {expected}, got {result}"

        # Test FAST mode
        result = hpcs.median(x, mode='fast')
        assert np.isclose(result, expected), f"median FAST: expected {expected}, got {result}"

        # Test DETERMINISTIC mode
        result = hpcs.median(x, mode='deterministic')
        assert np.isclose(result, expected), f"median DETERMINISTIC: expected {expected}, got {result}"

        # Test with global mode
        result = hpcs.median(x)
        assert np.isclose(result, expected), f"median with global mode: expected {expected}, got {result}"

    def test_mad_all_modes(self):
        """Test MAD with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 1.0  # MAD: median=3, deviations={2,1,0,1,2}, MAD=1

        assert np.isclose(hpcs.mad(x, mode='safe'), expected)
        assert np.isclose(hpcs.mad(x, mode='fast'), expected)
        assert np.isclose(hpcs.mad(x, mode='deterministic'), expected)

    def test_quantile_all_modes(self):
        """Test quantile with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test median (50th percentile)
        expected_median = 3.0
        assert np.isclose(hpcs.quantile(x, 0.5, mode='safe'), expected_median)
        assert np.isclose(hpcs.quantile(x, 0.5, mode='fast'), expected_median)
        assert np.isclose(hpcs.quantile(x, 0.5, mode='deterministic'), expected_median)

        # Test 25th percentile
        expected_q25 = 2.0
        assert np.isclose(hpcs.quantile(x, 0.25, mode='safe'), expected_q25)

        # Test 75th percentile
        expected_q75 = 4.0
        assert np.isclose(hpcs.quantile(x, 0.75, mode='fast'), expected_q75)


class TestRollingRobustStatsModes:
    """Test execution modes for rolling robust statistics functions."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_rolling_median_all_modes(self):
        """Test rolling_median with all execution modes."""
        x = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0])
        window = 3

        result_safe = hpcs.rolling_median(x, window, mode='safe')
        result_fast = hpcs.rolling_median(x, window, mode='fast')
        result_det = hpcs.rolling_median(x, window, mode='deterministic')

        # First two elements should be NaN
        assert np.isnan(result_safe[0])
        assert np.isnan(result_safe[1])
        assert np.isnan(result_fast[0])
        assert np.isnan(result_det[0])

        # result[2] = median of {1.0, 5.0, 2.0} = 2.0
        np.testing.assert_allclose(result_safe[2], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result_fast[2], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result_det[2], 2.0, rtol=1e-10)

    def test_rolling_mad_all_modes(self):
        """Test rolling_mad with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        window = 3

        result_safe = hpcs.rolling_mad(x, window, mode='safe')
        result_fast = hpcs.rolling_mad(x, window, mode='fast')
        result_det = hpcs.rolling_mad(x, window, mode='deterministic')

        # First two elements should be NaN
        assert np.isnan(result_safe[0])
        assert np.isnan(result_safe[1])

        # result[2] = MAD of {1,2,3}, median=2, deviations={1,0,1}, MAD=1.0
        np.testing.assert_allclose(result_safe[2], 1.0, rtol=1e-10)
        np.testing.assert_allclose(result_fast[2], 1.0, rtol=1e-10)
        np.testing.assert_allclose(result_det[2], 1.0, rtol=1e-10)

    def test_rolling_robust_zscore_all_modes(self):
        """Test rolling_robust_zscore with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        window = 3

        result_safe = hpcs.rolling_robust_zscore(x, window, mode='safe')
        result_fast = hpcs.rolling_robust_zscore(x, window, mode='fast')
        result_det = hpcs.rolling_robust_zscore(x, window, mode='deterministic')

        # First two elements should be NaN
        assert np.isnan(result_safe[0])
        assert np.isnan(result_safe[1])
        assert np.isnan(result_fast[0])
        assert np.isnan(result_det[0])

        # result[2] should have a computed value (not NaN)
        assert not np.isnan(result_safe[2])
        assert not np.isnan(result_fast[2])
        assert not np.isnan(result_det[2])


class TestNaNHandling:
    """Test NaN detection differences between SAFE and FAST modes."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_median_nan_safe_mode(self):
        """SAFE mode should detect NaN and return NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.median(x, mode='safe')
        assert np.isnan(result), "SAFE mode should return NaN for NaN input"

    def test_mad_nan_safe_mode(self):
        """SAFE mode should detect NaN in MAD."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.mad(x, mode='safe')
        assert np.isnan(result), "SAFE mode should return NaN for NaN input"

    def test_quantile_nan_safe_mode(self):
        """SAFE mode should detect NaN in quantile."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.quantile(x, 0.5, mode='safe')
        assert np.isnan(result), "SAFE mode should return NaN for NaN input"

    def test_median_nan_deterministic_mode(self):
        """DETERMINISTIC mode should also detect NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.median(x, mode='deterministic')
        assert np.isnan(result), "DETERMINISTIC mode should return NaN for NaN input"

    def test_rolling_median_nan_safe_mode(self):
        """SAFE mode should detect NaN in rolling median."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        result = hpcs.rolling_median(x, 3, mode='safe')
        # Window containing NaN should produce NaN
        assert np.isnan(result[3])  # Window: {2, NaN, 4}


class TestModeOverride:
    """Test per-call mode override vs global mode."""

    def test_override_global_mode(self):
        """Per-call mode should override global mode."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Set global to FAST
        hpcs.set_execution_mode('fast')
        assert hpcs.get_execution_mode() == 'fast'

        # Call with explicit SAFE - should use SAFE, not FAST
        result = hpcs.median(x, mode='safe')
        assert np.isclose(result, 3.0)

        # Call without mode - should use global FAST
        result = hpcs.median(x)
        assert np.isclose(result, 3.0)

        # Reset to SAFE
        hpcs.set_execution_mode('safe')

    def test_rolling_override_global_mode(self):
        """Rolling functions should also support mode override."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        window = 3

        # Set global to DETERMINISTIC
        hpcs.set_execution_mode('deterministic')

        # Override with FAST
        result = hpcs.rolling_median(x, window, mode='fast')
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result[2:], expected[2:], rtol=1e-10)

        # Reset
        hpcs.set_execution_mode('safe')


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_single_element(self):
        """Test functions with single element."""
        x = np.array([42.0])

        assert np.isclose(hpcs.median(x), 42.0)
        assert np.isclose(hpcs.mad(x), 0.0)  # MAD of single element is 0
        assert np.isclose(hpcs.quantile(x, 0.5), 42.0)

    def test_two_elements(self):
        """Test functions with two elements."""
        x = np.array([2.0, 4.0])

        assert np.isclose(hpcs.median(x), 3.0)  # Average of two middle values
        assert np.isclose(hpcs.mad(x), 1.0)

    def test_large_array(self):
        """Test with large array to verify performance modes work."""
        x = np.random.randn(10000)

        result_safe = hpcs.median(x, mode='safe')
        result_fast = hpcs.median(x, mode='fast')
        result_det = hpcs.median(x, mode='deterministic')

        # All should give same result (within floating point tolerance)
        assert np.isclose(result_safe, result_fast, rtol=1e-10)
        assert np.isclose(result_safe, result_det, rtol=1e-10)

    def test_all_same_values(self):
        """Test with array of all same values."""
        x = np.full(100, 3.14)

        assert np.isclose(hpcs.median(x), 3.14)
        assert np.isclose(hpcs.mad(x), 0.0)  # No variation
        assert np.isclose(hpcs.quantile(x, 0.5), 3.14)


class TestDeterministicRepeatability:
    """Test that DETERMINISTIC mode gives bit-exact reproducibility."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_median_deterministic_repeatability(self):
        """DETERMINISTIC mode should give bit-exact results."""
        x = np.random.randn(10000)

        results = []
        for _ in range(5):
            result = hpcs.median(x, mode='deterministic')
            results.append(result)

        # All results should be EXACTLY equal (bit-exact)
        assert all(r == results[0] for r in results), "DETERMINISTIC mode should be bit-exact"

    def test_mad_deterministic_repeatability(self):
        """DETERMINISTIC mode should give bit-exact results for MAD."""
        x = np.random.randn(10000)

        results = []
        for _ in range(5):
            result = hpcs.mad(x, mode='deterministic')
            results.append(result)

        assert all(r == results[0] for r in results), "DETERMINISTIC mode should be bit-exact"

    def test_rolling_median_deterministic_repeatability(self):
        """DETERMINISTIC mode should give bit-exact results for rolling median."""
        x = np.random.randn(1000)

        results = []
        for _ in range(3):
            result = hpcs.rolling_median(x, 50, mode='deterministic')
            results.append(result)

        # All results should be exactly equal
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


if __name__ == '__main__':
    # Run with pytest
    pytest.main([__file__, '-v'])
