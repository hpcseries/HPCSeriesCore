"""
Test Suite for Phase 1 Reduction Functions with Execution Modes (v0.8.0)

Tests Safe vs Fast vs Deterministic execution modes for 9 core reductions:
  - sum, min, max, mean, variance, std
  - group_sum, group_mean, group_var

Validates:
  1. Mode API (set/get_execution_mode)
  2. Per-call mode override
  3. NaN detection (SAFE vs FAST)
  4. Correctness across all modes
  5. Grouped operations with mode support
  6. Invalid argument handling
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import hpcs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

import hpcs


class TestBasicReductionModes:
    """Test execution modes for basic reduction functions."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_sum_all_modes(self):
        """Test sum with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 15.0

        # Test SAFE mode
        result = hpcs.sum(x, mode='safe')
        assert np.isclose(result, expected), f"sum SAFE: expected {expected}, got {result}"

        # Test FAST mode
        result = hpcs.sum(x, mode='fast')
        assert np.isclose(result, expected), f"sum FAST: expected {expected}, got {result}"

        # Test DETERMINISTIC mode
        result = hpcs.sum(x, mode='deterministic')
        assert np.isclose(result, expected), f"sum DETERMINISTIC: expected {expected}, got {result}"

        # Test with global mode
        result = hpcs.sum(x)
        assert np.isclose(result, expected), f"sum with global mode: expected {expected}, got {result}"

    def test_min_all_modes(self):
        """Test min with all execution modes."""
        x = np.array([5.0, 3.0, 7.0, 1.0, 4.0])
        expected = 1.0

        assert np.isclose(hpcs.min(x, mode='safe'), expected)
        assert np.isclose(hpcs.min(x, mode='fast'), expected)
        assert np.isclose(hpcs.min(x, mode='deterministic'), expected)

    def test_max_all_modes(self):
        """Test max with all execution modes."""
        x = np.array([5.0, 3.0, 7.0, 1.0, 4.0])
        expected = 7.0

        assert np.isclose(hpcs.max(x, mode='safe'), expected)
        assert np.isclose(hpcs.max(x, mode='fast'), expected)
        assert np.isclose(hpcs.max(x, mode='deterministic'), expected)

    def test_mean_all_modes(self):
        """Test mean with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 3.0

        assert np.isclose(hpcs.mean(x, mode='safe'), expected)
        assert np.isclose(hpcs.mean(x, mode='fast'), expected)
        assert np.isclose(hpcs.mean(x, mode='deterministic'), expected)

    def test_var_all_modes(self):
        """Test variance with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 2.0  # Population variance

        assert np.isclose(hpcs.var(x, mode='safe'), expected)
        assert np.isclose(hpcs.var(x, mode='fast'), expected)
        assert np.isclose(hpcs.var(x, mode='deterministic'), expected)

    def test_std_all_modes(self):
        """Test standard deviation with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.sqrt(2.0)  # Population std

        assert np.isclose(hpcs.std(x, mode='safe'), expected)
        assert np.isclose(hpcs.std(x, mode='fast'), expected)
        assert np.isclose(hpcs.std(x, mode='deterministic'), expected)


class TestGroupedReductionModes:
    """Test execution modes for grouped reduction functions."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_group_sum_all_modes(self):
        """Test group_sum with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        group_ids = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
        n_groups = 2
        expected = np.array([8.0, 13.0])  # Group 0: 1+2+5=8, Group 1: 3+4+6=13

        result_safe = hpcs.group_sum(x, group_ids, n_groups, mode='safe')
        result_fast = hpcs.group_sum(x, group_ids, n_groups, mode='fast')
        result_det = hpcs.group_sum(x, group_ids, n_groups, mode='deterministic')

        np.testing.assert_allclose(result_safe, expected)
        np.testing.assert_allclose(result_fast, expected)
        np.testing.assert_allclose(result_det, expected)

    def test_group_mean_all_modes(self):
        """Test group_mean with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        group_ids = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
        n_groups = 2
        expected = np.array([8.0/3.0, 13.0/3.0])  # Group 0: mean of {1,2,5}, Group 1: mean of {3,4,6}

        result_safe = hpcs.group_mean(x, group_ids, n_groups, mode='safe')
        result_fast = hpcs.group_mean(x, group_ids, n_groups, mode='fast')
        result_det = hpcs.group_mean(x, group_ids, n_groups, mode='deterministic')

        np.testing.assert_allclose(result_safe, expected, rtol=1e-10)
        np.testing.assert_allclose(result_fast, expected, rtol=1e-10)
        np.testing.assert_allclose(result_det, expected, rtol=1e-10)

    def test_group_var_all_modes(self):
        """Test group_var with all execution modes."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        group_ids = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
        n_groups = 2
        # Group 0: {1, 2, 5}, mean=8/3, var = sum((xi-mean)^2)/n
        # Group 1: {3, 4, 6}, mean=13/3, var = sum((xi-mean)^2)/n
        expected = np.array([78.0/27.0, 14.0/9.0])  # Computed manually

        result_safe = hpcs.group_var(x, group_ids, n_groups, mode='safe')
        result_fast = hpcs.group_var(x, group_ids, n_groups, mode='fast')
        result_det = hpcs.group_var(x, group_ids, n_groups, mode='deterministic')

        np.testing.assert_allclose(result_safe, expected, rtol=1e-9)
        np.testing.assert_allclose(result_fast, expected, rtol=1e-9)
        np.testing.assert_allclose(result_det, expected, rtol=1e-9)


class TestNaNHandling:
    """Test NaN detection differences between SAFE and FAST modes."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_sum_nan_safe_mode(self):
        """SAFE mode should detect NaN and return NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.sum(x, mode='safe')
        assert np.isnan(result), "SAFE mode should return NaN for NaN input"

    def test_mean_nan_safe_mode(self):
        """SAFE mode should detect NaN in mean."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.mean(x, mode='safe')
        assert np.isnan(result), "SAFE mode should return NaN for NaN input"

    def test_var_nan_safe_mode(self):
        """SAFE mode should detect NaN in variance."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.var(x, mode='safe')
        assert np.isnan(result), "SAFE mode should return NaN for NaN input"

    def test_std_nan_safe_mode(self):
        """SAFE mode should detect NaN in std."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.std(x, mode='safe')
        assert np.isnan(result), "SAFE mode should return NaN for NaN input"

    def test_sum_nan_deterministic_mode(self):
        """DETERMINISTIC mode should also detect NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hpcs.sum(x, mode='deterministic')
        assert np.isnan(result), "DETERMINISTIC mode should return NaN for NaN input"


class TestModeOverride:
    """Test per-call mode override vs global mode."""

    def test_override_global_mode(self):
        """Per-call mode should override global mode."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Set global to FAST
        hpcs.set_execution_mode('fast')
        assert hpcs.get_execution_mode() == 'fast'

        # Call with explicit SAFE - should use SAFE, not FAST
        result = hpcs.sum(x, mode='safe')
        assert np.isclose(result, 15.0)

        # Call without mode - should use global FAST
        result = hpcs.sum(x)
        assert np.isclose(result, 15.0)

        # Reset to SAFE
        hpcs.set_execution_mode('safe')

    def test_grouped_override_global_mode(self):
        """Grouped functions should also support mode override."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        group_ids = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
        n_groups = 2

        # Set global to DETERMINISTIC
        hpcs.set_execution_mode('deterministic')

        # Override with FAST
        result = hpcs.group_sum(x, group_ids, n_groups, mode='fast')
        expected = np.array([8.0, 13.0])
        np.testing.assert_allclose(result, expected)

        # Reset
        hpcs.set_execution_mode('safe')


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_single_element(self):
        """Test reductions with single element."""
        x = np.array([42.0])

        assert np.isclose(hpcs.sum(x), 42.0)
        assert np.isclose(hpcs.mean(x), 42.0)
        assert np.isclose(hpcs.var(x), 0.0)  # Variance of single element is 0
        assert np.isclose(hpcs.std(x), 0.0)
        assert np.isclose(hpcs.min(x), 42.0)
        assert np.isclose(hpcs.max(x), 42.0)

    def test_two_elements(self):
        """Test reductions with two elements."""
        x = np.array([2.0, 4.0])

        assert np.isclose(hpcs.sum(x), 6.0)
        assert np.isclose(hpcs.mean(x), 3.0)
        assert np.isclose(hpcs.var(x), 1.0)  # Population variance
        assert np.isclose(hpcs.std(x), 1.0)

    def test_large_array(self):
        """Test with large array to verify performance modes work."""
        x = np.random.randn(10000)

        result_safe = hpcs.sum(x, mode='safe')
        result_fast = hpcs.sum(x, mode='fast')
        result_det = hpcs.sum(x, mode='deterministic')

        # All should give same result (within floating point tolerance)
        assert np.isclose(result_safe, result_fast, rtol=1e-10)
        assert np.isclose(result_safe, result_det, rtol=1e-10)

    def test_all_same_values(self):
        """Test with array of all same values."""
        x = np.full(100, 3.14)

        assert np.isclose(hpcs.sum(x), 314.0)
        assert np.isclose(hpcs.mean(x), 3.14)
        assert np.isclose(hpcs.var(x), 0.0)  # No variation
        assert np.isclose(hpcs.std(x), 0.0)
        assert np.isclose(hpcs.min(x), 3.14)
        assert np.isclose(hpcs.max(x), 3.14)


class TestDeterministicRepeatability:
    """Test that DETERMINISTIC mode gives bit-exact reproducibility."""

    def setup_method(self):
        """Reset to SAFE mode before each test."""
        hpcs.set_execution_mode('safe')

    def test_sum_deterministic_repeatability(self):
        """DETERMINISTIC mode should give bit-exact results."""
        x = np.random.randn(10000)

        results = []
        for _ in range(5):
            result = hpcs.sum(x, mode='deterministic')
            results.append(result)

        # All results should be EXACTLY equal (bit-exact)
        assert all(r == results[0] for r in results), "DETERMINISTIC mode should be bit-exact"

    def test_mean_deterministic_repeatability(self):
        """DETERMINISTIC mode should give bit-exact results for mean."""
        x = np.random.randn(10000)

        results = []
        for _ in range(5):
            result = hpcs.mean(x, mode='deterministic')
            results.append(result)

        assert all(r == results[0] for r in results), "DETERMINISTIC mode should be bit-exact"


if __name__ == '__main__':
    # Run with pytest
    pytest.main([__file__, '-v'])
