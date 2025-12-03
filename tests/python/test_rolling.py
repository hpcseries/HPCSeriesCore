"""
HPCSeries Core v0.7 - Python Rolling Operations Tests
======================================================

Tests for rolling window statistical operations.
"""

import pytest
import numpy as np
import hpcs


class TestRollingMean:
    """Test rolling mean operation."""

    def test_rolling_mean_simple(self):
        """Test rolling mean with simple data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        result = hpcs.rolling_mean(data, window)

        # First two values should be NaN or 0 (warmup)
        # Then: mean([1,2,3])=2, mean([2,3,4])=3, mean([3,4,5])=4
        assert result.shape == (5,)
        np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0], rtol=1e-10)

    def test_rolling_mean_window_size_one(self):
        """Test rolling mean with window=1 (should equal input)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.rolling_mean(data, 1)
        np.testing.assert_allclose(result, data, rtol=1e-10)

    def test_rolling_mean_large_array(self):
        """Test rolling mean with large array."""
        data = np.random.randn(10000)
        window = 50
        result = hpcs.rolling_mean(data, window)
        assert result.shape == data.shape
        assert not np.any(np.isnan(result[window-1:]))  # No NaN after warmup


class TestRollingStd:
    """Test rolling standard deviation."""

    def test_rolling_std_simple(self):
        """Test rolling std with simple data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        result = hpcs.rolling_std(data, window)
        assert result.shape == (5,)

        # std([1,2,3]) ≈ 0.816
        # std([2,3,4]) ≈ 0.816
        # std([3,4,5]) ≈ 0.816
        expected_std = np.std([1.0, 2.0, 3.0], ddof=0)
        np.testing.assert_allclose(result[2], expected_std, rtol=1e-6)

    def test_rolling_std_constant_input(self):
        """Test rolling std with constant input (should be 0)."""
        data = [5.0] * 10
        window = 3
        result = hpcs.rolling_std(data, window)
        # After warmup, std should be 0
        np.testing.assert_allclose(result[window-1:], 0.0, atol=1e-10)


class TestRollingVar:
    """Test rolling variance."""

    def test_rolling_var_simple(self):
        """Test rolling variance."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        result = hpcs.rolling_var(data, window)
        assert result.shape == (5,)

        # var([1,2,3]) = 2/3
        expected_var = np.var([1.0, 2.0, 3.0], ddof=0)
        np.testing.assert_allclose(result[2], expected_var, rtol=1e-6)

    def test_rolling_var_vs_std_squared(self):
        """Test that variance equals std squared."""
        data = np.random.randn(100)
        window = 10
        var_result = hpcs.rolling_var(data, window)
        std_result = hpcs.rolling_std(data, window)

        # After warmup, var should equal std^2
        np.testing.assert_allclose(
            var_result[window-1:],
            std_result[window-1:]**2,
            rtol=1e-6
        )


class TestRollingMedian:
    """Test rolling median (robust statistic)."""

    def test_rolling_median_simple(self):
        """Test rolling median with simple data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        result = hpcs.rolling_median(data, window)
        assert result.shape == (5,)

        # median([1,2,3])=2, median([2,3,4])=3, median([3,4,5])=4
        np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0], rtol=1e-10)

    def test_rolling_median_even_window(self):
        """Test rolling median with even window size."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        window = 4
        result = hpcs.rolling_median(data, window)

        # median([1,2,3,4])=2.5, median([2,3,4,5])=3.5
        np.testing.assert_allclose(result[3], 2.5, rtol=1e-10)
        np.testing.assert_allclose(result[4], 3.5, rtol=1e-10)

    def test_rolling_median_with_outliers(self):
        """Test rolling median is robust to outliers."""
        data = [1.0, 2.0, 100.0, 4.0, 5.0]  # 100 is outlier
        window = 3
        result = hpcs.rolling_median(data, window)

        # median([1,2,100])=2, median([2,100,4])=4, median([100,4,5])=5
        # Median should not be heavily affected by outlier
        assert result[2] == 2.0
        assert result[3] == 4.0
        assert result[4] == 5.0


class TestRollingMAD:
    """Test rolling MAD (Median Absolute Deviation)."""

    def test_rolling_mad_simple(self):
        """Test rolling MAD with simple data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        result = hpcs.rolling_mad(data, window)
        assert result.shape == (5,)

        # MAD([1,2,3]): median=2, |x-2|=[1,0,1], MAD=1
        np.testing.assert_allclose(result[2], 1.0, rtol=1e-10)

    def test_rolling_mad_constant(self):
        """Test rolling MAD with constant values (should be 0)."""
        data = [5.0] * 10
        window = 3
        result = hpcs.rolling_mad(data, window)
        # MAD of constant values should be 0
        np.testing.assert_allclose(result[window-1:], 0.0, atol=1e-10)


class TestRollingEdgeCases:
    """Test edge cases for rolling operations."""

    def test_window_equals_array_length(self):
        """Test when window size equals array length."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 5
        result = hpcs.rolling_mean(data, window)
        # Last value should be mean of all data
        assert result[-1] == 3.0

    def test_window_one(self):
        """Test window size of 1."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.rolling_mean(data, 1)
        np.testing.assert_allclose(result, data, rtol=1e-10)

    def test_large_window_small_data(self):
        """Test large window on small data."""
        data = [1.0, 2.0, 3.0]
        window = 10  # Window larger than data
        result = hpcs.rolling_mean(data, window)
        assert result.shape == (3,)


class TestRollingPerformance:
    """Test performance characteristics of rolling operations."""

    @pytest.mark.slow
    def test_rolling_mean_performance(self):
        """Test rolling mean performance on large array."""
        data = np.random.randn(1000000)
        window = 100
        result = hpcs.rolling_mean(data, window)
        assert result.shape == data.shape

    @pytest.mark.slow
    def test_rolling_median_performance(self):
        """Test rolling median performance (uses fast heap algorithm)."""
        data = np.random.randn(100000)
        window = 200
        result = hpcs.rolling_median(data, window)
        assert result.shape == data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
