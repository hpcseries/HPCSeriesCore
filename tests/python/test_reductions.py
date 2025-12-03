"""
HPCSeries Core v0.7 - Python Reduction Tests
=============================================

Tests for SIMD-accelerated reduction operations.
"""

import pytest
import numpy as np
import hpcs


class TestBasicReductions:
    """Test basic reduction operations (sum, mean, std, min, max)."""

    def test_sum_simple(self):
        """Test sum with simple integer data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.sum(data)
        assert result == 15.0

    def test_sum_numpy_array(self):
        """Test sum with NumPy array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpcs.sum(data)
        assert result == 15.0

    def test_sum_large_array(self):
        """Test sum with large array."""
        data = np.ones(1000000)
        result = hpcs.sum(data)
        assert result == 1000000.0

    def test_mean_simple(self):
        """Test mean calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.mean(data)
        assert result == 3.0

    def test_mean_vs_numpy(self):
        """Test mean matches NumPy."""
        data = np.random.randn(10000)
        hpcs_result = hpcs.mean(data)
        numpy_result = np.mean(data)
        np.testing.assert_allclose(hpcs_result, numpy_result, rtol=1e-10)

    def test_std_simple(self):
        """Test standard deviation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.std(data)
        expected = np.std(data, ddof=0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_var_vs_numpy(self):
        """Test variance matches NumPy."""
        data = np.random.randn(10000)
        hpcs_result = hpcs.var(data)
        numpy_result = np.var(data, ddof=0)
        np.testing.assert_allclose(hpcs_result, numpy_result, rtol=1e-10)

    def test_min_simple(self):
        """Test minimum value."""
        data = [5.0, 2.0, 8.0, 1.0, 9.0]
        result = hpcs.min(data)
        assert result == 1.0

    def test_max_simple(self):
        """Test maximum value."""
        data = [5.0, 2.0, 8.0, 1.0, 9.0]
        result = hpcs.max(data)
        assert result == 9.0

    def test_min_max_with_negatives(self):
        """Test min/max with negative values."""
        data = [-5.0, 2.0, -8.0, 1.0, 9.0]
        assert hpcs.min(data) == -8.0
        assert hpcs.max(data) == 9.0


class TestRobustStatistics:
    """Test robust statistical operations (median, MAD)."""

    def test_median_odd_length(self):
        """Test median with odd-length array."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.median(data)
        assert result == 3.0

    def test_median_even_length(self):
        """Test median with even-length array."""
        data = [1.0, 2.0, 3.0, 4.0]
        result = hpcs.median(data)
        assert result == 2.5  # Average of 2 and 3

    def test_median_vs_numpy(self):
        """Test median matches NumPy."""
        data = np.random.randn(10001)
        hpcs_result = hpcs.median(data)
        numpy_result = np.median(data)
        np.testing.assert_allclose(hpcs_result, numpy_result, rtol=1e-10)

    def test_mad_simple(self):
        """Test MAD (Median Absolute Deviation)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.mad(data)
        # MAD = median(|x - median(x)|)
        # median = 3, deviations = [2, 1, 0, 1, 2], MAD = 1
        assert result == 1.0

    def test_mad_with_outliers(self):
        """Test MAD is robust to outliers."""
        data_clean = [1.0, 2.0, 3.0, 4.0, 5.0]
        data_outlier = [1.0, 2.0, 3.0, 4.0, 100.0]

        mad_clean = hpcs.mad(data_clean)
        mad_outlier = hpcs.mad(data_outlier)

        # MAD should be more resistant to outliers than std
        std_clean = hpcs.std(data_clean)
        std_outlier = hpcs.std(data_outlier)

        # MAD ratio should be smaller than std ratio
        assert (mad_outlier / mad_clean) < (std_outlier / std_clean)


class TestTypeConversions:
    """Test automatic type conversions."""

    def test_list_conversion(self):
        """Test Python list is converted correctly."""
        data = [1, 2, 3, 4, 5]  # integers
        result = hpcs.sum(data)
        assert result == 15.0

    def test_float32_conversion(self):
        """Test float32 arrays are converted."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = hpcs.sum(data)
        assert result == 15.0

    def test_int_array_conversion(self):
        """Test integer arrays are converted."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = hpcs.sum(data)
        assert result == 15.0

    def test_non_contiguous_array(self):
        """Test non-contiguous arrays are handled."""
        data = np.array([[1, 2], [3, 4], [5, 6]])[:, 0]  # Non-contiguous
        assert not data.flags.c_contiguous
        result = hpcs.sum(data)
        assert result == 9.0  # 1 + 3 + 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element(self):
        """Test with single element array."""
        data = [42.0]
        assert hpcs.sum(data) == 42.0
        assert hpcs.mean(data) == 42.0
        assert hpcs.median(data) == 42.0
        assert hpcs.min(data) == 42.0
        assert hpcs.max(data) == 42.0

    def test_two_elements(self):
        """Test with two element array."""
        data = [1.0, 2.0]
        assert hpcs.sum(data) == 3.0
        assert hpcs.mean(data) == 1.5
        assert hpcs.median(data) == 1.5

    def test_all_same_values(self):
        """Test with all same values."""
        data = [5.0] * 100
        assert hpcs.sum(data) == 500.0
        assert hpcs.mean(data) == 5.0
        assert hpcs.std(data) == 0.0
        assert hpcs.var(data) == 0.0
        assert hpcs.median(data) == 5.0
        assert hpcs.mad(data) == 0.0

    def test_zeros(self):
        """Test with all zeros."""
        data = np.zeros(1000)
        assert hpcs.sum(data) == 0.0
        assert hpcs.mean(data) == 0.0
        assert hpcs.std(data) == 0.0

    def test_large_values(self):
        """Test with large values."""
        data = [1e15, 2e15, 3e15]
        result = hpcs.sum(data)
        expected = 6e15
        np.testing.assert_allclose(result, expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
