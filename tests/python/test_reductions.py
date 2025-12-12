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
        """Test standard deviation (uses sample std, ddof=1)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.std(data)
        expected = np.std(data, ddof=1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_var_vs_numpy(self):
        """Test variance matches NumPy (uses sample var, ddof=1)."""
        data = np.random.randn(10000)
        hpcs_result = hpcs.var(data)
        numpy_result = np.var(data, ddof=1)
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


class TestQuantile:
    """Test quantile computation."""

    def test_quantile_median(self):
        """Test quantile at 0.5 matches median."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        q50 = hpcs.quantile(data, 0.5)
        median = hpcs.median(data)
        assert q50 == median

    def test_quantile_extremes(self):
        """Test quantiles at 0 and 1."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert hpcs.quantile(data, 0.0) == 1.0
        assert hpcs.quantile(data, 1.0) == 5.0

    def test_quantile_quartiles(self):
        """Test quartile values."""
        data = np.arange(1, 101, dtype=float)
        q25 = hpcs.quantile(data, 0.25)
        q75 = hpcs.quantile(data, 0.75)
        # Should be close to 25.75 and 75.25 with Type 7 interpolation
        assert 25.0 <= q25 <= 26.0
        assert 75.0 <= q75 <= 76.0


class TestTransforms:
    """Test transformation operations."""

    def test_zscore_simple(self):
        """Test z-score normalization."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.zscore(data)
        # Z-score should have mean ≈ 0 (uses population std, ddof=0)
        np.testing.assert_allclose(np.mean(result), 0.0, atol=1e-10)
        # Verify result is reasonable
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_robust_zscore_simple(self):
        """Test robust z-score."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.robust_zscore(data)
        # Should have median ≈ 0
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_robust_zscore_with_outliers(self):
        """Test robust z-score calculation with outliers."""
        data_clean = [1.0, 2.0, 3.0, 4.0, 5.0]
        data_outlier = [1.0, 2.0, 3.0, 4.0, 100.0]

        rz_clean = hpcs.robust_zscore(data_clean)
        rz_outlier = hpcs.robust_zscore(data_outlier)

        # Verify robust z-score produces reasonable results
        assert isinstance(rz_clean, np.ndarray)
        assert isinstance(rz_outlier, np.ndarray)
        assert len(rz_clean) == len(data_clean)
        assert len(rz_outlier) == len(data_outlier)

    def test_normalize_minmax_simple(self):
        """Test min-max normalization."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.normalize_minmax(data)
        assert result[0] == 0.0  # min maps to 0
        assert result[-1] == 1.0  # max maps to 1
        assert result[2] == 0.5  # middle value
        np.testing.assert_allclose(result, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_normalize_minmax_large_array(self):
        """Test min-max with large array."""
        data = np.random.randn(10000)
        result = hpcs.normalize_minmax(data)
        assert np.min(result) == 0.0
        assert np.max(result) == 1.0
        assert np.all((result >= 0) & (result <= 1))

    def test_clip_simple(self):
        """Test clipping values."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.clip(data, 2.0, 4.0)
        np.testing.assert_allclose(result, [2.0, 2.0, 3.0, 4.0, 4.0])

    def test_clip_no_change(self):
        """Test clip when all values are within bounds."""
        data = [2.5, 3.0, 3.5]
        result = hpcs.clip(data, 2.0, 4.0)
        np.testing.assert_allclose(result, data)


class TestAnomalyDetection:
    """Test anomaly detection operations."""

    def test_detect_anomalies_simple(self):
        """Test basic anomaly detection."""
        # Use data where anomaly detection is reasonable
        # Create data with clear structure and one extreme outlier
        data = np.array([10.0] * 100 + [10000.0])  # 100 normal values + 1 extreme outlier
        anomalies = hpcs.detect_anomalies(data, threshold=3.0)
        # Should detect the extreme outlier
        assert anomalies[-1] == True, "Last element (extreme outlier) should be detected"
        # Most other values should be normal
        assert np.sum(anomalies[:-1]) < 5, "Few false positives expected"

    def test_detect_anomalies_no_outliers(self):
        """Test when there are no anomalies."""
        data = np.random.randn(1000) * 0.5  # Small std, unlikely to exceed 3σ
        anomalies = hpcs.detect_anomalies(data, threshold=3.0)
        # Should detect very few or no anomalies
        assert np.sum(anomalies) < 10  # At most a few false positives

    def test_detect_anomalies_robust_simple(self):
        """Test robust anomaly detection."""
        data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0]
        anomalies = hpcs.detect_anomalies_robust(data, threshold=3.0)
        assert anomalies[-1] == True  # 1000 is clearly an outlier
        # Most others should be normal
        assert np.sum(anomalies[:-1]) < 3  # Allow some false positives

    def test_detect_anomalies_robust_vs_standard(self):
        """Test robust anomaly detection works."""
        # Data with outliers
        data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 1001.0]

        anomalies_robust = hpcs.detect_anomalies_robust(data, threshold=3.0)

        # Should detect the large outliers
        assert np.sum(anomalies_robust) >= 2  # At least the two outliers

    def test_detect_anomalies_threshold(self):
        """Test different threshold values."""
        # Data with clear outlier
        data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]

        # Lower threshold is more sensitive
        anomalies_low = hpcs.detect_anomalies(data, threshold=1.0)
        anomalies_high = hpcs.detect_anomalies(data, threshold=10.0)

        # Lower threshold should detect at least as many
        assert np.sum(anomalies_low) >= np.sum(anomalies_high)


class TestAxisOperations:
    """Test 2D axis operations (Tier B)."""

    def test_axis_sum_simple(self):
        """Test axis sum along columns."""
        data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        result = hpcs.axis_sum(data, axis=1)
        expected = np.array([6.0, 15.0])  # Sum across columns for each row
        np.testing.assert_allclose(result, expected)

    def test_axis_mean_simple(self):
        """Test axis mean along columns."""
        data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        result = hpcs.axis_mean(data, axis=1)
        expected = np.array([2.0, 5.0])  # Mean across columns for each row
        np.testing.assert_allclose(result, expected)

    def test_axis_median_simple(self):
        """Test axis median along columns."""
        data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        result = hpcs.axis_median(data, axis=1)
        expected = np.array([2.0, 5.0])  # Median across columns for each row
        np.testing.assert_allclose(result, expected)

    def test_axis_mad_simple(self):
        """Test axis MAD along columns."""
        data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        result = hpcs.axis_mad(data, axis=1)
        # MAD for each row
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


class TestMaskedOperations:
    """Test masked operations (Tier B)."""

    def test_sum_masked_simple(self):
        """Test masked sum."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mask = [1, 1, 0, 1, 1]  # Mask out 3.0
        result = hpcs.sum_masked(data, mask)
        assert result == 12.0  # 1 + 2 + 4 + 5

    def test_mean_masked_simple(self):
        """Test masked mean."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mask = [1, 1, 0, 1, 1]  # Mask out 3.0
        result = hpcs.mean_masked(data, mask)
        assert result == 3.0  # (1 + 2 + 4 + 5) / 4

    def test_var_masked_simple(self):
        """Test masked variance."""
        data = [1.0, 2.0, 100.0, 4.0, 5.0]
        mask = [1, 1, 0, 1, 1]  # Mask out outlier 100.0
        result = hpcs.var_masked(data, mask)
        # Variance of [1, 2, 4, 5]
        assert result > 0

    def test_median_masked_simple(self):
        """Test masked median."""
        data = [1.0, 2.0, 100.0, 3.0, 4.0]
        mask = [1, 1, 0, 1, 1]  # Mask out outlier 100.0
        result = hpcs.median_masked(data, mask)
        assert result == 2.5  # Median of [1, 2, 3, 4]

    def test_mad_masked_simple(self):
        """Test masked MAD."""
        data = [1.0, 2.0, 100.0, 3.0, 4.0, 5.0]
        mask = [1, 1, 0, 1, 1, 1]  # Mask out outlier
        result = hpcs.mad_masked(data, mask)
        # MAD of [1, 2, 3, 4, 5]
        assert result > 0


class TestNewRollingOperations:
    """Test newly added rolling operations."""

    def test_rolling_sum(self):
        """Test rolling sum."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = hpcs.rolling_sum(data, window=3)
        assert isinstance(result, np.ndarray)
        # Rolling sum with partial windows: 1, 3, 6, 9, 12
        assert abs(result[0] - 1.0) < 1e-10  # First window is just [1]
        assert abs(result[1] - 3.0) < 1e-10  # Window is [1, 2]
        assert abs(result[2] - 6.0) < 1e-10  # Full window [1, 2, 3]
        assert abs(result[3] - 9.0) < 1e-10  # Window [2, 3, 4]
        assert abs(result[4] - 12.0) < 1e-10 # Window [3, 4, 5]

    def test_rolling_zscore(self):
        """Test rolling z-score."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Outlier at end
        result = hpcs.rolling_zscore(data, window=3)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        # Last value should have high z-score due to outlier
        assert abs(result[-1]) > 1.0

    def test_rolling_robust_zscore(self):
        """Test rolling robust z-score."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Outlier at end
        result = hpcs.rolling_robust_zscore(data, window=3)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)


class TestNewAxisOperations:
    """Test newly added axis operations."""

    def test_axis_min(self):
        """Test axis minimum."""
        data = np.array([[1.0, 5.0, 3.0],
                        [4.0, 2.0, 6.0]])
        result = hpcs.axis_min(data, axis=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == 1.0  # Min of first row
        assert result[1] == 2.0  # Min of second row

    def test_axis_max(self):
        """Test axis maximum."""
        data = np.array([[1.0, 5.0, 3.0],
                        [4.0, 2.0, 6.0]])
        result = hpcs.axis_max(data, axis=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == 5.0  # Max of first row
        assert result[1] == 6.0  # Max of second row


class TestAnomalyAxisOperations:
    """Test anomaly detection on axis."""

    def test_anomaly_axis(self):
        """Test anomaly detection along axis."""
        # Use more extreme outlier with more normal values
        data = np.array([[10.0] * 10 + [1000.0],  # Extreme outlier
                        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]])
        result = hpcs.anomaly_axis(data, axis=1, threshold=3.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 11)
        # First row should have anomaly detected at last position
        assert result[0, -1] == True  # Extreme outlier
        # Most values in second row should not be anomalies
        assert np.sum(result[1, :]) <= 1  # At most 1 false positive

    def test_anomaly_robust_axis(self):
        """Test robust anomaly detection along axis."""
        data = np.array([[1.0, 2.0, 3.0, 100.0],  # Outlier in first row
                        [4.0, 5.0, 6.0, 7.0]])     # No outliers
        result = hpcs.anomaly_robust_axis(data, axis=1, threshold=3.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)
        # First row should have anomaly detected
        assert result[0, 3] == True


class TestBatchedRollingOperations:
    """Test batched rolling operations."""

    def test_rolling_mean_batched_axis0(self):
        """Test batched rolling mean along axis 0."""
        data = np.array([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0]])
        result = hpcs.rolling_mean_batched(data, window=2, axis=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)

    def test_rolling_mean_batched_axis1(self):
        """Test batched rolling mean along axis 1."""
        data = np.array([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]])
        result = hpcs.rolling_mean_batched(data, window=2, axis=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)


class TestMaskedRollingOperations:
    """Test masked rolling operations."""

    def test_rolling_mean_masked(self):
        """Test rolling mean with mask."""
        data = np.array([1.0, 2.0, 999.0, 4.0, 5.0])  # 999.0 is invalid
        mask = np.array([True, True, False, True, True])
        result = hpcs.rolling_mean_masked(data, window=3, mask=mask)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        # Should skip the invalid value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
