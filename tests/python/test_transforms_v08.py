"""
HPCSeries Core v0.8.0 - Transform & Robust Statistics Python Tests
===================================================================

Comprehensive Python tests for all 9 v0.8.0 kernels with comparisons to:
  - NumPy (diff, cumulative operations)
  - SciPy (convolution, trimmed/winsorized statistics)
  - pandas (exponential weighted statistics)

Test Categories:
  1. Exponential Weighted Statistics (EWMA, EWVAR, EWSTD)
  2. Differencing & Cumulative Transforms
  3. FIR Convolution
  4. Robust Descriptive Statistics
"""

import pytest
import numpy as np
import hpcs

# Skip scipy tests if not available
try:
    from scipy import stats as scipy_stats
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Skip pandas tests if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================================
# Test Suite 1: Exponential Weighted Moving Average (EWMA)
# ============================================================================

class TestEWMA:
    """Test EWMA against pandas implementation."""

    def test_ewma_constant_array(self):
        """EWMA of constant array should equal the constant."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = hpcs.ewma(data, alpha=0.3)
        expected = np.full(5, 5.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ewma_alpha_1(self):
        """EWMA with alpha=1.0 should equal input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpcs.ewma(data, alpha=1.0)
        np.testing.assert_allclose(result, data, rtol=1e-10)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_ewma_vs_pandas(self):
        """Compare EWMA with pandas (span-based parameterization)."""
        data = np.random.randn(100)
        alpha = 0.3

        # HPCSeries implementation
        hpcs_result = hpcs.ewma(data, alpha=alpha)

        # pandas implementation (adjust=False matches our recurrence)
        pandas_result = pd.Series(data).ewm(alpha=alpha, adjust=False).mean().values

        np.testing.assert_allclose(hpcs_result, pandas_result, rtol=1e-10)

    def test_ewma_simple_values(self):
        """Test EWMA with hand-calculated values."""
        data = np.array([2.0, 4.0, 6.0, 8.0])
        alpha = 0.5
        result = hpcs.ewma(data, alpha=alpha)

        # Manual calculation:
        # y[0] = 2.0
        # y[1] = 0.5*4 + 0.5*2 = 3.0
        # y[2] = 0.5*6 + 0.5*3 = 4.5
        # y[3] = 0.5*8 + 0.5*4.5 = 6.25
        expected = np.array([2.0, 3.0, 4.5, 6.25])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ewma_large_array(self):
        """Test EWMA with large array."""
        data = np.random.randn(10000)
        result = hpcs.ewma(data, alpha=0.2)
        assert result.shape == data.shape
        assert not np.any(np.isnan(result[1:]))  # No NaN except possibly first


# ============================================================================
# Test Suite 2: Exponential Weighted Variance (EWVAR)
# ============================================================================

class TestEWVAR:
    """Test EWVAR against pandas implementation."""

    def test_ewvar_constant_array(self):
        """EWVAR of constant array should be zero."""
        data = np.array([3.0, 3.0, 3.0, 3.0])
        result = hpcs.ewvar(data, alpha=0.4)
        expected = np.zeros(4)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_ewvar_first_element_zero(self):
        """First element variance should be zero."""
        data = np.array([1.0, 5.0, 2.0, 8.0])
        result = hpcs.ewvar(data, alpha=0.5)
        assert result[0] == 0.0

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_ewvar_vs_pandas(self):
        """Compare EWVAR with pandas."""
        data = np.random.randn(100)
        alpha = 0.3

        # HPCSeries implementation
        hpcs_result = hpcs.ewvar(data, alpha=alpha)

        # pandas implementation (bias=True means no bias correction, matches our formula)
        pandas_result = pd.Series(data).ewm(alpha=alpha, adjust=False).var(bias=True).values

        # Skip first element (pandas returns 0, we return 0)
        np.testing.assert_allclose(hpcs_result, pandas_result, rtol=1e-10, atol=1e-10)


# ============================================================================
# Test Suite 3: Exponential Weighted Std Dev (EWSTD)
# ============================================================================

class TestEWSTD:
    """Test EWSTD (should equal sqrt of EWVAR)."""

    def test_ewstd_constant_array(self):
        """EWSTD of constant array should be zero."""
        data = np.array([7.0, 7.0, 7.0])
        result = hpcs.ewstd(data, alpha=0.5)
        expected = np.zeros(3)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_ewstd_equals_sqrt_ewvar(self):
        """EWSTD should equal sqrt(EWVAR)."""
        data = np.random.randn(50)
        alpha = 0.4

        ewvar = hpcs.ewvar(data, alpha=alpha)
        ewstd = hpcs.ewstd(data, alpha=alpha)

        np.testing.assert_allclose(ewstd, np.sqrt(ewvar), rtol=1e-10)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_ewstd_vs_pandas(self):
        """Compare EWSTD with pandas."""
        data = np.random.randn(100)
        alpha = 0.25

        hpcs_result = hpcs.ewstd(data, alpha=alpha)
        # pandas std() with bias=True (no bias correction, matches our formula)
        pandas_result = pd.Series(data).ewm(alpha=alpha, adjust=False).std(bias=True).values

        np.testing.assert_allclose(hpcs_result, pandas_result, rtol=1e-10, atol=1e-10)


# ============================================================================
# Test Suite 4: Finite Differencing (DIFF)
# ============================================================================

class TestDiff:
    """Test differencing against NumPy."""

    def test_diff_order1(self):
        """Test first difference."""
        data = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        result = hpcs.diff(data, order=1)

        # First element is NaN, rest are differences
        assert np.isnan(result[0])
        expected_diffs = np.array([2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result[1:], expected_diffs, rtol=1e-10)

    def test_diff_vs_numpy(self):
        """Compare first difference with NumPy."""
        data = np.random.randn(100)
        result = hpcs.diff(data, order=1)

        # NumPy diff produces n-1 elements; ours produces n with first NaN
        numpy_diff = np.diff(data)
        np.testing.assert_allclose(result[1:], numpy_diff, rtol=1e-10)

    def test_diff_order2(self):
        """Test second-order difference."""
        data = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
        result = hpcs.diff(data, order=2)

        # First 2 elements are NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # y[2] = 4 - 1 = 3
        # y[3] = 7 - 2 = 5
        # y[4] = 11 - 4 = 7
        expected = np.array([3.0, 5.0, 7.0])
        np.testing.assert_allclose(result[2:], expected, rtol=1e-10)

    def test_diff_large_array(self):
        """Test diff with large array."""
        data = np.cumsum(np.random.randn(10000))
        result = hpcs.diff(data, order=1)
        numpy_diff = np.diff(data)
        np.testing.assert_allclose(result[1:], numpy_diff, rtol=1e-10)


# ============================================================================
# Test Suite 5: Cumulative Minimum
# ============================================================================

class TestCumulativeMin:
    """Test cumulative minimum against NumPy."""

    def test_cumulative_min_simple(self):
        """Test with known values."""
        data = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        result = hpcs.cumulative_min(data)
        expected = np.array([5.0, 3.0, 3.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_cumulative_min_vs_numpy(self):
        """Compare with NumPy minimum.accumulate."""
        data = np.random.randn(100)
        result = hpcs.cumulative_min(data)
        expected = np.minimum.accumulate(data)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_cumulative_min_ascending(self):
        """Ascending array → first element repeated."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = hpcs.cumulative_min(data)
        expected = np.full(4, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ============================================================================
# Test Suite 6: Cumulative Maximum
# ============================================================================

class TestCumulativeMax:
    """Test cumulative maximum against NumPy."""

    def test_cumulative_max_simple(self):
        """Test with known values."""
        data = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        result = hpcs.cumulative_max(data)
        expected = np.array([1.0, 5.0, 5.0, 7.0, 7.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_cumulative_max_vs_numpy(self):
        """Compare with NumPy maximum.accumulate."""
        data = np.random.randn(100)
        result = hpcs.cumulative_max(data)
        expected = np.maximum.accumulate(data)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_cumulative_max_descending(self):
        """Descending array → first element repeated."""
        data = np.array([4.0, 3.0, 2.0, 1.0])
        result = hpcs.cumulative_max(data)
        expected = np.full(4, 4.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ============================================================================
# Test Suite 7: Convolution (Valid Mode)
# ============================================================================

class TestConvolve:
    """Test convolution against NumPy/SciPy."""

    def test_convolve_simple_smoothing(self):
        """Test with simple 3-tap smoothing kernel."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kernel = np.array([0.25, 0.5, 0.25])
        result = hpcs.convolve_valid(data, kernel)

        # Expected: [2.0, 3.0, 4.0] (output length = 5 - 3 + 1 = 3)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_convolve_vs_numpy(self):
        """Compare with NumPy convolve (mode='valid')."""
        data = np.random.randn(100)
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

        result = hpcs.convolve_valid(data, kernel)
        expected = np.convolve(data, kernel, mode='valid')

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_convolve_vs_scipy(self):
        """Compare with SciPy convolve."""
        data = np.random.randn(200)
        kernel = np.array([0.05, 0.15, 0.3, 0.3, 0.15, 0.05])

        result = hpcs.convolve_valid(data, kernel)
        expected = scipy_signal.convolve(data, kernel, mode='valid')

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_convolve_size7(self):
        """Test with size-7 kernel (tests template specialization)."""
        data = np.arange(1.0, 11.0)
        kernel = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])

        result = hpcs.convolve_valid(data, kernel)
        expected = np.convolve(data, kernel, mode='valid')

        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ============================================================================
# Test Suite 8: Trimmed Mean
# ============================================================================

class TestTrimmedMean:
    """Test trimmed mean against SciPy."""

    def test_trimmed_mean_no_trim(self):
        """Trimmed mean with trim_frac=0 should equal mean."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpcs.trimmed_mean(data, trim_frac=0.0)
        expected = np.mean(data)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_trimmed_mean_20percent(self):
        """Test 20% trimming."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # outlier
        result = hpcs.trimmed_mean(data, trim_frac=0.2)

        # Removes {1} and {100}, mean of {2, 3, 4} = 3.0
        expected = 3.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_trimmed_mean_vs_scipy(self):
        """Compare with SciPy trim_mean."""
        data = np.random.randn(101)
        trim_frac = 0.1

        result = hpcs.trimmed_mean(data, trim_frac=trim_frac)
        expected = scipy_stats.trim_mean(data, proportiontocut=trim_frac)

        np.testing.assert_allclose(result, expected, rtol=1e-9)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_trimmed_mean_large_array(self):
        """Test with large array."""
        np.random.seed(42)
        data = np.random.randn(10000)

        result = hpcs.trimmed_mean(data, trim_frac=0.05)
        expected = scipy_stats.trim_mean(data, proportiontocut=0.05)

        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ============================================================================
# Test Suite 9: Winsorized Mean
# ============================================================================

class TestWinsorizedMean:
    """Test winsorized mean against manual calculation."""

    def test_winsorized_mean_no_winsor(self):
        """Winsorized mean with win_frac=0 should equal mean."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hpcs.winsorized_mean(data, win_frac=0.0)
        expected = np.mean(data)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_winsorized_mean_10percent(self):
        """Test 10% winsorization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
        result = hpcs.winsorized_mean(data, win_frac=0.1)

        # Clamps {1} → 2, {100} → 9
        # Mean of {2,2,3,4,5,6,7,8,9,9} = 5.5
        expected = 5.5
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_winsorized_mean_symmetric_effect(self):
        """Winsorization should handle symmetric outliers."""
        data = np.array([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = hpcs.winsorized_mean(data, win_frac=0.15)

        # Should clamp both extremes
        assert abs(result - 3.0) < 0.5  # Approximate check


# ============================================================================
# Edge Cases & Robustness Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases across all functions."""

    def test_single_element_arrays(self):
        """Test all functions with n=1."""
        data = np.array([5.0])

        # EWMA, EWVAR, EWSTD should work
        assert hpcs.ewma(data, alpha=0.5)[0] == 5.0
        assert hpcs.ewvar(data, alpha=0.5)[0] == 0.0
        assert hpcs.ewstd(data, alpha=0.5)[0] == 0.0

        # Cumulative ops should work
        assert hpcs.cumulative_min(data)[0] == 5.0
        assert hpcs.cumulative_max(data)[0] == 5.0

    def test_two_element_arrays(self):
        """Test with n=2."""
        data = np.array([1.0, 2.0])

        diff_result = hpcs.diff(data, order=1)
        assert np.isnan(diff_result[0])
        assert diff_result[1] == 1.0

    def test_large_arrays_performance(self):
        """Smoke test with large arrays."""
        data = np.random.randn(100000)

        # Should complete without error
        _ = hpcs.ewma(data, alpha=0.1)
        _ = hpcs.cumulative_min(data)
        _ = hpcs.cumulative_max(data)
        _ = hpcs.diff(data, order=1)


# ============================================================================
# Benchmark Comparisons (Optional - Run with pytest --benchmark)
# ============================================================================

class TestPerformance:
    """Performance comparisons (informational only)."""

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_ewma_performance_vs_pandas(self, benchmark=None):
        """Compare EWMA performance with pandas (if pytest-benchmark available)."""
        data = np.random.randn(100000)
        alpha = 0.2

        if benchmark:
            result = benchmark(hpcs.ewma, data, alpha=alpha)
        else:
            # Manual timing for reference
            import time
            start = time.time()
            result = hpcs.ewma(data, alpha=alpha)
            hpcs_time = time.time() - start

            start = time.time()
            pandas_result = pd.Series(data).ewm(alpha=alpha, adjust=False).mean().values
            pandas_time = time.time() - start

            print(f"\n  HPCSeries: {hpcs_time*1000:.2f} ms")
            print(f"  pandas:    {pandas_time*1000:.2f} ms")
            print(f"  Speedup:   {pandas_time/hpcs_time:.2f}x")


# ============================================================================
# Pipeline API Tests - New Stages (v0.8.0)
# ============================================================================
# Tests for 12 new pipeline stages:
# - cumulative_min, cumulative_max: Running min/max
# - fill_forward: Forward fill (LOCF)
# - prefix_sum: Cumulative sum
# - convolve: FIR filter convolution
# - lag: Time-lagged values
# - log_return, pct_change: Financial returns
# - scale, shift: Linear transformations
# - abs, sqrt: Element-wise math operations


class TestPipelineCumulativeStages:
    """Test pipeline cumulative_min and cumulative_max stages."""

    def test_cumulative_min_basic(self):
        """Test running minimum."""
        pipe = hpcs.pipeline()
        pipe.cumulative_min()

        x = np.array([5.0, 3.0, 7.0, 2.0, 8.0])
        result = pipe.execute(x)

        expected = np.array([5.0, 3.0, 3.0, 2.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_cumulative_max_basic(self):
        """Test running maximum (high-water mark)."""
        pipe = hpcs.pipeline()
        pipe.cumulative_max()

        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = pipe.execute(x)

        expected = np.array([1.0, 3.0, 3.0, 5.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_cumulative_min_monotonic_increasing(self):
        """Cumulative min of increasing sequence is first element."""
        pipe = hpcs.pipeline()
        pipe.cumulative_min()

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pipe.execute(x)

        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_cumulative_max_monotonic_decreasing(self):
        """Cumulative max of decreasing sequence is first element."""
        pipe = hpcs.pipeline()
        pipe.cumulative_max()

        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = pipe.execute(x)

        expected = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        np.testing.assert_array_equal(result, expected)


class TestPipelineFillForward:
    """Test pipeline fill_forward stage."""

    def test_fill_forward_basic(self):
        """Test forward fill (LOCF)."""
        pipe = hpcs.pipeline()
        pipe.fill_forward()

        x = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        result = pipe.execute(x)

        expected = np.array([1.0, 1.0, 1.0, 4.0, 4.0])
        np.testing.assert_array_equal(result, expected)

    def test_fill_forward_no_nans(self):
        """Fill forward with no NaNs should pass through."""
        pipe = hpcs.pipeline()
        pipe.fill_forward()

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pipe.execute(x)

        np.testing.assert_array_equal(result, x)

    def test_fill_forward_leading_nan(self):
        """Fill forward with leading NaN keeps the NaN."""
        pipe = hpcs.pipeline()
        pipe.fill_forward()

        x = np.array([np.nan, np.nan, 3.0, np.nan, 5.0])
        result = pipe.execute(x)

        # Leading NaNs remain NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_array_equal(result[2:], [3.0, 3.0, 5.0])


class TestPipelinePrefixSum:
    """Test pipeline prefix_sum stage."""

    def test_prefix_sum_basic(self):
        """Test cumulative sum."""
        pipe = hpcs.pipeline()
        pipe.prefix_sum()

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pipe.execute(x)

        expected = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        np.testing.assert_array_equal(result, expected)

    def test_prefix_sum_matches_numpy(self):
        """Prefix sum should match np.cumsum."""
        pipe = hpcs.pipeline()
        pipe.prefix_sum()

        x = np.random.randn(100)
        result = pipe.execute(x)

        np.testing.assert_allclose(result, np.cumsum(x), rtol=1e-10)


class TestPipelineConvolve:
    """Test pipeline convolve stage."""

    def test_convolve_basic(self):
        """Test FIR filter convolution."""
        pipe = hpcs.pipeline()
        kernel = np.array([0.25, 0.5, 0.25])  # Simple smoothing filter
        pipe.convolve(kernel)

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pipe.execute(x)

        # Valid mode: output length = 5 - 3 + 1 = 3
        assert len(result) == 3

    def test_convolve_moving_average(self):
        """Test moving average via convolution."""
        pipe = hpcs.pipeline()
        kernel = np.ones(3) / 3.0  # 3-point moving average
        pipe.convolve(kernel)

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pipe.execute(x)

        # Expected: [2.0, 3.0, 4.0] (averages of [1,2,3], [2,3,4], [3,4,5])
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_convolve_output_length(self):
        """Test convolution output length is correct."""
        for m in [3, 5, 7, 9]:
            pipe = hpcs.pipeline()
            kernel = np.ones(m) / m
            pipe.convolve(kernel)

            x = np.random.randn(100)
            result = pipe.execute(x)

            expected_len = 100 - m + 1
            assert len(result) == expected_len, f"Kernel size {m}: expected {expected_len}, got {len(result)}"


class TestPipelineLag:
    """Test pipeline lag stage."""

    def test_lag_basic(self):
        """Test lag operator."""
        pipe = hpcs.pipeline()
        pipe.lag(k=2)

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pipe.execute(x)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_array_equal(result[2:], x[:3])

    def test_lag_k1(self):
        """Test lag with k=1."""
        pipe = hpcs.pipeline()
        pipe.lag(k=1)

        x = np.array([10.0, 20.0, 30.0, 40.0])
        result = pipe.execute(x)

        assert np.isnan(result[0])
        np.testing.assert_array_equal(result[1:], [10.0, 20.0, 30.0])


class TestPipelineFinancialReturns:
    """Test pipeline log_return and pct_change stages."""

    def test_log_return_basic(self):
        """Test log returns."""
        pipe = hpcs.pipeline()
        pipe.log_return()

        x = np.array([100.0, 105.0, 103.0])
        result = pipe.execute(x)

        assert np.isnan(result[0])
        np.testing.assert_almost_equal(result[1], np.log(105.0 / 100.0))
        np.testing.assert_almost_equal(result[2], np.log(103.0 / 105.0))

    def test_pct_change_basic(self):
        """Test percentage change."""
        pipe = hpcs.pipeline()
        pipe.pct_change()

        x = np.array([100.0, 110.0, 99.0])
        result = pipe.execute(x)

        assert np.isnan(result[0])
        np.testing.assert_almost_equal(result[1], 0.10)  # 10% increase
        np.testing.assert_almost_equal(result[2], -0.10)  # 10% decrease

    def test_log_return_and_pct_change_similar_for_small_changes(self):
        """Log return and pct_change should be similar for small changes."""
        pipe_log = hpcs.pipeline().log_return()
        pipe_pct = hpcs.pipeline().pct_change()

        # Small price changes (1% moves)
        x = np.array([100.0, 101.0, 100.5, 101.2, 100.8])

        log_ret = pipe_log.execute(x)
        pct_ret = pipe_pct.execute(x)

        # For small changes, log return ≈ pct change
        np.testing.assert_allclose(log_ret[1:], pct_ret[1:], rtol=0.02)


class TestPipelineLinearTransforms:
    """Test pipeline scale and shift stages."""

    def test_scale_basic(self):
        """Test scaling."""
        pipe = hpcs.pipeline()
        pipe.scale(factor=2.5)

        x = np.array([1.0, 2.0, 3.0])
        result = pipe.execute(x)

        np.testing.assert_array_equal(result, x * 2.5)

    def test_shift_basic(self):
        """Test shifting."""
        pipe = hpcs.pipeline()
        pipe.shift(offset=-10.0)

        x = np.array([15.0, 20.0, 25.0])
        result = pipe.execute(x)

        np.testing.assert_array_equal(result, x - 10.0)

    def test_scale_and_shift_composition(self):
        """Test scale followed by shift."""
        pipe = hpcs.pipeline()
        pipe.scale(factor=2.0)
        pipe.shift(offset=5.0)

        x = np.array([1.0, 2.0, 3.0])
        result = pipe.execute(x)

        # y = 2*x + 5
        expected = x * 2.0 + 5.0
        np.testing.assert_array_equal(result, expected)


class TestPipelineMathOperations:
    """Test pipeline abs and sqrt stages."""

    def test_abs_basic(self):
        """Test absolute value."""
        pipe = hpcs.pipeline()
        pipe.abs()

        x = np.array([-1.0, 2.0, -3.0, 4.0])
        result = pipe.execute(x)

        np.testing.assert_array_equal(result, np.abs(x))

    def test_sqrt_basic(self):
        """Test square root."""
        pipe = hpcs.pipeline()
        pipe.sqrt()

        x = np.array([1.0, 4.0, 9.0, 16.0])
        result = pipe.execute(x)

        np.testing.assert_array_equal(result, np.sqrt(x))


class TestPipelineDomainWorkflows:
    """Test domain-specific pipeline compositions."""

    def test_finance_volatility(self):
        """Finance: log returns -> ewvar -> sqrt = realized volatility."""
        pipe = hpcs.pipeline()
        pipe.log_return()
        pipe.ewvar(alpha=0.1)
        pipe.sqrt()

        prices = np.array([100.0, 102.0, 101.0, 103.0, 102.5, 104.0])
        vol = pipe.execute(prices)

        assert len(vol) == len(prices)
        assert vol[0] == 0.0  # ewvar[0] = 0 (initial variance), sqrt(0) = 0

    def test_finance_drawdown_components(self):
        """Finance: cumulative_max for high-water mark."""
        prices = np.array([100.0, 105.0, 102.0, 108.0, 104.0])

        pipe = hpcs.pipeline()
        pipe.cumulative_max()
        hwm = pipe.execute(prices)

        # High-water mark should be [100, 105, 105, 108, 108]
        expected_hwm = np.array([100.0, 105.0, 105.0, 108.0, 108.0])
        np.testing.assert_array_equal(hwm, expected_hwm)

        # Drawdown = price - high_water_mark
        drawdown = prices - hwm
        assert drawdown[0] == 0  # No drawdown at start
        assert drawdown[2] == 102 - 105  # -3

    def test_signal_lowpass_filter(self):
        """Engineering: convolve with moving average kernel."""
        # 5-point moving average as low-pass filter
        kernel = np.ones(5) / 5.0

        pipe = hpcs.pipeline()
        pipe.convolve(kernel)

        # Noisy signal
        np.random.seed(42)
        x = np.random.randn(100) + 10.0
        smoothed = pipe.execute(x)

        assert len(smoothed) == 96  # 100 - 5 + 1

    def test_data_quality_pipeline(self):
        """Data quality: fill_forward -> abs."""
        pipe = hpcs.pipeline()
        pipe.fill_forward()
        pipe.abs()

        # Data with gaps and negatives
        x = np.array([1.0, np.nan, -3.0, np.nan, np.nan, -6.0])
        result = pipe.execute(x)

        # After fill_forward: [1, 1, -3, -3, -3, -6]
        # After abs: [1, 1, 3, 3, 3, 6]
        expected = np.array([1.0, 1.0, 3.0, 3.0, 3.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_annualized_volatility(self):
        """Finance: log returns -> rolling_std -> scale(sqrt(252))."""
        pipe = hpcs.pipeline()
        pipe.log_return()
        pipe.rolling_std(window=20)
        pipe.scale(factor=np.sqrt(252))  # Annualize

        # Simulated daily prices
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        ann_vol = pipe.execute(prices)
        assert len(ann_vol) == len(prices)


class TestPipelineSummary:
    """Test that new stages appear correctly in summary."""

    def test_summary_includes_new_stages(self):
        """Test that pipeline summary includes new stage names."""
        pipe = hpcs.pipeline()
        pipe.cumulative_max()
        pipe.log_return()
        pipe.scale(factor=2.0)
        pipe.sqrt()

        summary = pipe.summary()

        assert 'cumulative_max' in summary
        assert 'log_return' in summary
        assert 'scale' in summary
        assert 'sqrt' in summary

    def test_summary_shows_params(self):
        """Test that summary shows stage parameters."""
        pipe = hpcs.pipeline()
        pipe.lag(k=5)
        pipe.convolve(np.ones(7) / 7.0)
        pipe.scale(factor=3.14)
        pipe.shift(offset=-100.5)

        summary = pipe.summary()

        assert 'k=5' in summary
        assert 'm=7' in summary
        assert '3.14' in summary
        assert '-100.5' in summary


class TestPipelineChaining:
    """Test that new stages support method chaining."""

    def test_chaining_new_stages(self):
        """Test that new stages return self for chaining."""
        pipe = (hpcs.pipeline()
                .fill_forward()
                .log_return()
                .abs()
                .rolling_mean(window=5)
                .scale(factor=100))

        summary = pipe.summary()
        assert 'fill_forward' in summary
        assert 'log_return' in summary
        assert 'abs' in summary
        assert 'rolling_mean' in summary
        assert 'scale' in summary
