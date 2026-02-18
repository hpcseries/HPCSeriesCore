#!/usr/bin/env python3
"""
Quick test for Phase 4.1: Delegation functions with execution mode support
Tests: median_axis1, mad_axis1, median_masked, mad_masked
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/c/Users/Samkelo/OneDrive - iqlab/Clients/HPCSeriesCore')
import hpcs

def test_axis_median_modes():
    """Test axis_median with different execution modes"""
    print("\n=== Testing axis_median ===")
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Test without mode (uses global)
    result_default = hpcs.axis_median(x, axis=1)
    print(f"Default mode: {result_default}")

    # Test with explicit modes
    result_safe = hpcs.axis_median(x, axis=1, mode='safe')
    print(f"Safe mode:    {result_safe}")

    result_fast = hpcs.axis_median(x, axis=1, mode='fast')
    print(f"Fast mode:    {result_fast}")

    result_det = hpcs.axis_median(x, axis=1, mode='deterministic')
    print(f"Deterministic mode: {result_det}")

    # All should be similar
    assert np.allclose(result_safe, result_fast), "SAFE and FAST results differ!"
    assert np.allclose(result_safe, result_det), "SAFE and DETERMINISTIC results differ!"
    assert np.allclose(result_safe, [2.0, 5.0, 8.0]), "Results incorrect!"
    print("✓ axis_median: All modes produce correct results")

def test_axis_mad_modes():
    """Test axis_mad with different execution modes"""
    print("\n=== Testing axis_mad ===")
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    result_safe = hpcs.axis_mad(x, axis=1, mode='safe')
    print(f"Safe mode:    {result_safe}")

    result_fast = hpcs.axis_mad(x, axis=1, mode='fast')
    print(f"Fast mode:    {result_fast}")

    result_det = hpcs.axis_mad(x, axis=1, mode='deterministic')
    print(f"Deterministic mode: {result_det}")

    # All should be similar
    assert np.allclose(result_safe, result_fast), "SAFE and FAST results differ!"
    assert np.allclose(result_safe, result_det), "SAFE and DETERMINISTIC results differ!"
    assert np.allclose(result_safe, [1.0, 1.0, 1.0], atol=0.1), "Results incorrect!"
    print("✓ axis_mad: All modes produce correct results")

def test_median_masked_modes():
    """Test median_masked with different execution modes"""
    print("\n=== Testing median_masked ===")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mask = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.int32)  # Skip indices 2, 5, 8

    # Expected: median of [1, 2, 4, 5, 7, 8, 10] = 5.0
    result_safe = hpcs.median_masked(x, mask, mode='safe')
    print(f"Safe mode:    {result_safe}")

    result_fast = hpcs.median_masked(x, mask, mode='fast')
    print(f"Fast mode:    {result_fast}")

    result_det = hpcs.median_masked(x, mask, mode='deterministic')
    print(f"Deterministic mode: {result_det}")

    result_default = hpcs.median_masked(x, mask)
    print(f"Default mode: {result_default}")

    # All should be similar
    assert np.isclose(result_safe, result_fast), "SAFE and FAST results differ!"
    assert np.isclose(result_safe, result_det), "SAFE and DETERMINISTIC results differ!"
    assert np.isclose(result_safe, 5.0), f"Result incorrect! Expected 5.0, got {result_safe}"
    print("✓ median_masked: All modes produce correct results")

def test_mad_masked_modes():
    """Test mad_masked with different execution modes"""
    print("\n=== Testing mad_masked ===")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mask = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.int32)

    result_safe = hpcs.mad_masked(x, mask, mode='safe')
    print(f"Safe mode:    {result_safe}")

    result_fast = hpcs.mad_masked(x, mask, mode='fast')
    print(f"Fast mode:    {result_fast}")

    result_det = hpcs.mad_masked(x, mask, mode='deterministic')
    print(f"Deterministic mode: {result_det}")

    # All should be similar
    assert np.isclose(result_safe, result_fast), "SAFE and FAST results differ!"
    assert np.isclose(result_safe, result_det), "SAFE and DETERMINISTIC results differ!"
    print("✓ mad_masked: All modes produce correct results")

def test_nan_handling():
    """Test NaN handling in SAFE mode"""
    print("\n=== Testing NaN Handling ===")

    # Test axis_median with NaN
    x = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    result = hpcs.axis_median(x, axis=1, mode='safe')
    print(f"axis_median with NaN: {result}")
    assert np.isnan(result[0]), "Should return NaN for row with NaN"
    assert np.isclose(result[1], 5.0), "Second row should be valid"
    print("✓ NaN handling works correctly")

def test_all_masked():
    """Test behavior when all elements are masked"""
    print("\n=== Testing All-Masked Case ===")
    x = np.array([1.0, 2.0, 3.0])
    mask = np.array([0, 0, 0], dtype=np.int32)  # All masked

    # All-masked should raise RuntimeError with status 2 (HPCS_ERR_NUMERIC_FAIL)
    try:
        result = hpcs.median_masked(x, mask, mode='safe')
        print(f"All-masked result: {result}")
        # If we get here, the function returned without error (unexpected)
        assert False, "Should have raised an error for all-masked case"
    except RuntimeError as e:
        # Expected: error code 2
        assert "Error code 2" in str(e), f"Expected error code 2, got: {e}"
        print(f"✓ All-masked case correctly raises error: {e}")
    print("✓ All-masked case handled correctly")

def test_global_mode():
    """Test that mode=None uses global execution mode"""
    print("\n=== Testing Global Mode Setting ===")
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Set global mode to fast
    hpcs.set_execution_mode('fast')
    result_global = hpcs.axis_median(x, axis=1)
    result_explicit_fast = hpcs.axis_median(x, axis=1, mode='fast')

    print(f"Global mode result: {result_global}")
    print(f"Explicit fast result: {result_explicit_fast}")
    assert np.allclose(result_global, result_explicit_fast), "Global mode should use FAST"

    # Reset to safe
    hpcs.set_execution_mode('safe')
    print("✓ Global mode setting works correctly")

if __name__ == '__main__':
    print("="*60)
    print("Phase 4.1 Quick Test: Delegation Functions")
    print("Testing: median_axis1, mad_axis1, median_masked, mad_masked")
    print("="*60)

    try:
        test_axis_median_modes()
        test_axis_mad_modes()
        test_median_masked_modes()
        test_mad_masked_modes()
        test_nan_handling()
        test_all_masked()
        test_global_mode()

        print("\n" + "="*60)
        print("✅ ALL PHASE 4.1 TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
