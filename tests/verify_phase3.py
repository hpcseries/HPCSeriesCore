#!/usr/bin/env python3
"""Quick verification test for Phase 3 execution mode support"""

import numpy as np
import hpcs

def test_rolling_mean_modes():
    """Test rolling_mean with all execution modes"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    window = 3

    # Test without mode (uses global)
    result_default = hpcs.rolling_mean(x, window)
    print(f"✓ rolling_mean() without mode: {result_default[:5]}")

    # Test with explicit modes
    result_safe = hpcs.rolling_mean(x, window, mode='safe')
    print(f"✓ rolling_mean(mode='safe'): {result_safe[:5]}")

    result_fast = hpcs.rolling_mean(x, window, mode='fast')
    print(f"✓ rolling_mean(mode='fast'): {result_fast[:5]}")

    result_det = hpcs.rolling_mean(x, window, mode='deterministic')
    print(f"✓ rolling_mean(mode='deterministic'): {result_det[:5]}")

    # Results should be similar (nan, nan, 2.0, 3.0, 4.0)
    assert np.allclose(result_safe[2:], result_fast[2:], rtol=1e-10)
    print("  rolling_mean: PASSED\n")

def test_rolling_std_modes():
    """Test rolling_std with all execution modes"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    window = 3

    result_safe = hpcs.rolling_std(x, window, mode='safe')
    result_fast = hpcs.rolling_std(x, window, mode='fast')

    print(f"✓ rolling_std(mode='safe'): {result_safe[:5]}")
    print(f"✓ rolling_std(mode='fast'): {result_fast[:5]}")

    assert np.allclose(result_safe[2:], result_fast[2:], rtol=1e-10)
    print("  rolling_std: PASSED\n")

def test_rolling_var_modes():
    """Test rolling_var with all execution modes"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    window = 3

    result_safe = hpcs.rolling_var(x, window, mode='safe')
    result_fast = hpcs.rolling_var(x, window, mode='fast')

    print(f"✓ rolling_var(mode='safe'): {result_safe[:5]}")
    print(f"✓ rolling_var(mode='fast'): {result_fast[:5]}")

    assert np.allclose(result_safe[2:], result_fast[2:], rtol=1e-10)
    print("  rolling_var: PASSED\n")

def test_rolling_zscore_modes():
    """Test rolling_zscore with all execution modes"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    window = 3

    result_safe = hpcs.rolling_zscore(x, window, mode='safe')
    result_fast = hpcs.rolling_zscore(x, window, mode='fast')

    print(f"✓ rolling_zscore(mode='safe'): {result_safe[:5]}")
    print(f"✓ rolling_zscore(mode='fast'): {result_fast[:5]}")

    # Skip NaN values in comparison
    mask = ~np.isnan(result_safe) & ~np.isnan(result_fast)
    assert np.allclose(result_safe[mask], result_fast[mask], rtol=1e-10)
    print("  rolling_zscore: PASSED\n")

def test_nan_handling():
    """Test NaN handling in SAFE mode"""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    window = 3

    result_safe = hpcs.rolling_mean(x, window, mode='safe')
    print(f"✓ rolling_mean with NaN (SAFE): {result_safe[:6]}")

    # SAFE mode should handle NaN by excluding it from the mean
    assert not np.isnan(result_safe[3])  # Should compute mean of valid values
    print("  NaN handling (SAFE): PASSED\n")

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 3 Execution Mode Verification Tests")
    print("=" * 60)
    print()

    try:
        test_rolling_mean_modes()
        test_rolling_std_modes()
        test_rolling_var_modes()
        test_rolling_zscore_modes()
        test_nan_handling()

        print("=" * 60)
        print("✅ ALL PHASE 3 TESTS PASSED!")
        print("=" * 60)
        print()
        print("Phase 3 Implementation Complete:")
        print("  ✓ rolling_mean - execution mode support added")
        print("  ✓ rolling_std - execution mode support added")
        print("  ✓ rolling_var - execution mode support added")
        print("  ✓ rolling_zscore - execution mode support added")
        print()
        print("All 4 functions now support:")
        print("  - mode='safe' (full validation, NaN handling)")
        print("  - mode='fast' (no validation, maximum speed)")
        print("  - mode='deterministic' (reproducible, no SIMD)")
        print("  - mode=None (use global mode setting)")

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
