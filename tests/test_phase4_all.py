#!/usr/bin/env python3
"""
Comprehensive test for Phase 4: All 10 functions with execution mode support
Tests: 4 delegation + 3 masked + 2 axis-1 + 1 rolling
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/c/Users/Samkelo/OneDrive - iqlab/Clients/HPCSeriesCore')
import hpcs

def test_phase4_all_functions():
    """Test all 10 Phase 4 functions with all execution modes"""
    print("\n" + "="*70)
    print("PHASE 4 COMPREHENSIVE TEST: All 10 Functions")
    print("="*70)

    # Test data
    x_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    x_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    mask = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.int32)

    functions_tested = 0

    # ========== PHASE 4.1: DELEGATION FUNCTIONS (4 functions) ==========
    print("\n--- Phase 4.1: Delegation Functions ---")

    # 1. median_axis1
    print("\n1. median_axis1:")
    result_safe = hpcs.axis_median(x_2d, axis=1, mode='safe')
    result_fast = hpcs.axis_median(x_2d, axis=1, mode='fast')
    result_det = hpcs.axis_median(x_2d, axis=1, mode='deterministic')
    result_default = hpcs.axis_median(x_2d, axis=1)

    print(f"   SAFE:          {result_safe}")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")
    print(f"   DEFAULT:       {result_default}")

    assert np.allclose(result_safe, [2.0, 5.0, 8.0]), "median_axis1 SAFE incorrect"
    assert np.allclose(result_fast, result_safe), "median_axis1 FAST differs"
    assert np.allclose(result_det, result_safe), "median_axis1 DETERMINISTIC differs"
    print("   ✓ median_axis1 passed")
    functions_tested += 1

    # 2. mad_axis1
    print("\n2. mad_axis1:")
    result_safe = hpcs.axis_mad(x_2d, axis=1, mode='safe')
    result_fast = hpcs.axis_mad(x_2d, axis=1, mode='fast')
    result_det = hpcs.axis_mad(x_2d, axis=1, mode='deterministic')

    print(f"   SAFE:          {result_safe}")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert np.allclose(result_safe, result_fast, atol=0.1), "mad_axis1 FAST differs"
    assert np.allclose(result_det, result_safe, atol=0.1), "mad_axis1 DETERMINISTIC differs"
    print("   ✓ mad_axis1 passed")
    functions_tested += 1

    # 3. median_masked
    print("\n3. median_masked:")
    result_safe = hpcs.median_masked(x_1d, mask, mode='safe')
    result_fast = hpcs.median_masked(x_1d, mask, mode='fast')
    result_det = hpcs.median_masked(x_1d, mask, mode='deterministic')

    print(f"   SAFE:          {result_safe}")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert np.isclose(result_safe, 5.0), f"median_masked SAFE incorrect: {result_safe}"
    assert np.isclose(result_fast, result_safe), "median_masked FAST differs"
    assert np.isclose(result_det, result_safe), "median_masked DETERMINISTIC differs"
    print("   ✓ median_masked passed")
    functions_tested += 1

    # 4. mad_masked
    print("\n4. mad_masked:")
    result_safe = hpcs.mad_masked(x_1d, mask, mode='safe')
    result_fast = hpcs.mad_masked(x_1d, mask, mode='fast')
    result_det = hpcs.mad_masked(x_1d, mask, mode='deterministic')

    print(f"   SAFE:          {result_safe}")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert np.isclose(result_fast, result_safe, atol=0.5), "mad_masked FAST differs"
    assert np.isclose(result_det, result_safe, atol=0.5), "mad_masked DETERMINISTIC differs"
    print("   ✓ mad_masked passed")
    functions_tested += 1

    # ========== PHASE 4.2: MASKED REDUCTIONS (3 functions) ==========
    print("\n--- Phase 4.2: Masked Reductions ---")

    # 5. sum_masked
    print("\n5. sum_masked:")
    result_safe = hpcs.sum_masked(x_1d, mask, mode='safe')
    result_fast = hpcs.sum_masked(x_1d, mask, mode='fast')
    result_det = hpcs.sum_masked(x_1d, mask, mode='deterministic')

    expected_sum = 1 + 2 + 4 + 5 + 7 + 8 + 10  # Unmasked indices
    print(f"   SAFE:          {result_safe} (expected: {expected_sum})")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert np.isclose(result_safe, expected_sum), f"sum_masked SAFE incorrect"
    assert np.isclose(result_fast, expected_sum), "sum_masked FAST differs"
    assert np.isclose(result_det, expected_sum), "sum_masked DETERMINISTIC differs"
    print("   ✓ sum_masked passed")
    functions_tested += 1

    # 6. mean_masked
    print("\n6. mean_masked:")
    result_safe = hpcs.mean_masked(x_1d, mask, mode='safe')
    result_fast = hpcs.mean_masked(x_1d, mask, mode='fast')
    result_det = hpcs.mean_masked(x_1d, mask, mode='deterministic')

    expected_mean = expected_sum / 7  # 7 unmasked elements
    print(f"   SAFE:          {result_safe} (expected: {expected_mean:.4f})")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert np.isclose(result_safe, expected_mean), "mean_masked SAFE incorrect"
    assert np.isclose(result_fast, expected_mean), "mean_masked FAST differs"
    assert np.isclose(result_det, expected_mean), "mean_masked DETERMINISTIC differs"
    print("   ✓ mean_masked passed")
    functions_tested += 1

    # 7. var_masked (variance_masked)
    print("\n7. var_masked:")
    result_safe = hpcs.var_masked(x_1d, mask, mode='safe')
    result_fast = hpcs.var_masked(x_1d, mask, mode='fast')
    result_det = hpcs.var_masked(x_1d, mask, mode='deterministic')

    print(f"   SAFE:          {result_safe}")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert result_safe > 0, "var_masked SAFE should be positive"
    assert np.isclose(result_fast, result_safe, rtol=0.01), "var_masked FAST differs"
    assert np.isclose(result_det, result_safe, rtol=0.01), "var_masked DETERMINISTIC differs"
    print("   ✓ var_masked passed")
    functions_tested += 1

    # ========== PHASE 4.3: AXIS-1 REDUCTIONS (2 functions) ==========
    print("\n--- Phase 4.3: Axis-1 Reductions ---")

    # 8. axis_sum (reduce_sum_axis1)
    print("\n8. axis_sum:")
    result_safe = hpcs.axis_sum(x_2d, axis=1, mode='safe')
    result_fast = hpcs.axis_sum(x_2d, axis=1, mode='fast')
    result_det = hpcs.axis_sum(x_2d, axis=1, mode='deterministic')

    expected = np.array([6.0, 15.0, 24.0])  # Row sums
    print(f"   SAFE:          {result_safe} (expected: {expected})")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert np.allclose(result_safe, expected), "axis_sum SAFE incorrect"
    assert np.allclose(result_fast, expected), "axis_sum FAST differs"
    assert np.allclose(result_det, expected), "axis_sum DETERMINISTIC differs"
    print("   ✓ axis_sum passed")
    functions_tested += 1

    # 9. axis_mean (reduce_mean_axis1)
    print("\n9. axis_mean:")
    result_safe = hpcs.axis_mean(x_2d, axis=1, mode='safe')
    result_fast = hpcs.axis_mean(x_2d, axis=1, mode='fast')
    result_det = hpcs.axis_mean(x_2d, axis=1, mode='deterministic')

    expected = np.array([2.0, 5.0, 8.0])  # Row means
    print(f"   SAFE:          {result_safe} (expected: {expected})")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert np.allclose(result_safe, expected), "axis_mean SAFE incorrect"
    assert np.allclose(result_fast, expected), "axis_mean FAST differs"
    assert np.allclose(result_det, expected), "axis_mean DETERMINISTIC differs"
    print("   ✓ axis_mean passed")
    functions_tested += 1

    # ========== PHASE 4.4: ROLLING SUM (1 function) ==========
    print("\n--- Phase 4.4: Rolling Sum ---")

    # 10. rolling_sum
    print("\n10. rolling_sum:")
    x_roll = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3

    result_safe = hpcs.rolling_sum(x_roll, window, mode='safe')
    result_fast = hpcs.rolling_sum(x_roll, window, mode='fast')
    result_det = hpcs.rolling_sum(x_roll, window, mode='deterministic')

    expected = np.array([1.0, 3.0, 6.0, 9.0, 12.0])
    print(f"   SAFE:          {result_safe} (expected: {expected})")
    print(f"   FAST:          {result_fast}")
    print(f"   DETERMINISTIC: {result_det}")

    assert np.allclose(result_safe, expected), "rolling_sum SAFE incorrect"
    assert np.allclose(result_fast, expected), "rolling_sum FAST differs"
    assert np.allclose(result_det, expected), "rolling_sum DETERMINISTIC differs"
    print("   ✓ rolling_sum passed")
    functions_tested += 1

    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print(f"✅ ALL {functions_tested}/10 PHASE 4 FUNCTIONS PASSED!")
    print("="*70)
    print("\nBreakdown by category:")
    print("  • Phase 4.1 (Delegation):      4/4 functions ✓")
    print("  • Phase 4.2 (Masked):          3/3 functions ✓")
    print("  • Phase 4.3 (Axis-1):          2/2 functions ✓")
    print("  • Phase 4.4 (Rolling):         1/1 function  ✓")
    print("\nAll functions support 3 execution modes:")
    print("  - SAFE (full validation)")
    print("  - FAST (no validation, maximum speed)")
    print("  - DETERMINISTIC (reproducible results)")
    print("="*70)

if __name__ == '__main__':
    try:
        test_phase4_all_functions()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
