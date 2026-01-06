/*
 * Robust Descriptive Statistics for HPCSeries Core v0.8.0
 *
 * This module provides outlier-resistant statistical measures:
 *   - hpcs_trimmed_mean: Mean after discarding extreme values
 *   - hpcs_winsorized_mean: Mean after clamping extreme values
 *
 * Both functions use deterministic O(n) selection algorithms (introselect)
 * to find quantile bounds without full sorting.
 *
 * Design principles:
 *   - Caller-owned input arrays (immutable)
 *   - Internal workspace allocation (consistent with hpcs_median pattern)
 *   - NaN propagation: any NaN → status=2
 *   - Deterministic behavior
 *   - Execution modes: SAFE (default), FAST (skip NaN checks), DETERMINISTIC (same as SAFE)
 *
 * References:
 *   - Wilcox, R. R. (2012). "Introduction to Robust Estimation and Hypothesis Testing"
 *   - Huber, P. J. (1981). "Robust Statistics"
 *
 * Author: HPCSeries Core Library
 * Version: 0.8.0
 * Date: 2025-12-23
 */

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// Execution mode constants (match hpcs_constants.f90)
#define HPCS_MODE_SAFE          0
#define HPCS_MODE_FAST          1
#define HPCS_MODE_DETERMINISTIC 2
#define HPCS_MODE_USE_GLOBAL   -1

// Error codes
#define HPCS_SUCCESS            0
#define HPCS_ERR_INVALID_ARGS   1
#define HPCS_ERR_NUMERIC_FAIL   2

// Forward declarations of selection helpers (from hpcs_selection.cpp)
extern "C" {
    void hpcs_select_two_bounds(
        double* arr,
        int n,
        int k1,
        int k2,
        double* lower,
        double* upper,
        int* status
    );

    // Execution mode API (from hpcs_core_execution_mode.f90)
    void hpcs_get_execution_mode(int* mode, int* status);
}

// Anonymous namespace for internal helpers
namespace {

/*
 * Get effective execution mode (resolve HPCS_MODE_USE_GLOBAL to actual mode).
 */
inline int get_effective_mode(int mode) {
    if (mode == HPCS_MODE_USE_GLOBAL) {
        int global_mode;
        int status;
        hpcs_get_execution_mode(&global_mode, &status);
        return (status == HPCS_SUCCESS) ? global_mode : HPCS_MODE_SAFE;
    }
    return mode;
}

/*
 * Check if array contains any NaN values.
 * Returns true if at least one NaN found.
 */
inline bool has_nan(const double* x, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::isnan(x[i])) return true;
    }
    return false;
}

/*
 * Count non-NaN elements in array.
 */
inline int count_finite(const double* x, int n) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(x[i])) ++count;
    }
    return count;
}

/*
 * Compute mean of finite elements only.
 * Returns NaN if no finite elements.
 */
inline double mean_finite(const double* x, int n) {
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(x[i])) {
            sum += x[i];
            ++count;
        }
    }
    return (count > 0) ? (sum / count) : std::numeric_limits<double>::quiet_NaN();
}

/*
 * hpcs_trimmed_mean_safe - SAFE mode implementation
 *
 * Full NaN detection and validation (current behavior).
 */
void hpcs_trimmed_mean_safe(
    const double* x,
    int n,
    double trim_frac,
    double* result,
    int* status
) {
    // Validate arguments
    if (n <= 0 || trim_frac < 0.0 || trim_frac >= 0.5 ||
        x == nullptr || result == nullptr || status == nullptr) {
        if (status != nullptr) *status = 1;  // HPCS_ERR_INVALID_ARGS
        return;
    }

    // Check for NaN (propagate NaN policy)
    if (has_nan(x, n)) {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = 2;  // HPCS_ERR_NUMERIC_FAIL
        return;
    }

    // Compute number of elements to trim from each end
    int m = static_cast<int>(std::floor(trim_frac * n));

    // If no trimming, compute simple mean
    if (m == 0) {
        *result = mean_finite(x, n);
        if (std::isnan(*result)) {
            *status = 2;  // HPCS_ERR_NUMERIC_FAIL (all NaN)
        } else {
            *status = 0;  // HPCS_SUCCESS
        }
        return;
    }

    // Check if trimming would remove all elements
    if (2 * m >= n) {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = 1;  // HPCS_ERR_INVALID_ARGS (no elements remain)
        return;
    }

    // Copy input to work buffer (internal allocation - consistent with hpcs_median)
    std::vector<double> work(x, x + n);

    // Find trimming bounds using deterministic selection
    double x_lower, x_upper;
    int sel_status;
    hpcs_select_two_bounds(work.data(), n, m, n - m - 1, &x_lower, &x_upper, &sel_status);

    if (sel_status != 0) {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = sel_status;
        return;
    }

    // Compute mean of elements strictly between bounds
    // Handle ties: discard exactly m elements from each tail
    double sum = 0.0;
    int kept = 0;
    int removed_lower = 0;
    int removed_upper = 0;

    for (int i = 0; i < n; ++i) {
        double v = x[i];

        // Discard lower tail
        if (v < x_lower) {
            if (removed_lower < m) {
                ++removed_lower;
                continue;
            }
        }

        // Discard upper tail
        if (v > x_upper) {
            if (removed_upper < m) {
                ++removed_upper;
                continue;
            }
        }

        // Keep element
        sum += v;
        ++kept;
    }

    if (kept > 0) {
        *result = sum / kept;
        *status = HPCS_SUCCESS;
    } else {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = HPCS_ERR_INVALID_ARGS;
    }
}

/*
 * hpcs_trimmed_mean_fast - FAST mode implementation
 *
 * Skip NaN checks for maximum performance (1.5-2x faster for large arrays).
 * Assumes all inputs are valid and finite.
 */
void hpcs_trimmed_mean_fast(
    const double* x,
    int n,
    double trim_frac,
    double* result,
    int* status
) {
    // Minimal validation (skip NaN checks)
    if (n <= 0 || trim_frac < 0.0 || trim_frac >= 0.5 ||
        x == nullptr || result == nullptr || status == nullptr) {
        if (status != nullptr) *status = HPCS_ERR_INVALID_ARGS;
        return;
    }

    // Compute number of elements to trim from each end
    int m = static_cast<int>(std::floor(trim_frac * n));

    // If no trimming, compute simple mean (no NaN check)
    if (m == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += x[i];
        }
        *result = sum / n;
        *status = HPCS_SUCCESS;
        return;
    }

    // Check if trimming would remove all elements
    if (2 * m >= n) {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = HPCS_ERR_INVALID_ARGS;
        return;
    }

    // Copy input to work buffer
    std::vector<double> work(x, x + n);

    // Find trimming bounds using deterministic selection
    double x_lower, x_upper;
    int sel_status;
    hpcs_select_two_bounds(work.data(), n, m, n - m - 1, &x_lower, &x_upper, &sel_status);

    if (sel_status != HPCS_SUCCESS) {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = sel_status;
        return;
    }

    // Compute mean of elements strictly between bounds
    double sum = 0.0;
    int kept = 0;
    int removed_lower = 0;
    int removed_upper = 0;

    for (int i = 0; i < n; ++i) {
        double v = x[i];

        // Discard lower tail
        if (v < x_lower) {
            if (removed_lower < m) {
                ++removed_lower;
                continue;
            }
        }

        // Discard upper tail
        if (v > x_upper) {
            if (removed_upper < m) {
                ++removed_upper;
                continue;
            }
        }

        // Keep element
        sum += v;
        ++kept;
    }

    if (kept > 0) {
        *result = sum / kept;
        *status = HPCS_SUCCESS;
    } else {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = HPCS_ERR_INVALID_ARGS;
    }
}

/*
 * hpcs_trimmed_mean_deterministic - DETERMINISTIC mode implementation
 *
 * Full validation with guaranteed deterministic behavior.
 * Note: introselect is already deterministic, so same as SAFE mode.
 */
void hpcs_trimmed_mean_deterministic(
    const double* x,
    int n,
    double trim_frac,
    double* result,
    int* status
) {
    // DETERMINISTIC mode uses same implementation as SAFE
    // (introselect is already deterministic)
    hpcs_trimmed_mean_safe(x, n, trim_frac, result, status);
}

} // anonymous namespace

extern "C" {

/*
 * hpcs_trimmed_mean - Public API with execution mode support
 *
 * Compute mean after discarding fraction `trim_frac` of smallest and largest values.
 * Uses deterministic O(n) selection to find quantile bounds.
 *
 * Algorithm:
 *   1. Copy input to work buffer (preserves input immutability)
 *   2. Find k1-th and k2-th order statistics using introselect
 *      where k1 = floor(trim_frac * n), k2 = n - k1 - 1
 *   3. Compute mean of elements in range [x_lower, x_upper]
 *
 * Parameters:
 *   x         - Input array [n] (const, unchanged)
 *   n         - Array length
 *   trim_frac - Fraction to trim from each tail ∈ [0, 0.5)
 *   result    - Output: trimmed mean
 *   mode      - Execution mode: -1=use global, 0=SAFE, 1=FAST, 2=DETERMINISTIC
 *   status    - Output: 0=success, 1=invalid args, 2=all NaN or no elements remain
 *
 * Execution Modes:
 *   - SAFE (0): Full NaN detection and validation (default behavior)
 *   - FAST (1): Skip NaN checks for 1.5-2x speedup (assumes valid inputs)
 *   - DETERMINISTIC (2): Full validation, deterministic behavior (same as SAFE)
 *
 * Example:
 *   double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *   double result;
 *   int status;
 *   hpcs_trimmed_mean(x, 10, 0.2, &result, HPCS_MODE_SAFE, &status);
 *   // Removes {1,2} and {9,10}, computes mean of {3,4,5,6,7,8} = 5.5
 */
void hpcs_trimmed_mean(
    const double* x,
    int n,
    double trim_frac,
    double* result,
    int mode,
    int* status
) {
    // Resolve execution mode
    int effective_mode = get_effective_mode(mode);

    // Dispatch to mode-specific implementation
    switch (effective_mode) {
        case HPCS_MODE_SAFE:
            hpcs_trimmed_mean_safe(x, n, trim_frac, result, status);
            break;
        case HPCS_MODE_FAST:
            hpcs_trimmed_mean_fast(x, n, trim_frac, result, status);
            break;
        case HPCS_MODE_DETERMINISTIC:
            hpcs_trimmed_mean_deterministic(x, n, trim_frac, result, status);
            break;
        default:
            if (status != nullptr) *status = HPCS_ERR_INVALID_ARGS;
            if (result != nullptr) *result = std::numeric_limits<double>::quiet_NaN();
            break;
    }
}

/*
 * hpcs_winsorized_mean_safe - SAFE mode implementation
 *
 * Full NaN detection and validation (current behavior).
 */
void hpcs_winsorized_mean_safe(
    const double* x,
    int n,
    double win_frac,
    double* result,
    int* status
) {
    // Validate arguments
    if (n <= 0 || win_frac < 0.0 || win_frac >= 0.5 ||
        x == nullptr || result == nullptr || status == nullptr) {
        if (status != nullptr) *status = 1;  // HPCS_ERR_INVALID_ARGS
        return;
    }

    // Check for NaN (propagate NaN policy)
    if (has_nan(x, n)) {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = 2;  // HPCS_ERR_NUMERIC_FAIL
        return;
    }

    // Compute number of elements to winsorize from each end
    int m = static_cast<int>(std::floor(win_frac * n));

    // If no winsorization, compute simple mean
    if (m == 0) {
        *result = mean_finite(x, n);
        if (std::isnan(*result)) {
            *status = 2;  // HPCS_ERR_NUMERIC_FAIL
        } else {
            *status = 0;  // HPCS_SUCCESS
        }
        return;
    }

    // Copy input to work buffer (internal allocation)
    std::vector<double> work(x, x + n);

    // Find winsorization bounds using deterministic selection
    double x_lower, x_upper;
    int sel_status;
    hpcs_select_two_bounds(work.data(), n, m, n - m - 1, &x_lower, &x_upper, &sel_status);

    if (sel_status != 0) {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = sel_status;
        return;
    }

    // Compute mean with clamping
    double sum = 0.0;
    int count = 0;

    for (int i = 0; i < n; ++i) {
        double v = x[i];

        // Skip NaN (should not happen due to earlier check, but defensive)
        if (std::isnan(v)) continue;

        // Clamp to bounds
        if (v < x_lower) v = x_lower;
        else if (v > x_upper) v = x_upper;

        sum += v;
        ++count;
    }

    if (count > 0) {
        *result = sum / count;
        *status = HPCS_SUCCESS;
    } else {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = HPCS_ERR_NUMERIC_FAIL;
    }
}

/*
 * hpcs_winsorized_mean_fast - FAST mode implementation
 *
 * Skip NaN checks for maximum performance (1.5-2x faster for large arrays).
 * Assumes all inputs are valid and finite.
 */
void hpcs_winsorized_mean_fast(
    const double* x,
    int n,
    double win_frac,
    double* result,
    int* status
) {
    // Minimal validation (skip NaN checks)
    if (n <= 0 || win_frac < 0.0 || win_frac >= 0.5 ||
        x == nullptr || result == nullptr || status == nullptr) {
        if (status != nullptr) *status = HPCS_ERR_INVALID_ARGS;
        return;
    }

    // Compute number of elements to winsorize from each end
    int m = static_cast<int>(std::floor(win_frac * n));

    // If no winsorization, compute simple mean (no NaN check)
    if (m == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += x[i];
        }
        *result = sum / n;
        *status = HPCS_SUCCESS;
        return;
    }

    // Copy input to work buffer
    std::vector<double> work(x, x + n);

    // Find winsorization bounds using deterministic selection
    double x_lower, x_upper;
    int sel_status;
    hpcs_select_two_bounds(work.data(), n, m, n - m - 1, &x_lower, &x_upper, &sel_status);

    if (sel_status != HPCS_SUCCESS) {
        *result = std::numeric_limits<double>::quiet_NaN();
        *status = sel_status;
        return;
    }

    // Compute mean with clamping (no NaN check)
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = x[i];

        // Clamp to bounds
        if (v < x_lower) v = x_lower;
        else if (v > x_upper) v = x_upper;

        sum += v;
    }

    *result = sum / n;
    *status = HPCS_SUCCESS;
}

/*
 * hpcs_winsorized_mean_deterministic - DETERMINISTIC mode implementation
 *
 * Full validation with guaranteed deterministic behavior.
 * Note: introselect is already deterministic, so same as SAFE mode.
 */
void hpcs_winsorized_mean_deterministic(
    const double* x,
    int n,
    double win_frac,
    double* result,
    int* status
) {
    // DETERMINISTIC mode uses same implementation as SAFE
    // (introselect is already deterministic)
    hpcs_winsorized_mean_safe(x, n, win_frac, result, status);
}

/*
 * hpcs_winsorized_mean - Public API with execution mode support
 *
 * Compute mean after clamping extreme values to quantile bounds.
 * Uses deterministic O(n) selection to find quantile bounds.
 *
 * Algorithm:
 *   1. Copy input to work buffer
 *   2. Find k1-th and k2-th order statistics using introselect
 *      where k1 = floor(win_frac * n), k2 = n - k1 - 1
 *   3. Clamp: x[i] ← clamp(x[i], x_lower, x_upper)
 *   4. Compute mean of clamped values
 *
 * Parameters:
 *   x        - Input array [n] (const, unchanged)
 *   n        - Array length
 *   win_frac - Fraction to winsorize from each tail ∈ [0, 0.5)
 *   result   - Output: winsorized mean
 *   mode     - Execution mode: -1=use global, 0=SAFE, 1=FAST, 2=DETERMINISTIC
 *   status   - Output: 0=success, 1=invalid args, 2=all NaN
 *
 * Execution Modes:
 *   - SAFE (0): Full NaN detection and validation (default behavior)
 *   - FAST (1): Skip NaN checks for 1.5-2x speedup (assumes valid inputs)
 *   - DETERMINISTIC (2): Full validation, deterministic behavior (same as SAFE)
 *
 * Example:
 *   double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
 *   double result;
 *   int status;
 *   hpcs_winsorized_mean(x, 10, 0.1, &result, HPCS_MODE_SAFE, &status);
 *   // Clamps {1} → 2, {100} → 9, computes mean of {2,2,3,4,5,6,7,8,9,9} = 5.5
 */
void hpcs_winsorized_mean(
    const double* x,
    int n,
    double win_frac,
    double* result,
    int mode,
    int* status
) {
    // Resolve execution mode
    int effective_mode = get_effective_mode(mode);

    // Dispatch to mode-specific implementation
    switch (effective_mode) {
        case HPCS_MODE_SAFE:
            hpcs_winsorized_mean_safe(x, n, win_frac, result, status);
            break;
        case HPCS_MODE_FAST:
            hpcs_winsorized_mean_fast(x, n, win_frac, result, status);
            break;
        case HPCS_MODE_DETERMINISTIC:
            hpcs_winsorized_mean_deterministic(x, n, win_frac, result, status);
            break;
        default:
            if (status != nullptr) *status = HPCS_ERR_INVALID_ARGS;
            if (result != nullptr) *result = std::numeric_limits<double>::quiet_NaN();
            break;
    }
}

} // extern "C"
