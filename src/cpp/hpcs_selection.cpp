/*
 * Deterministic Selection Algorithms for HPCSeries Core
 *
 * This module provides O(n) average-case, deterministic selection algorithms
 * used by robust statistics kernels (trimmed_mean, winsorized_mean).
 *
 * Algorithm: Introselect (hybrid quickselect with heapselect fallback)
 *   - Uses Hoare partition for average O(n) performance
 *   - Deterministic pivot selection (median-of-three)
 *   - Falls back to heapselect when recursion depth exceeds 2*log(n)
 *   - Guarantees O(n log n) worst-case performance
 *
 * Reference:
 *   - Musser, D. R. (1997). "Introspective Sorting and Selection Algorithms"
 *   - Hoare, C. A. R. (1961). "Algorithm 65: Find"
 *
 * Author: HPCSeries Core Library
 * Version: 0.8.0
 * Date: 2025-12-23
 */

#include <algorithm>
#include <cmath>
#include <cstddef>

// Anonymous namespace for internal helpers
namespace {

/*
 * Median-of-three pivot selection for better average-case performance.
 * Returns index of median among arr[left], arr[mid], arr[right].
 */
inline size_t median_of_three(double* arr, size_t left, size_t right) {
    size_t mid = left + (right - left) / 2;

    // Sort three elements: arr[left], arr[mid], arr[right]
    if (arr[mid] < arr[left]) std::swap(arr[left], arr[mid]);
    if (arr[right] < arr[left]) std::swap(arr[left], arr[right]);
    if (arr[right] < arr[mid]) std::swap(arr[mid], arr[right]);

    // Return middle element
    return mid;
}

/*
 * Hoare partition scheme (more efficient than Lomuto).
 * Partitions arr[left..right] around pivot at arr[pivot_idx].
 * Returns final position of pivot.
 */
size_t hoare_partition(double* arr, size_t left, size_t right, size_t pivot_idx) {
    double pivot = arr[pivot_idx];

    // Move pivot to end
    std::swap(arr[pivot_idx], arr[right]);

    size_t i = left;
    for (size_t j = left; j < right; ++j) {
        if (arr[j] < pivot) {
            std::swap(arr[i], arr[j]);
            ++i;
        }
    }

    // Move pivot to final position
    std::swap(arr[i], arr[right]);
    return i;
}

/*
 * Heapselect fallback for worst-case protection.
 * Partially sorts arr[left..right] to place k-th smallest at position k.
 * Uses partial heap construction - O(n + k log n).
 */
void heapselect(double* arr, size_t left, size_t right, size_t k) {
    std::partial_sort(arr + left, arr + k + 1, arr + right + 1);
}

/*
 * Introselect: deterministic quickselect with heapselect fallback.
 * Finds k-th smallest element in arr[left..right] (0-indexed).
 * Modifies array in-place.
 */
void introselect_impl(double* arr, size_t left, size_t right, size_t k, size_t max_depth) {
    while (right > left) {
        // Fallback to heapselect if recursion too deep
        if (max_depth == 0) {
            heapselect(arr, left, right, k);
            return;
        }
        --max_depth;

        // Choose pivot using median-of-three
        size_t pivot_idx = median_of_three(arr, left, right);

        // Partition around pivot
        pivot_idx = hoare_partition(arr, left, right, pivot_idx);

        // Recurse on appropriate partition
        if (k == pivot_idx) {
            return;  // Found k-th element
        } else if (k < pivot_idx) {
            right = pivot_idx - 1;  // Search left partition
        } else {
            left = pivot_idx + 1;   // Search right partition
        }
    }
}

} // anonymous namespace

extern "C" {

/*
 * hpcs_select_kth
 *
 * Find the k-th smallest element in array (0-indexed).
 * Uses introselect algorithm: O(n) average case, O(n log n) worst case.
 * Array is partially sorted in-place (k-th element at position k).
 *
 * Parameters:
 *   arr    - Array to select from (modified in-place)
 *   n      - Array length
 *   k      - Target rank (0-indexed, 0 = minimum, n-1 = maximum)
 *   result - Output: k-th smallest value
 *   status - Output: 0=success, 1=invalid args
 *
 * Example:
 *   double arr[] = {3.0, 1.0, 4.0, 1.0, 5.0};
 *   double result;
 *   int status;
 *   hpcs_select_kth(arr, 5, 2, &result, &status);  // result = 3.0 (median)
 */
void hpcs_select_kth(
    double* arr,
    int n,
    int k,
    double* result,
    int* status
) {
    // Validate arguments
    if (n <= 0 || k < 0 || k >= n || arr == nullptr || result == nullptr || status == nullptr) {
        if (status != nullptr) *status = 1;  // HPCS_ERR_INVALID_ARGS
        return;
    }

    // Compute max recursion depth: 2 * log2(n)
    size_t max_depth = static_cast<size_t>(2.0 * std::log2(static_cast<double>(n)));

    // Run introselect
    introselect_impl(arr, 0, n - 1, static_cast<size_t>(k), max_depth);

    // Return k-th element
    *result = arr[k];
    *status = 0;  // HPCS_SUCCESS
}

/*
 * hpcs_select_two_bounds
 *
 * Efficiently find two order statistics: k1-th and k2-th smallest (k1 < k2).
 * Uses introselect twice with optimized partitioning.
 * Useful for computing quantile bounds in one pass.
 *
 * Parameters:
 *   arr      - Array to select from (modified in-place)
 *   n        - Array length
 *   k1       - Lower rank (0-indexed)
 *   k2       - Upper rank (0-indexed, must be > k1)
 *   lower    - Output: k1-th smallest value
 *   upper    - Output: k2-th smallest value
 *   status   - Output: 0=success, 1=invalid args
 *
 * Example:
 *   // Find 10th and 90th percentiles for trimming
 *   double arr[100];
 *   double p10, p90;
 *   int status;
 *   hpcs_select_two_bounds(arr, 100, 9, 89, &p10, &p90, &status);
 */
void hpcs_select_two_bounds(
    double* arr,
    int n,
    int k1,
    int k2,
    double* lower,
    double* upper,
    int* status
) {
    // Validate arguments
    if (n <= 0 || k1 < 0 || k2 >= n || k1 >= k2 ||
        arr == nullptr || lower == nullptr || upper == nullptr || status == nullptr) {
        if (status != nullptr) *status = 1;  // HPCS_ERR_INVALID_ARGS
        return;
    }

    size_t max_depth = static_cast<size_t>(2.0 * std::log2(static_cast<double>(n)));

    // Find k1-th element first
    introselect_impl(arr, 0, n - 1, static_cast<size_t>(k1), max_depth);
    *lower = arr[k1];

    // Find k2-th element in right partition only (arr[k1+1..n-1])
    if (k2 > k1 + 1) {
        introselect_impl(arr, k1 + 1, n - 1, static_cast<size_t>(k2), max_depth);
    }
    *upper = arr[k2];

    *status = 0;  // HPCS_SUCCESS
}

} // extern "C"
