// tests/test_core_cpp.cpp
//
// HPCSeries-QA-Scientist
// Small-array correctness tests for v0.1 kernels via the C interface.
//
// Assumes C prototypes from include/hpcs_core.h:
//
//   void hpcs_rolling_sum   (const double *x, int n, int window, double *y, int *status);
//   void hpcs_rolling_mean  (const double *x, int n, int window, double *y, int *status);
//   void hpcs_group_reduce_sum (const double *x, int n,
//                               const int *group_ids, int n_groups, double *y, int *status);
//   void hpcs_group_reduce_mean(const double *x, int n,
//                               const int *group_ids, int n_groups, double *y, int *status);
//   void hpcs_reduce_sum    (const double *x, int n, double *out, int *status);
//   void hpcs_reduce_min    (const double *x, int n, double *out, int *status);
//   void hpcs_reduce_max    (const double *x, int n, double *out, int *status);
//   void hpcs_zscore        (const double *x, int n, double *y, int *status);
//
//   void hpcs_fill_value    (double *x, int n, double value, int *status);
//   void hpcs_copy          (double *dst, const double *src, int n, int *status);
//
// Build example (adjust include/lib paths as needed):
//   g++ -std=c++17 -I../include test_core_cpp.cpp -L../build -lhpcs_core -o test_core_cpp
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>
#include <algorithm>

#include "hpcs_core.h"

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

constexpr double ABS_TOL = 1e-12;
constexpr double REL_TOL = 1e-12;

bool almost_equal(double a, double b,
                  double abs_tol = ABS_TOL,
                  double rel_tol = REL_TOL)
{
    if (std::isnan(a) && std::isnan(b)) {
        // treat NaN vs NaN as equal for tests that expect NaN propagation
        return true;
    }
    if (std::isinf(a) || std::isinf(b)) {
        return a == b;
    }
    double diff = std::fabs(a - b);
    if (diff <= abs_tol) return true;
    double maxab = std::max(std::fabs(a), std::fabs(b));
    if (maxab == 0.0) return diff <= abs_tol;
    return diff <= rel_tol * maxab;
}

void check(bool cond, const char* msg)
{
    if (!cond) {
        std::fprintf(stderr, "[FAIL] %s\n", msg);
        std::exit(1);
    }
}

void check_status_ok(int status, const char* msg)
{
    if (status != 0) {
        std::fprintf(stderr, "[FAIL] %s (status=%d, expected 0)\n", msg, status);
        std::exit(1);
    }
}

void check_status_not_ok(int status, const char* msg)
{
    if (status == 0) {
        std::fprintf(stderr, "[FAIL] %s (status=%d, expected non-zero)\n", msg, status);
        std::exit(1);
    }
}

void check_array_close(const std::vector<double>& got,
                       const std::vector<double>& ref,
                       const char* msg)
{
    check(got.size() == ref.size(), "Size mismatch in check_array_close");
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!almost_equal(got[i], ref[i])) {
            std::fprintf(stderr,
                         "[FAIL] %s at index %zu: got=%.17g, expected=%.17g\n",
                         msg, i, got[i], ref[i]);
            std::exit(1);
        }
    }
}

// -----------------------------------------------------------------------------
// Reference implementations (CPU-simple, numerically straightforward)
// -----------------------------------------------------------------------------

// Truncated sliding-window rolling sum:
// y[i] = sum of x[max(0, i-window+1) .. i]
void ref_rolling_sum(const std::vector<double>& x, int window,
                     std::vector<double>& y)
{
    int n = static_cast<int>(x.size());
    y.assign(n, 0.0);

    if (n <= 0 || window <= 0) {
        return;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
        if (i >= window) {
            sum -= x[i - window];
        }
        y[i] = sum;
    }
}

// Rolling mean uses same sliding sum divided by effective window length
void ref_rolling_mean(const std::vector<double>& x, int window,
                      std::vector<double>& y)
{
    int n = static_cast<int>(x.size());
    y.assign(n, 0.0);

    if (n <= 0 || window <= 0) {
        return;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
        if (i >= window) {
            sum -= x[i - window];
        }
        int k = std::min(i + 1, window); // current window length
        y[i] = sum / static_cast<double>(k);
    }
}

// Grouped sum: ignore invalid group_ids (<0 or >= n_groups)
void ref_group_reduce_sum(const std::vector<double>& x,
                          const std::vector<int>& gid,
                          int n_groups,
                          std::vector<double>& out)
{
    int n = static_cast<int>(x.size());
    out.assign(n_groups, 0.0);

    if (n <= 0 || n_groups <= 0) {
        return;
    }

    for (int i = 0; i < n; ++i) {
        int g = gid[i];
        if (g < 0 || g >= n_groups) continue;
        out[g] += x[i];
    }
}

// Grouped mean: NaN for groups with zero count
void ref_group_reduce_mean(const std::vector<double>& x,
                           const std::vector<int>& gid,
                           int n_groups,
                           std::vector<double>& out)
{
    int n = static_cast<int>(x.size());
    out.assign(n_groups, std::numeric_limits<double>::quiet_NaN());

    if (n <= 0 || n_groups <= 0) {
        return;
    }

    std::vector<double> sum(n_groups, 0.0);
    std::vector<int> count(n_groups, 0);

    for (int i = 0; i < n; ++i) {
        int g = gid[i];
        if (g < 0 || g >= n_groups) continue;
        sum[g] += x[i];
        count[g] += 1;
    }

    for (int g = 0; g < n_groups; ++g) {
        if (count[g] > 0) {
            out[g] = sum[g] / static_cast<double>(count[g]);
        } else {
            out[g] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

double ref_reduce_sum(const std::vector<double>& x)
{
    double s = 0.0;
    for (double v : x) s += v;
    return s;
}

double ref_reduce_min(const std::vector<double>& x)
{
    if (x.empty()) {
        return std::numeric_limits<double>::infinity(); // sentinel
    }
    double m = x[0];
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] < m) m = x[i];
    }
    return m;
}

double ref_reduce_max(const std::vector<double>& x)
{
    if (x.empty()) {
        return -std::numeric_limits<double>::infinity(); // sentinel
    }
    double m = x[0];
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] > m) m = x[i];
    }
    return m;
}

// Simple two-pass z-score (population std). For small n this is fine.
void ref_zscore(const std::vector<double>& x,
                std::vector<double>& y,
                int& status_out)
{
    int n = static_cast<int>(x.size());
    y.assign(n, 0.0);
    status_out = 0;

    if (n <= 0) {
        status_out = 1; // we'll treat this as "invalid args"
        return;
    }

    double sum = 0.0;
    for (double v : x) sum += v;
    double mean = sum / static_cast<double>(n);

    double sq = 0.0;
    for (double v : x) {
        double d = v - mean;
        sq += d * d;
    }
    double variance = sq / static_cast<double>(n);
    if (variance < 0.0) variance = 0.0;
    double std = std::sqrt(variance);

    if (std == 0.0) {
        // All elements equal -> z-scores = 0, status=2 (numeric failure)
        status_out = 2;
        std::fill(y.begin(), y.end(), 0.0);
        return;
    }

    for (int i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) / std;
    }
}

// -----------------------------------------------------------------------------
// Tests: rolling_sum
// -----------------------------------------------------------------------------

void test_rolling_sum_basic()
{
    std::printf("Running test_rolling_sum_basic...\n");

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    int n = static_cast<int>(x.size());
    int window = 3;

    std::vector<double> ref;
    ref_rolling_sum(x, window, ref);

    std::vector<double> y(n, 0.0);
    int status;
    hpcs_rolling_sum(x.data(), n, window, y.data(), &status);
    check_status_ok(status, "hpcs_rolling_sum basic status");

    check_array_close(y, ref, "hpcs_rolling_sum basic values");
}

void test_rolling_sum_window_1_and_big()
{
    std::printf("Running test_rolling_sum_window_1_and_big...\n");

    std::vector<double> x = {2.0, -1.0, 0.5};
    int n = static_cast<int>(x.size());

    // window = 1 => y == x
    {
        int window = 1;
        std::vector<double> ref;
        ref_rolling_sum(x, window, ref);

        std::vector<double> y(n, 0.0);
        int status;
        hpcs_rolling_sum(x.data(), n, window, y.data(), &status);
        check_status_ok(status, "hpcs_rolling_sum window=1 status");
        check_array_close(y, ref, "hpcs_rolling_sum window=1 values");
    }

    // window > n => truncated window behaves like prefix sum
    {
        int window = 10;
        std::vector<double> ref;
        ref_rolling_sum(x, window, ref);

        std::vector<double> y(n, 0.0);
        int status;
        hpcs_rolling_sum(x.data(), n, window, y.data(), &status);
        check_status_ok(status, "hpcs_rolling_sum window>n status");
        check_array_close(y, ref, "hpcs_rolling_sum window>n values");
    }
}

void test_rolling_sum_nan_propagation()
{
    std::printf("Running test_rolling_sum_nan_propagation...\n");

    double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> x = {1.0, nan, 3.0};
    int n = static_cast<int>(x.size());
    int window = 2;

    std::vector<double> ref;
    ref_rolling_sum(x, window, ref);

    std::vector<double> y(n, 0.0);
    int status;
    hpcs_rolling_sum(x.data(), n, window, y.data(), &status);
    check_status_ok(status, "hpcs_rolling_sum NaN status");
    check_array_close(y, ref, "hpcs_rolling_sum NaN values");
}

void test_rolling_sum_errors()
{
    std::printf("Running test_rolling_sum_errors...\n");

    std::vector<double> x = {1.0, 2.0};
    double y[2];

    // window <= 0
    int status;
    hpcs_rolling_sum(x.data(), 2, 0, y, &status);
    check_status_not_ok(status, "hpcs_rolling_sum window<=0 should error");

    // n <= 0
    hpcs_rolling_sum(x.data(), 0, 2, y, &status);
    check_status_not_ok(status, "hpcs_rolling_sum n<=0 should error");
}

// -----------------------------------------------------------------------------
// Tests: rolling_mean
// -----------------------------------------------------------------------------

void test_rolling_mean_basic()
{
    std::printf("Running test_rolling_mean_basic...\n");

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    int n = static_cast<int>(x.size());
    int window = 3;

    std::vector<double> ref;
    ref_rolling_mean(x, window, ref);

    std::vector<double> y(n, 0.0);
    int status;
    hpcs_rolling_mean(x.data(), n, window, y.data(), &status);
    check_status_ok(status, "hpcs_rolling_mean basic status");
    check_array_close(y, ref, "hpcs_rolling_mean basic values");
}

void test_rolling_mean_window_1_and_big()
{
    std::printf("Running test_rolling_mean_window_1_and_big...\n");

    std::vector<double> x = {2.0, -2.0, 2.0};
    int n = static_cast<int>(x.size());

    // window = 1 => y == x
    {
        int window = 1;
        std::vector<double> ref;
        ref_rolling_mean(x, window, ref);

        std::vector<double> y(n, 0.0);
        int status;
        hpcs_rolling_mean(x.data(), n, window, y.data(), &status);
        check_status_ok(status, "hpcs_rolling_mean window=1 status");
        check_array_close(y, ref, "hpcs_rolling_mean window=1 values");
    }

    // window > n
    {
        int window = 10;
        std::vector<double> ref;
        ref_rolling_mean(x, window, ref);

        std::vector<double> y(n, 0.0);
        int status;
        hpcs_rolling_mean(x.data(), n, window, y.data(), &status);
        check_status_ok(status, "hpcs_rolling_mean window>n status");
        check_array_close(y, ref, "hpcs_rolling_mean window>n values");
    }
}

void test_rolling_mean_errors()
{
    std::printf("Running test_rolling_mean_errors...\n");

    std::vector<double> x = {1.0, 2.0};
    double y[2];

    int status;
    hpcs_rolling_mean(x.data(), 2, 0, y, &status);
    check_status_not_ok(status, "hpcs_rolling_mean window<=0 should error");

    hpcs_rolling_mean(x.data(), 0, 2, y, &status);
    check_status_not_ok(status, "hpcs_rolling_mean n<=0 should error");
}

// -----------------------------------------------------------------------------
// Tests: grouped reductions
// -----------------------------------------------------------------------------

void test_group_reduce_sum_basic()
{
    std::printf("Running test_group_reduce_sum_basic...\n");

    // x indexed by i: 0..4
    std::vector<double> x    = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<int>    gid  = {0,   1,   0,   1,   1  };  // 2 groups
    int n = static_cast<int>(x.size());
    int n_groups = 2;

    std::vector<double> ref;
    ref_group_reduce_sum(x, gid, n_groups, ref);

    std::vector<double> y(n_groups, 0.0);
    int status;
    hpcs_group_reduce_sum(x.data(), n, gid.data(), n_groups, y.data(), &status);
    check_status_ok(status, "hpcs_group_reduce_sum basic status");
    check_array_close(y, ref, "hpcs_group_reduce_sum basic values");
}

void test_group_reduce_sum_invalid_groups()
{
    std::printf("Running test_group_reduce_sum_invalid_groups...\n");

    std::vector<double> x   = {1.0, 2.0, 3.0, 4.0};
    std::vector<int> gid    = {0,   -1,  2,   1   }; // -1 and 2 are invalid when n_groups=2
    int n = static_cast<int>(x.size());
    int n_groups = 2;

    std::vector<double> ref;
    ref_group_reduce_sum(x, gid, n_groups, ref);

    std::vector<double> y(n_groups, 0.0);
    int status;
    hpcs_group_reduce_sum(x.data(), n, gid.data(), n_groups, y.data(), &status);
    check_status_ok(status, "hpcs_group_reduce_sum invalid groups status");
    check_array_close(y, ref, "hpcs_group_reduce_sum invalid groups values");
}

void test_group_reduce_mean_basic()
{
    std::printf("Running test_group_reduce_mean_basic...\n");

    std::vector<double> x   = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<int> gid    = {0,   1,   0,   1,   1  };
    int n = static_cast<int>(x.size());
    int n_groups = 2;

    std::vector<double> ref;
    ref_group_reduce_mean(x, gid, n_groups, ref);

    std::vector<double> y(n_groups, 0.0);
    int status;
    hpcs_group_reduce_mean(x.data(), n, gid.data(), n_groups, y.data(), &status);
    check_status_ok(status, "hpcs_group_reduce_mean basic status");
    check_array_close(y, ref, "hpcs_group_reduce_mean basic values");
}

void test_group_reduce_mean_empty_group()
{
    std::printf("Running test_group_reduce_mean_empty_group...\n");

    // group 2 has no elements
    std::vector<double> x   = {10.0, 20.0, 30.0};
    std::vector<int> gid    = {0,    1,    0   };
    int n = static_cast<int>(x.size());
    int n_groups = 3;

    std::vector<double> ref;
    ref_group_reduce_mean(x, gid, n_groups, ref);

    std::vector<double> y(n_groups, 0.0);
    int status;
    hpcs_group_reduce_mean(x.data(), n, gid.data(), n_groups, y.data(), &status);
    check_status_ok(status, "hpcs_group_reduce_mean empty group status");

    // group 2 should be NaN
    check_array_close(y, ref, "hpcs_group_reduce_mean empty group values");
}

void test_group_reduce_errors()
{
    std::printf("Running test_group_reduce_errors...\n");

    std::vector<double> x = {1.0, 2.0};
    std::vector<int> gid  = {0, 1};
    double y[2];

    int status;
    hpcs_group_reduce_sum(x.data(), 0, gid.data(), 2, y, &status);
    check_status_not_ok(status, "hpcs_group_reduce_sum n<=0 should error");

    hpcs_group_reduce_sum(x.data(), 2, gid.data(), 0, y, &status);
    check_status_not_ok(status, "hpcs_group_reduce_sum n_groups<=0 should error");

    hpcs_group_reduce_mean(x.data(), 0, gid.data(), 2, y, &status);
    check_status_not_ok(status, "hpcs_group_reduce_mean n<=0 should error");

    hpcs_group_reduce_mean(x.data(), 2, gid.data(), 0, y, &status);
    check_status_not_ok(status, "hpcs_group_reduce_mean n_groups<=0 should error");
}

// -----------------------------------------------------------------------------
// Tests: simple reductions
// -----------------------------------------------------------------------------

void test_reduce_sum_min_max_basic()
{
    std::printf("Running test_reduce_sum_min_max_basic...\n");

    std::vector<double> x = {-1.0, 0.0, 2.5, 10.0};
    int n = static_cast<int>(x.size());

    double ref_sum = ref_reduce_sum(x);
    double ref_min = ref_reduce_min(x);
    double ref_max = ref_reduce_max(x);

    double out_sum = 0.0, out_min = 0.0, out_max = 0.0;

    int status;
    hpcs_reduce_sum(x.data(), n, &out_sum, &status);
    check_status_ok(status, "hpcs_reduce_sum basic status");
    check(almost_equal(out_sum, ref_sum), "hpcs_reduce_sum basic value");

    hpcs_reduce_min(x.data(), n, &out_min, &status);
    check_status_ok(status, "hpcs_reduce_min basic status");
    check(almost_equal(out_min, ref_min), "hpcs_reduce_min basic value");

    hpcs_reduce_max(x.data(), n, &out_max, &status);
    check_status_ok(status, "hpcs_reduce_max basic status");
    check(almost_equal(out_max, ref_max), "hpcs_reduce_max basic value");
}

void test_reduce_sum_min_max_nan()
{
    std::printf("Running test_reduce_sum_min_max_nan...\n");

    double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> x = {1.0, nan, 3.0};
    int n = static_cast<int>(x.size());

    double out_sum = 0.0, out_min = 0.0, out_max = 0.0;

    int status;
    hpcs_reduce_sum(x.data(), n, &out_sum, &status);
    check_status_ok(status, "hpcs_reduce_sum NaN status");
    check(std::isnan(out_sum), "hpcs_reduce_sum NaN propagation");

    hpcs_reduce_min(x.data(), n, &out_min, &status);
    check_status_ok(status, "hpcs_reduce_min NaN status");
    check(std::isnan(out_min), "hpcs_reduce_min NaN propagation");

    hpcs_reduce_max(x.data(), n, &out_max, &status);
    check_status_ok(status, "hpcs_reduce_max NaN status");
    check(std::isnan(out_max), "hpcs_reduce_max NaN propagation");
}

void test_reduce_errors()
{
    std::printf("Running test_reduce_errors...\n");

    std::vector<double> x = {1.0, 2.0};
    double out = 0.0;

    int status;
    hpcs_reduce_sum(x.data(), 0, &out, &status);
    check_status_not_ok(status, "hpcs_reduce_sum n<=0 should error");

    hpcs_reduce_min(x.data(), 0, &out, &status);
    check_status_not_ok(status, "hpcs_reduce_min n<=0 should error");

    hpcs_reduce_max(x.data(), 0, &out, &status);
    check_status_not_ok(status, "hpcs_reduce_max n<=0 should error");
}

// -----------------------------------------------------------------------------
// Tests: z-score transform
// -----------------------------------------------------------------------------

void test_zscore_basic()
{
    std::printf("Running test_zscore_basic...\n");

    std::vector<double> x = {1.0, 2.0, 3.0};
    int n = static_cast<int>(x.size());

    std::vector<double> ref;
    int ref_status = 0;
    ref_zscore(x, ref, ref_status);

    std::vector<double> y(n, 0.0);
    int status;
    hpcs_zscore(x.data(), n, y.data(), &status);
    check_status_ok(status, "hpcs_zscore basic status");
    check(ref_status == 0, "ref_zscore basic status");
    check_array_close(y, ref, "hpcs_zscore basic values");
}

void test_zscore_constant()
{
    std::printf("Running test_zscore_constant...\n");

    std::vector<double> x = {5.0, 5.0, 5.0};
    int n = static_cast<int>(x.size());

    std::vector<double> ref;
    int ref_status = 0;
    ref_zscore(x, ref, ref_status);

    std::vector<double> y(n, 0.0);
    int status;
    hpcs_zscore(x.data(), n, y.data(), &status);

    // Expect numeric failure (status=2) and all zeros
    check(status == 2, "hpcs_zscore constant status should be 2");
    check(ref_status == 2, "ref_zscore constant status should be 2");
    check_array_close(y, ref, "hpcs_zscore constant values");
}

void test_zscore_errors()
{
    std::printf("Running test_zscore_errors...\n");

    std::vector<double> x; // empty
    double* y = nullptr;

    int status;
    hpcs_zscore(x.data(), 0, y, &status);
    check_status_not_ok(status, "hpcs_zscore n<=0 should error");
}

// -----------------------------------------------------------------------------
// Tests: array utilities (fill_value, copy)
// -----------------------------------------------------------------------------

void test_array_utilities()
{
    std::printf("Running test_array_utilities...\n");

    std::vector<double> x(5, 0.0);
    int n = static_cast<int>(x.size());
    int status;

    // fill_value
    hpcs_fill_value(x.data(), n, 3.14159, &status);
    check_status_ok(status, "hpcs_fill_value");
    for (int i = 0; i < n; ++i) {
        check(almost_equal(x[i], 3.14159), "hpcs_fill_value basic");
    }

    // copy
    std::vector<double> y(n, 0.0);
    hpcs_copy(y.data(), x.data(), n, &status);
    check_status_ok(status, "hpcs_copy");
    check_array_close(y, x, "hpcs_copy basic");
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main()
{
    std::printf("=== HPCSeries Core v0.1: C++ QA Test Suite ===\n");

    // Rolling operations
    test_rolling_sum_basic();
    test_rolling_sum_window_1_and_big();
    test_rolling_sum_nan_propagation();
    test_rolling_sum_errors();

    test_rolling_mean_basic();
    test_rolling_mean_window_1_and_big();
    test_rolling_mean_errors();

    // Grouped reductions
    test_group_reduce_sum_basic();
    test_group_reduce_sum_invalid_groups();
    test_group_reduce_mean_basic();
    test_group_reduce_mean_empty_group();
    test_group_reduce_errors();

    // Simple reductions
    test_reduce_sum_min_max_basic();
    test_reduce_sum_min_max_nan();
    test_reduce_errors();

    // Z-score transform
    test_zscore_basic();
    test_zscore_constant();
    test_zscore_errors();

    // Utilities
    test_array_utilities();

    std::printf("All tests passed.\n");
    return 0;
}
