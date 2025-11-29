/*
 * HPCSeries Core v0.4 Parallel Benchmark Suite
 *
 * This benchmark measures the OpenMP speedup for v0.4 batched/axis operations:
 * 1. Batched rolling operations (sum, mean, median, MAD)
 * 2. Axis-0 reductions (sum, mean, min, max)
 * 3. Axis-1 reductions (sum, mean, median, MAD, quantile, robust_zscore)
 * 4. Anomaly detection (axis-1, robust axis-1, rolling 2D)
 *
 * Strategy:
 *   - Run with OMP_NUM_THREADS=1 (serial baseline)
 *   - Run with OMP_NUM_THREADS=4,8,16 (parallel)
 *   - Measure speedup vs serial version
 *
 * Output: CSV with columns:
 *   n, m, kernel, threads, elapsed_seconds, speedup
 *
 * Usage:
 *   OMP_NUM_THREADS=1 ./bench_v04_parallel > baseline.csv
 *   OMP_NUM_THREADS=8 ./bench_v04_parallel > parallel.csv
 */

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

extern "C" {
    // Batched rolling operations
    void hpcs_rolling_sum_batched(const double* x, int n, int m, int window, double* y, int* status);
    void hpcs_rolling_mean_batched(const double* x, int n, int m, int window, double* y, int* status);
    void hpcs_rolling_median_batched(const double* x, int n, int m, int window, double* y, int* status);
    void hpcs_rolling_mad_batched(const double* x, int n, int m, int window, double* y, int* status);

    // Axis-0 reductions
    void hpcs_reduce_sum_axis0(const double* x, int n, int m, double* y, int* status);
    void hpcs_reduce_mean_axis0(const double* x, int n, int m, double* y, int* status);
    void hpcs_reduce_min_axis0(const double* x, int n, int m, double* y, int* status);
    void hpcs_reduce_max_axis0(const double* x, int n, int m, double* y, int* status);

    // Axis-1 reductions
    void hpcs_reduce_sum_axis1(const double* x, int n, int m, double* y, int* status);
    void hpcs_reduce_mean_axis1(const double* x, int n, int m, double* y, int* status);
    void hpcs_median_axis1(const double* x, int n, int m, double* y, int* status);
    void hpcs_mad_axis1(const double* x, int n, int m, double* y, int* status);
    void hpcs_quantile_axis1(const double* x, int n, int m, double q, double* y, int* status);
    void hpcs_robust_zscore_axis1(const double* x, int n, int m, double scale, double* y, int* status);

    // Anomaly detection
    void hpcs_detect_anomalies_axis1(const double* x, int n, int m, double threshold, int* mask, int* status);
    void hpcs_detect_anomalies_robust_axis1(const double* x, int n, int m, double threshold, int* mask, int* status, const double* scale);
    void hpcs_rolling_detect_anomalies_2d(const double* x, int n, int m, int window, double threshold, int* mask, int* status);
}

struct Config {
    int n;      // rows
    int m;      // columns
    int window; // window size for rolling ops
};

void benchmark(const std::string& kernel_name,
               const Config& cfg,
               std::function<void()> fn) {
    int num_threads = omp_get_max_threads();

    // Warmup
    fn();

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Output: n, m, kernel, threads, elapsed_seconds
    std::cout << cfg.n << "," << cfg.m << "," << kernel_name << ","
              << num_threads << "," << std::fixed << std::setprecision(6)
              << elapsed << std::endl;
}

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    // Test configurations: (n, m, window)
    // Focus on arrays > 100k elements (OpenMP threshold)
    std::vector<Config> configs = {
        {1000, 500, 50},      // 500k elements
        {2000, 1000, 50},     // 2M elements
        {5000, 1000, 50},     // 5M elements
        {10000, 1000, 100},   // 10M elements
    };

    std::cout << "n,m,kernel,threads,elapsed_seconds" << std::endl;

    for (const auto& cfg : configs) {
        int n = cfg.n;
        int m = cfg.m;
        int window = cfg.window;
        int total = n * m;

        // Generate test data
        std::vector<double> data(total);
        for (int i = 0; i < total; ++i) {
            data[i] = dist(rng);
        }

        std::vector<double> output(total);
        std::vector<int> mask_output(total);
        int status = 0;

        // ================================================================
        // BATCHED ROLLING OPERATIONS
        // ================================================================

        benchmark("rolling_sum_batched", cfg, [&]() {
            hpcs_rolling_sum_batched(data.data(), n, m, window, output.data(), &status);
        });

        benchmark("rolling_mean_batched", cfg, [&]() {
            hpcs_rolling_mean_batched(data.data(), n, m, window, output.data(), &status);
        });

        benchmark("rolling_median_batched", cfg, [&]() {
            hpcs_rolling_median_batched(data.data(), n, m, window, output.data(), &status);
        });

        benchmark("rolling_mad_batched", cfg, [&]() {
            hpcs_rolling_mad_batched(data.data(), n, m, window, output.data(), &status);
        });

        // ================================================================
        // AXIS-0 REDUCTIONS (column-wise)
        // ================================================================

        std::vector<double> axis0_output(m);

        benchmark("reduce_sum_axis0", cfg, [&]() {
            hpcs_reduce_sum_axis0(data.data(), n, m, axis0_output.data(), &status);
        });

        benchmark("reduce_mean_axis0", cfg, [&]() {
            hpcs_reduce_mean_axis0(data.data(), n, m, axis0_output.data(), &status);
        });

        benchmark("reduce_min_axis0", cfg, [&]() {
            hpcs_reduce_min_axis0(data.data(), n, m, axis0_output.data(), &status);
        });

        benchmark("reduce_max_axis0", cfg, [&]() {
            hpcs_reduce_max_axis0(data.data(), n, m, axis0_output.data(), &status);
        });

        // ================================================================
        // AXIS-1 REDUCTIONS (row-wise)
        // ================================================================

        std::vector<double> axis1_output(n);

        benchmark("reduce_sum_axis1", cfg, [&]() {
            hpcs_reduce_sum_axis1(data.data(), n, m, axis1_output.data(), &status);
        });

        benchmark("reduce_mean_axis1", cfg, [&]() {
            hpcs_reduce_mean_axis1(data.data(), n, m, axis1_output.data(), &status);
        });

        benchmark("median_axis1", cfg, [&]() {
            hpcs_median_axis1(data.data(), n, m, axis1_output.data(), &status);
        });

        benchmark("mad_axis1", cfg, [&]() {
            hpcs_mad_axis1(data.data(), n, m, axis1_output.data(), &status);
        });

        benchmark("quantile_axis1", cfg, [&]() {
            double q = 0.75;
            hpcs_quantile_axis1(data.data(), n, m, q, axis1_output.data(), &status);
        });

        std::vector<double> zscore_output(total);
        benchmark("robust_zscore_axis1", cfg, [&]() {
            double scale = 1.4826;
            hpcs_robust_zscore_axis1(data.data(), n, m, scale, zscore_output.data(), &status);
        });

        // ================================================================
        // ANOMALY DETECTION
        // ================================================================

        benchmark("detect_anomalies_axis1", cfg, [&]() {
            double threshold = 3.0;
            hpcs_detect_anomalies_axis1(data.data(), n, m, threshold, mask_output.data(), &status);
        });

        benchmark("detect_anomalies_robust_axis1", cfg, [&]() {
            double threshold = 3.0;
            double scale = 1.4826;
            hpcs_detect_anomalies_robust_axis1(data.data(), n, m, threshold, mask_output.data(), &status, &scale);
        });

        // Rolling anomaly detection (expensive, only for smaller configs)
        if (total <= 2000000) {  // 2M elements max
            benchmark("rolling_detect_anomalies_2d", cfg, [&]() {
                double threshold = 3.0;
                hpcs_rolling_detect_anomalies_2d(data.data(), n, m, window, threshold, mask_output.data(), &status);
            });
        }
    }

    return 0;
}
