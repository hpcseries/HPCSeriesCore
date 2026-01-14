/*
 * Benchmark Comparison: Original vs Optimized v0.3 Functions
 *
 * This benchmark measures the speedup achieved by:
 * 1. OpenMP parallel versions of median, mad, quantile, robust_zscore
 * 2. C++ heap-based fast rolling_median and rolling_mad
 *
 * Output: CSV with columns:
 *   n, kernel, version, elapsed_seconds, speedup
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../include/hpcs_core.h"

struct BenchResult {
    int n;
    std::string kernel;
    std::string version;
    double elapsed;
    double speedup;
};

int main() {
    // Test sizes: focus on large arrays where optimization matters
    const std::vector<int> sizes = {100000, 500000, 1000000, 5000000, 10000000};
    const int window = 100;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<BenchResult> results;

    std::cout << "n,kernel,version,elapsed_seconds,speedup" << std::endl;

    for (int n : sizes) {
        // Generate random data
        std::vector<double> data(n);
        for (int i = 0; i < n; ++i) {
            data[i] = dist(rng);
        }

        int status = 0;
        double scalar_result = 0.0;

        // ====================================================================
        // MEDIAN: Original vs Parallel
        // ====================================================================
        {
            // Original
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_median(data.data(), n, &scalar_result, HPCS_MODE_FAST, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_orig = std::chrono::duration<double>(end - start).count();

            // Parallel
            start = std::chrono::high_resolution_clock::now();
            hpcs_median_parallel(data.data(), n, &scalar_result, &status);
            end = std::chrono::high_resolution_clock::now();
            double elapsed_par = std::chrono::duration<double>(end - start).count();

            double speedup = elapsed_orig / elapsed_par;

            std::cout << n << ",median,original," << std::fixed << std::setprecision(6) << elapsed_orig << ",1.000000" << std::endl;
            std::cout << n << ",median,parallel," << std::fixed << std::setprecision(6) << elapsed_par << "," << speedup << std::endl;
        }

        // ====================================================================
        // MAD: Original vs Parallel
        // ====================================================================
        {
            // Original
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_mad(data.data(), n, &scalar_result, HPCS_MODE_FAST, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_orig = std::chrono::duration<double>(end - start).count();

            // Parallel
            start = std::chrono::high_resolution_clock::now();
            hpcs_mad_parallel(data.data(), n, &scalar_result, &status);
            end = std::chrono::high_resolution_clock::now();
            double elapsed_par = std::chrono::duration<double>(end - start).count();

            double speedup = elapsed_orig / elapsed_par;

            std::cout << n << ",mad,original," << std::fixed << std::setprecision(6) << elapsed_orig << ",1.000000" << std::endl;
            std::cout << n << ",mad,parallel," << std::fixed << std::setprecision(6) << elapsed_par << "," << speedup << std::endl;
        }

        // ====================================================================
        // QUANTILE: Original vs Parallel
        // ====================================================================
        {
            double q = 0.75;

            // Original
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_quantile(data.data(), n, q, &scalar_result, HPCS_MODE_FAST, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_orig = std::chrono::duration<double>(end - start).count();

            // Parallel
            start = std::chrono::high_resolution_clock::now();
            hpcs_quantile_parallel(data.data(), n, q, &scalar_result, &status);
            end = std::chrono::high_resolution_clock::now();
            double elapsed_par = std::chrono::duration<double>(end - start).count();

            double speedup = elapsed_orig / elapsed_par;

            std::cout << n << ",quantile,original," << std::fixed << std::setprecision(6) << elapsed_orig << ",1.000000" << std::endl;
            std::cout << n << ",quantile,parallel," << std::fixed << std::setprecision(6) << elapsed_par << "," << speedup << std::endl;
        }

        // ====================================================================
        // ROBUST_ZSCORE: Original vs Parallel
        // ====================================================================
        {
            std::vector<double> zscores(n);

            // Original
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_robust_zscore(data.data(), n, zscores.data(), &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_orig = std::chrono::duration<double>(end - start).count();

            // Parallel
            start = std::chrono::high_resolution_clock::now();
            hpcs_robust_zscore_parallel(data.data(), n, zscores.data(), &status);
            end = std::chrono::high_resolution_clock::now();
            double elapsed_par = std::chrono::duration<double>(end - start).count();

            double speedup = elapsed_orig / elapsed_par;

            std::cout << n << ",robust_zscore,original," << std::fixed << std::setprecision(6) << elapsed_orig << ",1.000000" << std::endl;
            std::cout << n << ",robust_zscore,parallel," << std::fixed << std::setprecision(6) << elapsed_par << "," << speedup << std::endl;
        }

        // ====================================================================
        // ROLLING_MEDIAN: Original vs Fast (C++ heap-based)
        // ====================================================================
        // Only test up to 1M for rolling operations (original is very slow)
        if (n <= 1000000) {
            std::vector<double> rolling_out(n);

            // Original
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_rolling_median(data.data(), n, window, rolling_out.data(), HPCS_MODE_FAST, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_orig = std::chrono::duration<double>(end - start).count();

            // Fast
            start = std::chrono::high_resolution_clock::now();
            hpcs_rolling_median_fast(data.data(), n, window, rolling_out.data(), &status);
            end = std::chrono::high_resolution_clock::now();
            double elapsed_fast = std::chrono::duration<double>(end - start).count();

            double speedup = elapsed_orig / elapsed_fast;

            std::cout << n << ",rolling_median,original," << std::fixed << std::setprecision(6) << elapsed_orig << ",1.000000" << std::endl;
            std::cout << n << ",rolling_median,fast," << std::fixed << std::setprecision(6) << elapsed_fast << "," << speedup << std::endl;
        }

        // ====================================================================
        // ROLLING_MAD: Original vs Fast
        // ====================================================================
        if (n <= 1000000) {
            std::vector<double> rolling_out(n);

            // Original
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_rolling_mad(data.data(), n, window, rolling_out.data(), HPCS_MODE_FAST, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_orig = std::chrono::duration<double>(end - start).count();

            // Fast
            start = std::chrono::high_resolution_clock::now();
            hpcs_rolling_mad_fast(data.data(), n, window, rolling_out.data(), &status);
            end = std::chrono::high_resolution_clock::now();
            double elapsed_fast = std::chrono::duration<double>(end - start).count();

            double speedup = elapsed_orig / elapsed_fast;

            std::cout << n << ",rolling_mad,original," << std::fixed << std::setprecision(6) << elapsed_orig << ",1.000000" << std::endl;
            std::cout << n << ",rolling_mad,fast," << std::fixed << std::setprecision(6) << elapsed_fast << "," << speedup << std::endl;
        }
    }

    return 0;
}
