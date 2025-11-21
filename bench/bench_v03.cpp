/*
 * Benchmark harness for HPCSeries v0.3 kernels.
 *
 * This C++ program measures the runtime of the robust statistics and
 * data quality functions introduced in version 0.3 of the HPCSeries
 * Core library.  It generates random input arrays of various sizes
 * and invokes each kernel, timing the execution with
 * std::chrono.  The measured functions are:
 *
 *   - hpcs_median
 *   - hpcs_mad
 *   - hpcs_quantile (with a fixed q)
 *   - hpcs_rolling_median
 *   - hpcs_rolling_mad
 *   - hpcs_clip
 *   - hpcs_winsorize_by_quantiles
 *   - hpcs_robust_zscore
 *
 * The program prints results as CSV lines: n,kernel,elapsed_seconds.
 * Modify the parameters (window size, quantile probabilities, clip
 * bounds) as needed for your testing.  This harness assumes the
 * v0.3 Fortran kernels are available through the C ABI declared in
 * hpcs_core.h.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../include/hpcs_core.h"

int main() {
    // Array sizes to test
    const std::vector<int> sizes = {100000, 1000000, 10000000};
    const int window = 100;      // rolling window length
    const double q_test = 0.75;  // quantile probability for hpcs_quantile
    const double clip_min = 0.0, clip_max = 1.0; // clip bounds
    const double q_low = 0.05, q_high = 0.95;    // winsorisation quantiles
    // Note: robust_zscore now uses fixed default scale of 1.4826

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::cout << "n,kernel,elapsed_seconds" << std::endl;

    for (int n : sizes) {
        // Generate random data
        std::vector<double> data(n);
        for (int i = 0; i < n; ++i) {
            data[i] = dist(rng);
        }

        int status = 0;
        double scalar_result = 0.0;

        // Median
        {
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_median(data.data(), n, &scalar_result, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",median," << std::setprecision(6) << std::fixed << elapsed << std::endl;
        }

        // MAD
        {
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_mad(data.data(), n, &scalar_result, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",mad," << std::setprecision(6) << std::fixed << elapsed << std::endl;
        }

        // Quantile
        {
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_quantile(data.data(), n, q_test, &scalar_result, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",quantile," << std::setprecision(6) << std::fixed << elapsed << std::endl;
        }

        // Rolling median
        {
            std::vector<double> out(n);
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_rolling_median(data.data(), n, window, out.data(), &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",rolling_median," << std::setprecision(6) << std::fixed << elapsed << std::endl;
        }

        // Rolling MAD
        {
            std::vector<double> out(n);
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_rolling_mad(data.data(), n, window, out.data(), &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",rolling_mad," << std::setprecision(6) << std::fixed << elapsed << std::endl;
        }

        // Clip (in place) — use a copy to avoid affecting other tests
        {
            std::vector<double> tmp = data;
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_clip(tmp.data(), n, clip_min, clip_max, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",clip," << std::setprecision(6) << std::fixed << elapsed << std::endl;
        }

        // Winsorize by quantiles — in place on a copy
        {
            std::vector<double> tmp = data;
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_winsorize_by_quantiles(tmp.data(), n, q_low, q_high, &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",winsorize," << std::setprecision(6) << std::fixed << elapsed << std::endl;
        }

        // Robust z-score (uses default scale of 1.4826)
        {
            std::vector<double> z(n);
            auto start = std::chrono::high_resolution_clock::now();
            hpcs_robust_zscore(data.data(), n, z.data(), &status);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",robust_zscore," << std::setprecision(6) << std::fixed << elapsed << std::endl;
        }
    }
    return 0;
}