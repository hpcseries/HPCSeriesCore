/**
 * @file bench_core.cpp
 * @brief Performance benchmarks for HPC Series Core kernels
 *
 * Measures execution time on large arrays and reports throughput
 */

#include <iostream>
#include <vector>
#include <chrono>
#include "hpcs_core.h"

using namespace std::chrono;

void benchmark_rolling_mean(size_t n, size_t window) {
    std::vector<double> input(n);
    std::vector<double> output(n);

    // Initialize with test data
    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<double>(i);
    }

    auto start = high_resolution_clock::now();
    hpcs_rolling_mean(input.data(), n, window, output.data());
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start).count();
    double throughput = (n * sizeof(double)) / (duration / 1e6) / 1e9; // GB/s

    std::cout << "Rolling Mean (" << n << " elements, window=" << window << "): "
              << duration << " us, " << throughput << " GB/s\n";
}

int main() {
    std::cout << "HPC Series Core - Benchmark Suite\n";
    std::cout << "===================================\n\n";

    // TODO: Add comprehensive benchmarks for all kernels
    // - Various array sizes (1K, 10K, 100K, 1M, 10M)
    // - Different window sizes
    // - Cache effects analysis
    // - Comparison with naive implementations

    std::cout << "Benchmarks not yet fully implemented\n";

    // Example benchmark
    benchmark_rolling_mean(1000000, 20);

    return 0;
}
