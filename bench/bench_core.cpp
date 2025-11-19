/*
 * Simple benchmark harness for HPCSeries kernels.
 *
 * This program generates synthetic data (random doubles), executes
 * several computational kernels provided by the HPCSeries C API, and
 * measures their execution time using the C++ chrono library. The
 * intent is to establish a baseline of serial performance prior to
 * adding OpenMP or other forms of parallelism. Output is printed in
 * a comma‑separated format: the size of the input (n), the kernel
 * name, and the elapsed wall time in seconds.
 */

#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

#include "../include/hpc_series.h"

int main() {
    // Problem sizes to benchmark. These correspond to 1e5, 1e6 and 1e7
    // elements. You can adjust or extend this vector to add more
    // sizes. When adding OpenMP later, it will be straightforward to
    // recompile with different flags or call alternate implementations.
    const std::vector<size_t> sizes = {100000, 1000000, 10000000};
    const size_t window = 100; // sliding window size for rolling operations

    // Random number generator seeded for reproducibility
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Header for clarity when reading output
    std::cout << "n,kernel,elapsed_seconds" << std::endl;

    for (size_t n : sizes) {
        // Allocate and initialise input data
        std::vector<double> data(n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = dist(rng);
        }

        // Output buffer sized to n (rolling kernels produce n outputs)
        std::vector<double> out(n);

        // Benchmark rolling_sum
        {
            auto start = std::chrono::high_resolution_clock::now();
            rolling_sum(data.data(), out.data(), n, window);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",rolling_sum," << std::setprecision(6)
                      << std::fixed << elapsed << std::endl;
        }

        // Benchmark rolling_mean
        {
            auto start = std::chrono::high_resolution_clock::now();
            rolling_mean(data.data(), out.data(), n, window);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",rolling_mean," << std::setprecision(6)
                      << std::fixed << elapsed << std::endl;
        }

        // Benchmark reduce_sum
        {
            auto start = std::chrono::high_resolution_clock::now();
            double sum = reduce_sum(data.data(), n);
            (void)sum; // suppress unused variable warning
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << n << ",reduce_sum," << std::setprecision(6)
                      << std::fixed << elapsed << std::endl;
        }
    }
    return 0;
}