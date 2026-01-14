/*
 * GPU Acceleration Benchmark: CPU vs GPU Performance Comparison
 *
 * This benchmark compares CPU and GPU implementations of key kernels:
 * - median (high priority - 18x bottleneck)
 * - MAD (high priority - robust detection)
 * - rolling_median (highest priority - 60x bottleneck)
 * - reduce_sum (validation baseline)
 * - prefix_sum (scan operation)
 *
 * Output: CSV with columns:
 *   kernel,n,window,mode,elapsed_seconds,speedup
 *
 * Usage:
 *   CPU mode:  HPCS_MODE=cpu  ./bench_gpu_acceleration
 *   GPU mode:  HPCS_MODE=gpu  ./bench_gpu_acceleration
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../include/hpcs_core.h"

using namespace std;
using namespace std::chrono;

// Determine mode from environment variable
enum Mode { CPU, GPU };

Mode get_mode() {
    const char* mode_env = getenv("HPCS_MODE");
    if (mode_env && strcmp(mode_env, "gpu") == 0) {
        return GPU;
    }
    return CPU;
}

string mode_str(Mode mode) {
    return (mode == GPU) ? "gpu" : "cpu";
}

// Generate test data
vector<double> generate_data(int n, int seed = 42) {
    vector<double> data(n);
    mt19937_64 rng(seed);
    uniform_real_distribution<double> dist(0.0, 100.0);

    for (int i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    return data;
}

// Benchmark median
void bench_median(Mode mode, const vector<int>& sizes) {
    for (int n : sizes) {
        auto data = generate_data(n);
        double median_val;
        int status;

        auto start = high_resolution_clock::now();

        if (mode == GPU) {
            // GPU path: copy to device, run GPU kernel, copy back
            void *device_ptr = nullptr;

            hpcs_accel_copy_to_device(data.data(), n, &device_ptr, &status);
            if (status != HPCS_SUCCESS) {
                cerr << "Error: copy_to_device failed" << endl;
                continue;
            }

            hpcs_accel_median(device_ptr, n, &median_val, &status);
            if (status != HPCS_SUCCESS) {
                cerr << "Error: accel_median failed" << endl;
                continue;
            }

            hpcs_accel_free_device(device_ptr, &status);
        } else {
            // CPU path: use standard median
            hpcs_median(data.data(), n, &median_val, HPCS_MODE_FAST, &status);
        }

        auto end = high_resolution_clock::now();
        double elapsed = duration<double>(end - start).count();

        cout << "median," << n << ",N/A," << mode_str(mode) << ","
             << fixed << setprecision(6) << elapsed << ",N/A" << endl;
    }
}

// Benchmark MAD
void bench_mad(Mode mode, const vector<int>& sizes) {
    for (int n : sizes) {
        auto data = generate_data(n);
        double mad_val;
        int status;

        auto start = high_resolution_clock::now();

        if (mode == GPU) {
            void *device_ptr = nullptr;

            hpcs_accel_copy_to_device(data.data(), n, &device_ptr, &status);
            if (status != HPCS_SUCCESS) continue;

            hpcs_accel_mad(device_ptr, n, &mad_val, &status);
            if (status != HPCS_SUCCESS) continue;

            hpcs_accel_free_device(device_ptr, &status);
        } else {
            hpcs_mad(data.data(), n, &mad_val, HPCS_MODE_FAST, &status);
        }

        auto end = high_resolution_clock::now();
        double elapsed = duration<double>(end - start).count();

        cout << "mad," << n << ",N/A," << mode_str(mode) << ","
             << fixed << setprecision(6) << elapsed << ",N/A" << endl;
    }
}

// Benchmark rolling median
void bench_rolling_median(Mode mode, const vector<int>& sizes, const vector<int>& windows) {
    for (int n : sizes) {
        auto data = generate_data(n);

        for (int window : windows) {
            vector<double> output(n);
            int status;

            auto start = high_resolution_clock::now();

            if (mode == GPU) {
                void *device_ptr = nullptr;
                void *device_output = nullptr;

                hpcs_accel_copy_to_device(data.data(), n, &device_ptr, &status);
                if (status != HPCS_SUCCESS) continue;

                hpcs_accel_rolling_median(device_ptr, n, window, &device_output, &status);
                if (status != HPCS_SUCCESS) {
                    hpcs_accel_free_device(device_ptr, &status);
                    continue;
                }

                hpcs_accel_copy_from_device(device_output, n, output.data(), &status);

                hpcs_accel_free_device(device_ptr, &status);
                // Note: device_output is managed by hpcs_accel_rolling_median
            } else {
                hpcs_rolling_median(data.data(), n, window, output.data(), HPCS_MODE_FAST, &status);
            }

            auto end = high_resolution_clock::now();
            double elapsed = duration<double>(end - start).count();

            cout << "rolling_median," << n << "," << window << "," << mode_str(mode) << ","
                 << fixed << setprecision(6) << elapsed << ",N/A" << endl;
        }
    }
}

// Benchmark reduce sum
void bench_reduce_sum(Mode mode, const vector<int>& sizes) {
    for (int n : sizes) {
        auto data = generate_data(n);
        double sum_val;
        int status;

        auto start = high_resolution_clock::now();

        if (mode == GPU) {
            void *device_ptr = nullptr;

            hpcs_accel_copy_to_device(data.data(), n, &device_ptr, &status);
            if (status != HPCS_SUCCESS) continue;

            hpcs_accel_reduce_sum(device_ptr, n, &sum_val, &status);
            if (status != HPCS_SUCCESS) continue;

            hpcs_accel_free_device(device_ptr, &status);
        } else {
            hpcs_reduce_sum(data.data(), n, &sum_val, HPCS_MODE_FAST, &status);
        }

        auto end = high_resolution_clock::now();
        double elapsed = duration<double>(end - start).count();

        cout << "reduce_sum," << n << ",N/A," << mode_str(mode) << ","
             << fixed << setprecision(6) << elapsed << ",N/A" << endl;
    }
}

// Benchmark prefix sum
void bench_prefix_sum(Mode mode, const vector<int>& sizes) {
    for (int n : sizes) {
        auto data = generate_data(n);
        vector<double> output(n);
        int status;

        auto start = high_resolution_clock::now();

        if (mode == GPU) {
            void *device_input = nullptr;
            void *device_output = nullptr;

            hpcs_accel_copy_to_device(data.data(), n, &device_input, &status);
            if (status != HPCS_SUCCESS) continue;

            hpcs_accel_copy_to_device(output.data(), n, &device_output, &status);
            if (status != HPCS_SUCCESS) {
                hpcs_accel_free_device(device_input, &status);
                continue;
            }

            hpcs_accel_prefix_sum(device_input, n, device_output, &status);
            if (status != HPCS_SUCCESS) {
                hpcs_accel_free_device(device_input, &status);
                hpcs_accel_free_device(device_output, &status);
                continue;
            }

            hpcs_accel_copy_from_device(device_output, n, output.data(), &status);

            hpcs_accel_free_device(device_input, &status);
            hpcs_accel_free_device(device_output, &status);
        } else {
            // CPU path - use existing prefix_sum implementation
            double running_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                running_sum += data[i];
                output[i] = running_sum;
            }
            status = HPCS_SUCCESS;
        }

        auto end = high_resolution_clock::now();
        double elapsed = duration<double>(end - start).count();

        cout << "prefix_sum," << n << ",N/A," << mode_str(mode) << ","
             << fixed << setprecision(6) << elapsed << ",N/A" << endl;
    }
}

int main(int argc, char** argv) {
    Mode mode = get_mode();

    // Output banner to stderr so it doesn't interfere with CSV output
    cerr << "========================================" << endl;
    cerr << "GPU Acceleration Benchmark" << endl;
    cerr << "Mode: " << mode_str(mode) << endl;
    cerr << "========================================" << endl;
    cerr << endl;

    // Initialize GPU backend if in GPU mode
    if (mode == GPU) {
        int status;
        hpcs_accel_init(&status);
        if (status != HPCS_SUCCESS) {
            cerr << "Error: GPU backend initialization failed" << endl;
            return 1;
        }

        int count;
        hpcs_get_device_count(&count, &status);
        cerr << "GPU devices detected: " << count << endl;
        if (count == 0) {
            cerr << "Warning: No GPU devices available, results may reflect CPU fallback" << endl;
        }
        cerr << endl;
    }

    // Test sizes (smaller for rolling operations)
    vector<int> sizes = {100000, 500000, 1000000, 5000000};
    vector<int> rolling_sizes = {100000, 500000, 1000000};  // Skip 5M for rolling
    vector<int> windows = {50, 100, 200};

    // CSV header (stdout for CSV processing)
    cout << "kernel,n,window,mode,elapsed_seconds,speedup" << endl;

    // Run benchmarks (status messages to stderr)
    cerr << "# Running reduce_sum benchmark..." << endl;
    bench_reduce_sum(mode, sizes);

    cerr << "# Running median benchmark..." << endl;
    bench_median(mode, sizes);

    cerr << "# Running MAD benchmark..." << endl;
    bench_mad(mode, sizes);

    cerr << "# Running prefix_sum benchmark..." << endl;
    bench_prefix_sum(mode, sizes);

    cerr << "# Running rolling_median benchmark..." << endl;
    bench_rolling_median(mode, rolling_sizes, windows);

    // End banner to stderr
    cerr << endl;
    cerr << "========================================" << endl;
    cerr << "Benchmark Complete" << endl;
    cerr << "========================================" << endl;

    return 0;
}
