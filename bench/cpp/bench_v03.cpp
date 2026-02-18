/*
 * Benchmark harness for HPCSeries v0.3/v0.4 kernels with GPU support.
 *
 * This C++ program measures the runtime of robust statistics and
 * data quality functions, supporting both CPU and GPU execution modes.
 *
 * Usage:
 *   ./bench_v03               # CPU mode (default)
 *   HPCS_MODE=gpu ./bench_v03 # GPU mode
 *   HPCS_MODE=auto ./bench_v03 # Auto-select best
 *
 * The program prints results as CSV lines: n,kernel,mode,elapsed_seconds
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../include/hpcs_core.h"

using namespace std;
using namespace std::chrono;

// Execution mode
enum class Mode { CPU, GPU, AUTO };

Mode get_mode() {
    const char* env = getenv("HPCS_MODE");
    if (!env) return Mode::CPU;

    string mode_str(env);
    if (mode_str == "gpu") return Mode::GPU;
    if (mode_str == "auto") return Mode::AUTO;
    return Mode::CPU;
}

const char* mode_str(Mode m) {
    switch (m) {
        case Mode::CPU: return "cpu";
        case Mode::GPU: return "gpu";
        case Mode::AUTO: return "auto";
    }
    return "unknown";
}

// Benchmark configuration
struct BenchConfig {
    int warmup_iters = 2;
    int bench_iters = 5;
    bool show_device_time = true;
};

//============================================================================
// CPU Benchmark Functions
//============================================================================

double bench_cpu_median(const vector<double>& data, const BenchConfig& cfg) {
    int n = data.size();
    double result;
    int status;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        hpcs_median(data.data(), n, &result, HPCS_MODE_FAST, &status);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
        hpcs_median(data.data(), n, &result, HPCS_MODE_FAST, &status);
    }
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count() / cfg.bench_iters;
}

double bench_cpu_mad(const vector<double>& data, const BenchConfig& cfg) {
    int n = data.size();
    double result;
    int status;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        hpcs_mad(data.data(), n, &result, HPCS_MODE_FAST, &status);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
        hpcs_mad(data.data(), n, &result, HPCS_MODE_FAST, &status);
    }
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count() / cfg.bench_iters;
}

double bench_cpu_quantile(const vector<double>& data, double q, const BenchConfig& cfg) {
    int n = data.size();
    double result;
    int status;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        hpcs_quantile(data.data(), n, q, &result, HPCS_MODE_FAST, &status);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
        hpcs_quantile(data.data(), n, q, &result, HPCS_MODE_FAST, &status);
    }
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count() / cfg.bench_iters;
}

double bench_cpu_rolling_median(const vector<double>& data, int window, const BenchConfig& cfg) {
    int n = data.size();
    vector<double> out(n);
    int status;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        hpcs_rolling_median(data.data(), n, window, out.data(), HPCS_MODE_FAST, &status);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
        hpcs_rolling_median(data.data(), n, window, out.data(), HPCS_MODE_FAST, &status);
    }
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count() / cfg.bench_iters;
}

double bench_cpu_rolling_mad(const vector<double>& data, int window, const BenchConfig& cfg) {
    int n = data.size();
    vector<double> out(n);
    int status;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        hpcs_rolling_mad(data.data(), n, window, out.data(), HPCS_MODE_FAST, &status);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
        hpcs_rolling_mad(data.data(), n, window, out.data(), HPCS_MODE_FAST, &status);
    }
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count() / cfg.bench_iters;
}

//============================================================================
// GPU Benchmark Functions
//============================================================================

double bench_gpu_median(const vector<double>& data, const BenchConfig& cfg) {
    int n = data.size();
    double result;
    int status;
    void* device_ptr = nullptr;

    // Initialize GPU
    hpcs_accel_init(&status);
    if (status != 0) {
        cerr << "GPU init failed!" << endl;
        return -1.0;
    }

    // Copy to device once
    hpcs_accel_copy_to_device(data.data(), n, &device_ptr, &status);
    if (status != 0) {
        cerr << "Copy to device failed!" << endl;
        return -1.0;
    }

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        hpcs_accel_median(device_ptr, n, &result, &status);
    }

    // Benchmark (kernel time only)
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
        hpcs_accel_median(device_ptr, n, &result, &status);
    }
    auto end = high_resolution_clock::now();

    double kernel_time = duration<double>(end - start).count() / cfg.bench_iters;

    // Cleanup
    hpcs_accel_free_device(device_ptr, &status);

    return kernel_time;
}

double bench_gpu_mad(const vector<double>& data, const BenchConfig& cfg) {
    int n = data.size();
    double result;
    int status;
    void* device_ptr = nullptr;

    hpcs_accel_init(&status);
    if (status != 0) return -1.0;

    hpcs_accel_copy_to_device(data.data(), n, &device_ptr, &status);
    if (status != 0) return -1.0;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        hpcs_accel_mad(device_ptr, n, &result, &status);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
        hpcs_accel_mad(device_ptr, n, &result, &status);
    }
    auto end = high_resolution_clock::now();

    hpcs_accel_free_device(device_ptr, &status);

    return duration<double>(end - start).count() / cfg.bench_iters;
}

double bench_gpu_rolling_median(const vector<double>& data, int window, const BenchConfig& cfg) {
    int n = data.size();
    int status;
    void* device_ptr = nullptr;
    void* output_ptr = nullptr;
    vector<double> result(n);

    hpcs_accel_init(&status);
    if (status != 0) return -1.0;

    hpcs_accel_copy_to_device(data.data(), n, &device_ptr, &status);
    if (status != 0) return -1.0;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
        void* tmp_out = nullptr;
        hpcs_accel_rolling_median(device_ptr, n, window, &tmp_out, &status);
        if (tmp_out) hpcs_accel_free_device(tmp_out, &status);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
        hpcs_accel_rolling_median(device_ptr, n, window, &output_ptr, &status);
        if (i < cfg.bench_iters - 1 && output_ptr) {
            hpcs_accel_free_device(output_ptr, &status);
        }
    }
    auto end = high_resolution_clock::now();

    // Copy result back
    if (output_ptr) {
        hpcs_accel_copy_from_device(output_ptr, n, result.data(), &status);
        hpcs_accel_free_device(output_ptr, &status);
    }

    hpcs_accel_free_device(device_ptr, &status);

    return duration<double>(end - start).count() / cfg.bench_iters;
}

//============================================================================
// Main Benchmark
//============================================================================

int main() {
    Mode mode = get_mode();
    BenchConfig cfg;

    // Array sizes to test (matching v0.3 analysis)
    const vector<int> sizes = {100000, 1000000, 5000000};
    const int window = 200;      // rolling window (v0.3 used 200)
    const double q_test = 0.75;  // quantile probability

    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(0.0, 1.0);

    cout << "# HPCSeries v0.4 Benchmark" << endl;
    cout << "# Mode: " << mode_str(mode) << endl;
    cout << "# Warmup iterations: " << cfg.warmup_iters << endl;
    cout << "# Benchmark iterations: " << cfg.bench_iters << endl;
    cout << endl;
    cout << "n,kernel,mode,elapsed_seconds" << endl;

    for (int n : sizes) {
        // Generate random data
        vector<double> data(n);
        for (int i = 0; i < n; ++i) {
            data[i] = dist(rng);
        }

        // Median
        if (mode == Mode::CPU) {
            double elapsed = bench_cpu_median(data, cfg);
            cout << n << ",median,cpu," << setprecision(6) << fixed << elapsed << endl;
        } else if (mode == Mode::GPU) {
            double elapsed = bench_gpu_median(data, cfg);
            if (elapsed > 0) {
                cout << n << ",median,gpu," << setprecision(6) << fixed << elapsed << endl;
            }
        }

        // MAD
        if (mode == Mode::CPU) {
            double elapsed = bench_cpu_mad(data, cfg);
            cout << n << ",mad,cpu," << setprecision(6) << fixed << elapsed << endl;
        } else if (mode == Mode::GPU) {
            double elapsed = bench_gpu_mad(data, cfg);
            if (elapsed > 0) {
                cout << n << ",mad,gpu," << setprecision(6) << fixed << elapsed << endl;
            }
        }

        // Quantile (CPU only for now)
        if (mode == Mode::CPU) {
            double elapsed = bench_cpu_quantile(data, q_test, cfg);
            cout << n << ",quantile,cpu," << setprecision(6) << fixed << elapsed << endl;
        }

        // Rolling Median (use smaller sizes - it's expensive)
        if (n <= 1000000) {
            if (mode == Mode::CPU) {
                double elapsed = bench_cpu_rolling_median(data, window, cfg);
                cout << n << ",rolling_median,cpu," << setprecision(6) << fixed << elapsed << endl;
            } else if (mode == Mode::GPU) {
                double elapsed = bench_gpu_rolling_median(data, window, cfg);
                if (elapsed > 0) {
                    cout << n << ",rolling_median,gpu," << setprecision(6) << fixed << elapsed << endl;
                }
            }
        }

        // Rolling MAD (CPU only for now, skip for large sizes)
        if (mode == Mode::CPU && n <= 1000000) {
            double elapsed = bench_cpu_rolling_mad(data, window, cfg);
            cout << n << ",rolling_mad,cpu," << setprecision(6) << fixed << elapsed << endl;
        }
    }

    return 0;
}
