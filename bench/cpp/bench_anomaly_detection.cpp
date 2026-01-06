/*
 * Benchmark: Anomaly Detection Methods
 *
 * Compares performance of:
 * - Classical anomaly detection (mean/std)
 * - Robust anomaly detection (median/MAD)
 * - Iterative outlier removal
 * - Rolling anomaly detection
 *
 * Tests across different:
 * - Array sizes (100K, 500K, 1M, 5M)
 * - Contamination levels (0%, 5%, 10%)
 * - Window sizes (for rolling detection)
 */

#include "hpcs_core.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// Generate normal data with optional outliers
vector<double> generate_data(int n, double contamination_rate, int seed = 42) {
    vector<double> data(n);
    mt19937 gen(seed);
    normal_distribution<double> normal(100.0, 15.0);
    uniform_real_distribution<double> uniform(0.0, 1.0);
    uniform_real_distribution<double> outlier(500.0, 1000.0);

    for (int i = 0; i < n; i++) {
        if (uniform(gen) < contamination_rate) {
            data[i] = outlier(gen);  // Inject outlier
        } else {
            data[i] = normal(gen);   // Normal value
        }
    }

    return data;
}

// Benchmark classical anomaly detection
double bench_classical_detection(const vector<double>& data, double threshold) {
    int n = data.size();
    vector<int> anomaly(n);
    int status;

    auto start = high_resolution_clock::now();
    hpcs_detect_anomalies(data.data(), n, threshold, anomaly.data(), &status);
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count();
}

// Benchmark robust anomaly detection
double bench_robust_detection(const vector<double>& data, double threshold) {
    int n = data.size();
    vector<int> anomaly(n);
    int status;

    auto start = high_resolution_clock::now();
    hpcs_detect_anomalies_robust(data.data(), n, threshold, anomaly.data(), &status);
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count();
}

// Benchmark iterative outlier removal
double bench_iterative_removal(const vector<double>& data, double threshold, int max_iter,
                                int* n_clean_out, int* iterations_out) {
    int n = data.size();
    vector<double> cleaned(n);
    int n_clean, iterations, status;

    auto start = high_resolution_clock::now();
    hpcs_remove_outliers_iterative(data.data(), n, threshold, max_iter,
                                    cleaned.data(), &n_clean, &iterations, &status);
    auto end = high_resolution_clock::now();

    if (n_clean_out) *n_clean_out = n_clean;
    if (iterations_out) *iterations_out = iterations;

    return duration<double>(end - start).count();
}

// Benchmark rolling anomaly detection
double bench_rolling_detection(const vector<double>& data, int window, double threshold) {
    int n = data.size();
    vector<int> anomaly(n);
    int status;

    auto start = high_resolution_clock::now();
    hpcs_rolling_anomalies(data.data(), n, window, threshold, anomaly.data(), &status);
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count();
}

// Count actual outliers detected
int count_detections(const vector<double>& data, double threshold, bool use_robust) {
    int n = data.size();
    vector<int> anomaly(n);
    int status;

    if (use_robust) {
        hpcs_detect_anomalies_robust(data.data(), n, threshold, anomaly.data(), &status);
    } else {
        hpcs_detect_anomalies(data.data(), n, threshold, anomaly.data(), &status);
    }

    int count = 0;
    for (int i = 0; i < n; i++) {
        count += anomaly[i];
    }
    return count;
}

void print_header() {
    cout << "method,array_size,contamination,window,threshold,time_sec,speedup,detections,iterations,n_clean" << endl;
}

void run_detection_benchmarks() {
    vector<int> sizes = {100000, 500000, 1000000, 5000000};
    vector<double> contaminations = {0.0, 0.05, 0.10};
    double threshold = 3.0;

    for (int n : sizes) {
        for (double contam : contaminations) {
            auto data = generate_data(n, contam, 42);

            // Classical detection
            double time_classical = bench_classical_detection(data, threshold);
            int detections_classical = count_detections(data, threshold, false);

            // Robust detection
            double time_robust = bench_robust_detection(data, threshold);
            int detections_robust = count_detections(data, threshold, true);

            double speedup = time_classical / time_robust;

            // Output classical
            cout << "classical," << n << "," << (contam * 100) << ",N/A,"
                 << threshold << "," << time_classical << ",1.00,"
                 << detections_classical << ",N/A,N/A" << endl;

            // Output robust
            cout << "robust," << n << "," << (contam * 100) << ",N/A,"
                 << threshold << "," << time_robust << "," << speedup << ","
                 << detections_robust << ",N/A,N/A" << endl;
        }
    }
}

void run_iterative_benchmarks() {
    vector<int> sizes = {100000, 500000, 1000000};
    vector<double> contaminations = {0.05, 0.10, 0.20};
    double threshold = 3.0;
    int max_iter = 10;

    for (int n : sizes) {
        for (double contam : contaminations) {
            auto data = generate_data(n, contam, 42);

            int n_clean, iterations;
            double time_iter = bench_iterative_removal(data, threshold, max_iter,
                                                       &n_clean, &iterations);

            cout << "iterative," << n << "," << (contam * 100) << ",N/A,"
                 << threshold << "," << time_iter << ",N/A,N/A,"
                 << iterations << "," << n_clean << endl;
        }
    }
}

void run_rolling_benchmarks() {
    vector<int> sizes = {100000, 500000, 1000000};
    vector<int> windows = {50, 100, 200};
    double contamination = 0.05;
    double threshold = 3.0;

    for (int n : sizes) {
        auto data = generate_data(n, contamination, 42);

        for (int window : windows) {
            double time_rolling = bench_rolling_detection(data, window, threshold);

            // Count detections
            vector<int> anomaly(n);
            int status;
            hpcs_rolling_anomalies(data.data(), n, window, threshold, anomaly.data(), &status);
            int detections = 0;
            for (int i = 0; i < n; i++) detections += anomaly[i];

            cout << "rolling," << n << "," << (contamination * 100) << ","
                 << window << "," << threshold << "," << time_rolling << ",N/A,"
                 << detections << ",N/A,N/A" << endl;
        }
    }
}

void run_scalability_test() {
    vector<int> sizes = {10000, 50000, 100000, 500000, 1000000, 5000000};
    double contamination = 0.05;
    double threshold = 3.0;

    cout << "\n=== Scalability Test ===" << endl;
    cout << "array_size,classical_ms,robust_ms,robust_vs_classical" << endl;

    for (int n : sizes) {
        auto data = generate_data(n, contamination, 42);

        double time_classical = bench_classical_detection(data, threshold);
        double time_robust = bench_robust_detection(data, threshold);

        cout << n << ","
             << (time_classical * 1000) << ","
             << (time_robust * 1000) << ","
             << (time_robust / time_classical) << "x" << endl;
    }
}

void run_window_size_analysis() {
    int n = 1000000;
    vector<int> windows = {10, 20, 50, 100, 200, 500, 1000};
    double contamination = 0.05;
    double threshold = 3.0;

    auto data = generate_data(n, contamination, 42);

    cout << "\n=== Window Size Analysis (n=1M) ===" << endl;
    cout << "window_size,time_ms,time_per_element_ns" << endl;

    for (int window : windows) {
        double time_rolling = bench_rolling_detection(data, window, threshold);
        double ns_per_element = (time_rolling * 1e9) / n;

        cout << window << ","
             << (time_rolling * 1000) << ","
             << ns_per_element << endl;
    }
}

void run_accuracy_comparison() {
    int n = 100000;
    vector<double> contaminations = {0.01, 0.05, 0.10, 0.20};
    double threshold = 3.0;

    cout << "\n=== Accuracy Comparison ===" << endl;
    cout << "contamination,injected_outliers,classical_detected,robust_detected,classical_recall,robust_recall" << endl;

    for (double contam : contaminations) {
        auto data = generate_data(n, contam, 42);
        int injected = static_cast<int>(n * contam);

        int detected_classical = count_detections(data, threshold, false);
        int detected_robust = count_detections(data, threshold, true);

        double recall_classical = (injected > 0) ? (double)detected_classical / injected : 0.0;
        double recall_robust = (injected > 0) ? (double)detected_robust / injected : 0.0;

        cout << (contam * 100) << "%,"
             << injected << ","
             << detected_classical << ","
             << detected_robust << ","
             << (recall_classical * 100) << "%,"
             << (recall_robust * 100) << "%" << endl;
    }
}

int main(int argc, char** argv) {
    cout << fixed << setprecision(6);

    cout << "========================================" << endl;
    cout << "Anomaly Detection Benchmark Suite" << endl;
    cout << "========================================" << endl;
    cout << endl;

    // Main benchmark: Detection methods
    cout << "=== Detection Methods Benchmark ===" << endl;
    print_header();
    run_detection_benchmarks();
    cout << endl;

    // Iterative removal benchmark
    cout << "=== Iterative Removal Benchmark ===" << endl;
    print_header();
    run_iterative_benchmarks();
    cout << endl;

    // Rolling detection benchmark
    cout << "=== Rolling Detection Benchmark ===" << endl;
    print_header();
    run_rolling_benchmarks();
    cout << endl;

    // Scalability analysis
    run_scalability_test();
    cout << endl;

    // Window size analysis
    run_window_size_analysis();
    cout << endl;

    // Accuracy comparison
    run_accuracy_comparison();
    cout << endl;

    cout << "========================================" << endl;
    cout << "Benchmark Complete" << endl;
    cout << "========================================" << endl;

    return 0;
}
