#!/bin/bash
# Benchmark Runner with Logging
# Runs all benchmarks and saves results to logs/ directory

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
LOGS_DIR="$PROJECT_ROOT/logs"

# Detect mode (cpu or gpu)
MODE="${1:-cpu}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y%m%d)

# Export mode for benchmark executables
export HPCS_MODE="$MODE"

# Create log directories
mkdir -p "$LOGS_DIR/benchmarks/$MODE"
mkdir -p "$LOGS_DIR/tests/$MODE"
mkdir -p "$LOGS_DIR/reports"

echo "============================================================================"
echo "HPCSeries Benchmark Runner"
echo "============================================================================"
echo "Mode:        $MODE (HPCS_MODE=$HPCS_MODE)"
echo "Timestamp:   $TIMESTAMP"
echo "Build:       $BUILD_DIR"
echo "Logs:        $LOGS_DIR"
echo ""

# System information
echo "System Information:"
echo "-------------------"
echo "Hostname:    $(hostname)"
echo "OS:          $(uname -s) $(uname -r)"
echo "CPU:         $(grep -m 1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "Unknown")"
echo "CPU Cores:   $(nproc)"
echo "Memory:      $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "Unknown")"

if [ "$MODE" = "gpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Not detected")"
    echo "GPU Memory:  $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "Unknown")"
fi

echo ""

# Check build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    echo "Please run cmake and make first"
    exit 1
fi

cd "$BUILD_DIR"

# Function to run benchmark and capture output
# Note: HPCS_MODE environment variable is inherited by all benchmark executables
run_benchmark() {
    local bench_name=$1
    local bench_exec=$2
    local log_file="$LOGS_DIR/benchmarks/$MODE/${bench_name}_${MODE}_${TIMESTAMP}.csv"

    echo "-------------------------------------------------------------------"
    echo "Running: $bench_name (HPCS_MODE=$HPCS_MODE)"
    echo "Output:  $log_file"

    if [ -f "$bench_exec" ]; then
        # Add timestamp and mode columns to CSV output
        {
            echo "# Benchmark: $bench_name"
            echo "# Mode: $MODE"
            echo "# Timestamp: $(date -Iseconds)"
            echo "# Hostname: $(hostname)"
            if [ "$MODE" = "gpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
                echo "# GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
            fi
            echo "#"

            # Run benchmark and add timestamp column
            "$bench_exec" | awk -v ts="$(date -Iseconds)" '
                NR==1 { print "timestamp,mode," $0 }
                NR>1 { print ts ",'$MODE'," $0 }
            '
        } > "$log_file"

        echo "✓ Completed: $bench_name"
        echo "  Results: $log_file"

        # Show summary
        local num_results=$(tail -n +7 "$log_file" | grep -v "^#" | wc -l)
        echo "  Data points: $num_results"
    else
        echo "⚠️ Benchmark not found: $bench_exec"
    fi

    echo ""
}

echo "============================================================================"
echo "Running Benchmarks"
echo "============================================================================"
echo ""

# Run v0.3 optimized benchmarks
run_benchmark "v03_optimized" "./bench_v03_optimized"

# Run anomaly detection benchmarks
run_benchmark "anomaly_detection" "./bench_anomaly_detection"

# Run v0.3 baseline benchmarks (if exists)
if [ -f "./bench_v03" ]; then
    run_benchmark "v03_baseline" "./bench_v03"
fi

echo "============================================================================"
echo "Benchmark Summary"
echo "============================================================================"
echo ""
echo "Results saved to: $LOGS_DIR/benchmarks/$MODE/"
echo ""
ls -lh "$LOGS_DIR/benchmarks/$MODE/"*"$TIMESTAMP"* 2>/dev/null || echo "No results generated"

echo ""
echo "============================================================================"
echo "Next Steps"
echo "============================================================================"
echo ""
echo "1. View results:"
echo "   cat $LOGS_DIR/benchmarks/$MODE/*$TIMESTAMP*.csv"
echo ""
echo "2. Generate performance report:"
echo "   python3 scripts/generate_performance_report.py --date $DATE"
echo ""
echo "3. Compare with previous runs:"
echo "   python3 scripts/compare_benchmarks.py --mode $MODE --latest 2"
echo ""

exit 0
