#!/bin/bash
#
# HPCSeries Core v0.4 Benchmark Runner
#
# This script runs the v0.4 parallel benchmarks with different thread counts
# and generates a comparison report showing OpenMP speedup.
#
# Usage:
#   ./scripts/run_v04_benchmarks.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HPCSeries Core v0.4 Benchmark Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create output directory
RESULTS_DIR="benchmark_results_v04"
mkdir -p "$RESULTS_DIR"

# Check if benchmark binary exists
BENCH_BIN="./build/bench_v04_parallel"
if [ ! -f "$BENCH_BIN" ]; then
    echo "Error: Benchmark binary not found: $BENCH_BIN"
    echo "Building benchmark..."
    cmake --build build --target bench_v04_parallel
fi

# Thread counts to test
THREAD_COUNTS=(1 2 4 8)

echo -e "${GREEN}Running benchmarks with different thread counts...${NC}"
echo ""

for THREADS in "${THREAD_COUNTS[@]}"; do
    OUTPUT_FILE="$RESULTS_DIR/results_${THREADS}threads.csv"
    echo -e "${BLUE}Running with OMP_NUM_THREADS=$THREADS${NC}"

    OMP_NUM_THREADS=$THREADS $BENCH_BIN > "$OUTPUT_FILE"

    # Show first few lines
    echo "Sample results:"
    head -5 "$OUTPUT_FILE"
    echo ""
done

echo -e "${GREEN}✓ Benchmarks complete!${NC}"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Files generated:"
ls -lh "$RESULTS_DIR/"
echo ""

# Generate comparison report
echo -e "${BLUE}Generating speedup analysis...${NC}"
python3 scripts/analyze_v04_benchmarks.py "$RESULTS_DIR"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark suite complete!${NC}"
echo -e "${GREEN}========================================${NC}"
