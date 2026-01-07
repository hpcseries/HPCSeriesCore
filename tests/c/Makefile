# HPCSeries Core v0.7 - C Test Suite Makefile
# ============================================

# Compiler settings
CC = gcc
CFLAGS = -Wall -O2 -g -I../../include -fopenmp
LDFLAGS = -L../../build -lhpcs_core -lstdc++ -lgfortran -lm -fopenmp

# Test executables
TESTS = test_hpcs_robust \
        test_fortran_simd \
        test_transforms_v08 \
        test_execution_modes \
        test_reductions_modes \
        test_hpcs_gpu_infra \
        test_hpcs_gpu_kernels \
        test_hpcs_gpu_memory \
        test_cuda_device_count

# Build directory
BUILD_DIR = ../../build

.PHONY: all clean run help

all: $(TESTS)

# Build each test
%: %.c
	@echo "Building $@..."
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Run all tests
run: all
	@echo "================================"
	@echo "Running C Test Suite"
	@echo "================================"
	@for test in $(TESTS); do \
		if [ -f $$test ]; then \
			echo ""; \
			echo "Running $$test..."; \
			./$$test || echo "FAILED: $$test"; \
		fi; \
	done
	@echo ""
	@echo "================================"
	@echo "C Test Suite Complete"
	@echo "================================"

# Clean build artifacts
clean:
	rm -f $(TESTS) *.o

# Help target
help:
	@echo "HPCSeries Core C Test Suite"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all tests"
	@echo "  run     - Build and run all tests"
	@echo "  clean   - Remove test executables"
	@echo "  test_X  - Build and run specific test"
	@echo ""
	@echo "Available tests:"
	@for test in $(TESTS); do echo "  - $$test"; done
