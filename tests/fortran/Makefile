# HPCSeries Core v0.7 - Fortran Test Suite Makefile
# ===================================================

# Compiler settings
FC = gfortran
FFLAGS = -Wall -O2 -g -fopenmp -I../../build/fortran_modules
LDFLAGS = -L../../build -lhpcs_core -lm -fopenmp

# Test executables
TESTS = test_fortran_basic

# Build directory
BUILD_DIR = ../../build

.PHONY: all clean run help

all: $(TESTS)

# Build each test
test_fortran_basic: test_fortran_basic.f90
	@echo "Building $@..."
	$(FC) $(FFLAGS) $< -o $@ $(LDFLAGS)

# Run all tests
run: all
	@echo "================================"
	@echo "Running Fortran Test Suite"
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
	@echo "Fortran Test Suite Complete"
	@echo "================================"

# Run a specific test
test_%: test_%
	@echo "Running $@..."
	./$@

# Clean build artifacts
clean:
	rm -f $(TESTS) *.o *.mod

# Help target
help:
	@echo "HPCSeries Core Fortran Test Suite"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all tests"
	@echo "  run     - Build and run all tests"
	@echo "  clean   - Remove test executables"
	@echo "  test_X  - Build and run specific test"
	@echo ""
	@echo "Available tests:"
	@for test in $(TESTS); do echo "  - $$test"; done
