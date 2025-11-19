# HPC Series Core Library

High-performance computational kernels for time series analysis and statistical operations, implemented in modern Fortran with C/C++ interoperability.

## Overview

HPCSeriesCore provides optimized implementations of:
- Rolling window operations (mean, std, min, max, etc.)
- Statistical transformations (z-score, rank, etc.)
- Group-based and simple reductions
- Efficient data processing for large-scale numerical computations

## Project Structure

```
/HPCSeriesCore/
├── CMakeLists.txt         # Top-level build configuration
├── include/               # Public C headers
│   └── hpcs_core.h        # C-accessible kernel prototypes
├── src/
│   ├── fortran/           # Modern Fortran implementations
│   └── cpp/               # C++ wrappers (future)
├── tests/                 # Test suites
├── bench/                 # Performance benchmarks
├── docs/                  # Documentation
└── build/                 # Build directory (out-of-source)
```

## Building

### Prerequisites

- CMake 3.18 or higher
- Modern Fortran compiler (GFortran 9+, Intel Fortran, etc.)
- C/C++ compiler (GCC, Clang, MSVC, etc.)

### Build Instructions

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build .

# Run tests
ctest

# Install (optional)
cmake --install . --prefix /desired/install/path
```

### Build Options

- `BUILD_TESTS`: Enable/disable test suite (default: ON)
- `BUILD_BENCHMARKS`: Enable/disable benchmarks (default: ON)

```bash
cmake -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=ON ..
```

## Usage

Link against `libhpcs_core.a` and include `hpcs_core.h`:

```c
#include <hpcs_core.h>

// Example: Rolling mean
double input[1000];
double output[1000];
int window = 20;

hpcs_rolling_mean(input, 1000, window, output);
```

## Documentation

See [docs/](docs/) for:
- Kernel specifications
- API reference
- Design notes
- Performance characteristics

## Version

Current version: **0.1.0**

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
