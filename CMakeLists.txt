cmake_minimum_required(VERSION 3.18)

project(HPCSeriesCore
    VERSION 0.4.0
    DESCRIPTION "High-Performance Computing Series Core Library"
    LANGUAGES C CXX Fortran
)

# Find OpenMP (required for parallel kernels)
find_package(OpenMP REQUIRED)

# ============================================================================
# v0.4 GPU Acceleration Options (Phase 1)
# ============================================================================
# Optional GPU acceleration support via portable backends.
# GPU use is transparent and never breaks CPU-only workflows.
#
# Build Options:
#   -DHPCS_ENABLE_GPU_OPENMP=ON   : Enable OpenMP target offloading (default: OFF)
#   -DHPCS_ENABLE_GPU_CUDA=ON     : Enable CUDA backend (default: OFF)
#   -DHPCS_ENABLE_GPU_HIP=ON      : Enable HIP/ROCm backend (default: OFF)
#
# Note: Currently only OpenMP target is implemented. CUDA/HIP support
#       requires additional runtime bindings (future phases).
#
# Example: cmake -DHPCS_ENABLE_GPU_OPENMP=ON ..
# ============================================================================

option(HPCS_ENABLE_GPU_OPENMP "Enable GPU acceleration via OpenMP target offloading" OFF)
option(HPCS_ENABLE_GPU_CUDA "Enable GPU acceleration via CUDA (not yet implemented)" OFF)
option(HPCS_ENABLE_GPU_HIP "Enable GPU acceleration via HIP/ROCm (not yet implemented)" OFF)

# Configure GPU backend flags
if(HPCS_ENABLE_GPU_OPENMP)
    message(STATUS "HPCSeries: GPU acceleration enabled (OpenMP target offloading)")
    add_compile_definitions(HPCS_USE_OPENMP_TARGET)
    # Add OpenMP target offload flags for supported compilers
    if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -foffload=nvptx-none")
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fiopenmp -fopenmp-targets=spir64")
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mfree -mp=gpu -gpu=cc75")
    endif()
elseif(HPCS_ENABLE_GPU_CUDA)
    message(STATUS "HPCSeries: GPU acceleration enabled (CUDA backend via NVHPC)")
    add_compile_definitions(HPCS_USE_CUDA)
    # Configure CUDA Fortran flags when using NVHPC/PGI
    if(CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
        # Enable free-form, preprocessing, and CUDA GPU code for cc75 (T4)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mfree -Mpreprocess -cuda -gpu=cc75")
    endif()
elseif(HPCS_ENABLE_GPU_HIP)
    message(WARNING "HPCSeries: HIP/ROCm backend not yet implemented (Phase 1). Falling back to CPU-only.")
    # Future: add HIP runtime bindings
else()
    message(STATUS "HPCSeries: CPU-only build (GPU acceleration disabled)")
endif()

# Set C++ and C standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags - aggressive optimization for performance + OpenMP
# Note: Removed -ffast-math for Fortran to preserve IEEE 754 NaN handling for data utilities
# Note: -cpp flag enables preprocessing for #ifdef/#else/#endif directives in .f90 files
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -O3 -march=native ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS_DEBUG "-cpp -g -Wall -Wextra -pedantic -fbounds-check ${OpenMP_Fortran_FLAGS}")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp -O3 -xHost -fp-model fast -ipo ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS_DEBUG "-fpp -g -warn all -check all ${OpenMP_Fortran_FLAGS}")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mfree -Mpreprocess -O3 -fast ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS_DEBUG "-Mfree -Mpreprocess -g -Mbounds -Mchkptr -Mchkstk ${OpenMP_Fortran_FLAGS}")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fno-math-errno -Wall -Wextra ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -ffast-math -fno-math-errno -Wall -Wextra ${OpenMP_C_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast /W4 ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /fp:fast /W4 ${OpenMP_C_FLAGS}")
endif()

# Core library target
add_library(hpcs_core STATIC
    # v0.1/v0.2 modules (maintained for backward compatibility)
    src/fortran/hpcs_constants.f90
    src/fortran/hpcs_core_1d.f90
    src/fortran/hpcs_core_reductions.f90
    src/fortran/hpcs_core_utils.f90
    src/fortran/hpcs_core_prefix.f90
    src/fortran/hpcs_core_parallel.f90
    # v0.3 modules (robust statistics & data quality)
    src/fortran/hpcs_core_stats.f90
    src/fortran/hpcs_core_rolling.f90
    src/fortran/hpcs_core_quality.f90
    # v0.3 parallel modules (OpenMP optimized)
    src/fortran/hpcs_core_stats_parallel.f90
    src/fortran/hpcs_core_quality_parallel.f90
    # v0.3 fast rolling operations (C++ heap-based)
    src/hpcs_rolling_fast.cpp
    # v0.4 GPU acceleration infrastructure (Phase 1)
    src/fortran/hpcs_core_accel.f90
)

target_include_directories(hpcs_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenMP and C++ stdlib to the library
target_link_libraries(hpcs_core PUBLIC OpenMP::OpenMP_Fortran)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(hpcs_core PUBLIC stdc++)
endif()

# Installation rules
install(TARGETS hpcs_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES include/hpcs_core.h
    DESTINATION include
)

# ============================================================================
# Test Suite Configuration
# ============================================================================
# Organized test suite for CPU baseline validation and GPU acceleration testing.
#
# Test Hierarchy:
#   1. CPU Baseline Tests (validate core functionality)
#      - test_hpcs_baseline   : Comprehensive C++ tests for v0.1/v0.2 kernels
#      - test_hpcs_robust     : Robust statistics tests (v0.3)
#
#   2. GPU Acceleration Tests (validate GPU layers - Phase 3B/4)
#      - test_hpcs_gpu_infra  : GPU infrastructure and backend initialization
#      - test_hpcs_gpu_kernels: GPU statistical kernels (median, MAD, rolling, etc.)
#      - test_hpcs_gpu_memory : GPU memory management (allocation/transfer/deallocation)
#
# Recommended Test Order for GPU Deployment:
#   1. test_hpcs_baseline      → Validate CPU baseline
#   2. test_hpcs_robust        → Validate robust statistics
#   3. test_hpcs_gpu_infra     → Validate GPU backend
#   4. test_hpcs_gpu_memory    → Validate memory management
#   5. test_hpcs_gpu_kernels   → Validate GPU kernels
# ============================================================================

option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS)
    enable_testing()

    # ========================================================================
    # CPU Baseline Tests (2106 assertions total)
    # ========================================================================

    # Baseline functionality tests (v0.1/v0.2 kernels)
    add_executable(test_hpcs_baseline tests/test_hpcs_baseline.cpp)
    target_link_libraries(test_hpcs_baseline hpcs_core)
    add_test(NAME test_hpcs_baseline COMMAND test_hpcs_baseline)

    # Robust statistics and anomaly detection tests (v0.3+)
    add_executable(test_hpcs_robust tests/test_hpcs_robust.c)
    target_link_libraries(test_hpcs_robust hpcs_core)
    add_test(NAME test_hpcs_robust COMMAND test_hpcs_robust)

    # ========================================================================
    # GPU Acceleration Tests (2106 assertions total)
    # ========================================================================

    # GPU infrastructure and backend tests
    add_executable(test_hpcs_gpu_infra tests/test_hpcs_gpu_infra.c)
    target_link_libraries(test_hpcs_gpu_infra hpcs_core)
    add_test(NAME test_hpcs_gpu_infra COMMAND test_hpcs_gpu_infra)

    # GPU kernel implementation tests (Phase 3B optimized)
    add_executable(test_hpcs_gpu_kernels tests/test_hpcs_gpu_kernels.c)
    target_link_libraries(test_hpcs_gpu_kernels hpcs_core)
    add_test(NAME test_hpcs_gpu_kernels COMMAND test_hpcs_gpu_kernels)

    # GPU memory management tests (Phase 4A)
    add_executable(test_hpcs_gpu_memory tests/test_hpcs_gpu_memory.c)
    target_link_libraries(test_hpcs_gpu_memory hpcs_core)
    add_test(NAME test_hpcs_gpu_memory COMMAND test_hpcs_gpu_memory)
endif()

# Option to build benchmarks
option(BUILD_BENCHMARKS "Build benchmark suite" ON)
if(BUILD_BENCHMARKS)
    # v0.1/v0.2 benchmark uses simplified C API wrappers
    add_executable(bench_core bench/bench_core.cpp src/hpc_series.c)
    target_include_directories(bench_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # v0.3 benchmark for robust statistics
    add_executable(bench_v03 bench/bench_v03.cpp)
    target_link_libraries(bench_v03 hpcs_core)
    target_include_directories(bench_v03 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # v0.3 optimized benchmark (original vs parallel/fast)
    add_executable(bench_v03_optimized bench/bench_v03_optimized.cpp)
    target_link_libraries(bench_v03_optimized hpcs_core)
    target_include_directories(bench_v03_optimized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Anomaly detection benchmark (classical vs robust vs rolling)
    add_executable(bench_anomaly_detection bench/bench_anomaly_detection.cpp)
    target_link_libraries(bench_anomaly_detection hpcs_core)
    target_include_directories(bench_anomaly_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# When using the CUDA backend with NVHPC, ensure executables are linked with CUDA support
if(HPCS_ENABLE_GPU_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "PGI|NVHPC")
    add_link_options(-cuda)
endif()