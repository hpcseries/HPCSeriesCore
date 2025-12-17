cmake_minimum_required(VERSION 3.18)

project(HPCSeriesCore
    VERSION 0.7.0
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

# Ensure Fortran preprocessing is enabled for compilers that need it (e.g. GNU).
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -ffree-line-length-none")
endif()

elseif(HPCS_ENABLE_GPU_CUDA)
    message(STATUS "HPCSeries: GPU acceleration enabled (CUDA backend)")
    add_compile_definitions(HPCS_USE_CUDA)
    # Add CUDA-specific flags for NVHPC/PGI compiler
    if(CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mfree -Mpreprocess -cuda")
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

# Enable position-independent code (required for Python extensions)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Architecture-aware compiler flags
# Detects CPU architecture and applies appropriate optimization flags
# Supports SAFE (default, IEEE 754 compliant) and FAST (via HPCS_PROFILE=FAST) profiles
include(cmake/DetectArchitecture.cmake)
include(cmake/CompilerFlags.cmake)

# Compiler flags - architecture-aware optimization + OpenMP
# Note: -cpp flag enables preprocessing for #ifdef/#else/#endif directives in .f90 files
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp ${HPCS_COMPILER_FLAGS_FORTRAN} ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS_DEBUG "-cpp -g -Wall -Wextra -pedantic -fbounds-check ${OpenMP_Fortran_FLAGS}")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fpp ${HPCS_COMPILER_FLAGS_FORTRAN} -fp-model fast -ipo ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS_DEBUG "-fpp -g -warn all -check all ${OpenMP_Fortran_FLAGS}")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mfree -Mpreprocess ${HPCS_COMPILER_FLAGS_FORTRAN} ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS_DEBUG "-Mfree -Mpreprocess -g -Mbounds -Mchkptr -Mchkstk ${OpenMP_Fortran_FLAGS}")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HPCS_COMPILER_FLAGS_CXX} -Wall -Wextra ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${HPCS_COMPILER_FLAGS_C} -Wall -Wextra ${OpenMP_C_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast /W4 ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /fp:fast /W4 ${OpenMP_C_FLAGS}")
endif()

# Core library sources (order matters for Fortran module dependencies)
set(HPCS_CORE_SOURCES
    # v0.1/v0.2 modules (maintained for backward compatibility)
    src/fortran/hpcs_constants.f90
    # v0.6 SIMD interface (early in build order)
    src/fortran/hpcs_simd_interface.f90
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
    # v0.4 CPU modules (batched, masked, axis, anomaly)
    src/fortran/hpcs_cpu_detect.f90    # Hardware-aware adaptive parallelization
    src/hpcs_cpu_detect.c              # Enhanced CPU detection (C)
    src/hpcs_calibrate.c               # v0.5 Benchmark-based auto-tuning
    # v0.6 SIMD vectorization modules
    src/hpcs_simd_dispatch.c           # SIMD runtime dispatch
    src/hpcs_reduce_simd.c             # SIMD reduction kernels (OpenMP)
    src/hpcs_reduce_intrinsics.c       # SIMD reduction kernels (intrinsics)
    src/hpcs_rolling_simd.c            # SIMD rolling operations
    src/hpcs_zscore_simd.c             # SIMD z-score normalization
    src/hpcs_simd_align.c              # Memory alignment utilities
    src/hpcs_fortran_simd_bridge.c     # Fortran-SIMD integration bridge
    src/hpcs_axis_simd.c               # SIMD axis operations (architecture-aware)
    src/fortran/hpcs_core_batched.f90
    src/fortran/hpcs_core_masked.f90
    src/fortran/hpcs_core_axis.f90
    src/fortran/hpcs_core_anomaly.f90
)

# Add CUDA runtime module BEFORE hpcs_core_accel.f90 (dependency order)
if(HPCS_ENABLE_GPU_CUDA)
    list(APPEND HPCS_CORE_SOURCES src/fortran/hpcs_cuda_runtime.f90)
endif()

# v0.4 GPU acceleration infrastructure must come after CUDA runtime
list(APPEND HPCS_CORE_SOURCES src/fortran/hpcs_core_accel.f90)

# Core library target
add_library(hpcs_core STATIC ${HPCS_CORE_SOURCES})

target_include_directories(hpcs_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenMP and C++ stdlib to the library
target_link_libraries(hpcs_core PUBLIC OpenMP::OpenMP_Fortran)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(hpcs_core PUBLIC stdc++)
endif()

# Link CUDA runtime for NVHPC/PGI compiler when CUDA is enabled
if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    target_link_options(hpcs_core PUBLIC -cuda)
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
    # Moved to tests/c/ - use Makefile in tests/c/ to build
    # add_executable(test_hpcs_baseline tests/c/test_hpcs_baseline.cpp)
    # target_link_libraries(test_hpcs_baseline hpcs_core)
    # add_test(NAME test_hpcs_baseline COMMAND test_hpcs_baseline)

    # Robust statistics and anomaly detection tests (v0.3+)
    # Moved to tests/c/ - use Makefile in tests/c/ to build
    # add_executable(test_hpcs_robust tests/c/test_hpcs_robust.c)
    # target_link_libraries(test_hpcs_robust hpcs_core)
    # add_test(NAME test_hpcs_robust COMMAND test_hpcs_robust)

    # ========================================================================
    # GPU Acceleration Tests (2106 assertions total)
    # ========================================================================

    # Tests moved to tests/c/ - use Makefile in tests/c/ to build and run

    # CUDA device detection test (Phase 2 - validates runtime integration)
    # if(HPCS_ENABLE_GPU_OPENMP OR HPCS_ENABLE_GPU_CUDA OR HPCS_ENABLE_GPU_HIP)
    #     add_executable(test_cuda_device_count tests/c/test_cuda_device_count.c)
    #     target_link_libraries(test_cuda_device_count hpcs_core)
    #     if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    #         target_link_options(test_cuda_device_count PRIVATE -cuda)
    #     endif()
    #     add_test(NAME test_cuda_device_count COMMAND test_cuda_device_count)
    # endif()

    # GPU infrastructure and backend tests
    # add_executable(test_hpcs_gpu_infra tests/c/test_hpcs_gpu_infra.c)
    # target_link_libraries(test_hpcs_gpu_infra hpcs_core)
    # if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    #     target_link_options(test_hpcs_gpu_infra PRIVATE -cuda)
    # endif()
    # add_test(NAME test_hpcs_gpu_infra COMMAND test_hpcs_gpu_infra)

    # GPU kernel implementation tests (Phase 3B optimized)
    # add_executable(test_hpcs_gpu_kernels tests/c/test_hpcs_gpu_kernels.c)
    # target_link_libraries(test_hpcs_gpu_kernels hpcs_core)
    # if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    #     target_link_options(test_hpcs_gpu_kernels PRIVATE -cuda)
    # endif()
    # add_test(NAME test_hpcs_gpu_kernels COMMAND test_hpcs_gpu_kernels)

    # GPU memory management tests (Phase 4A)
    # add_executable(test_hpcs_gpu_memory tests/c/test_hpcs_gpu_memory.c)
    # target_link_libraries(test_hpcs_gpu_memory hpcs_core)
    # if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    #     target_link_options(test_hpcs_gpu_memory PRIVATE -cuda)
    # endif()
    # add_test(NAME test_hpcs_gpu_memory COMMAND test_hpcs_gpu_memory)
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
    if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
        target_link_options(bench_v03 PRIVATE -cuda)
    endif()

    # v0.3 optimized benchmark (original vs parallel/fast)
    add_executable(bench_v03_optimized bench/bench_v03_optimized.cpp)
    target_link_libraries(bench_v03_optimized hpcs_core)
    target_include_directories(bench_v03_optimized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
        target_link_options(bench_v03_optimized PRIVATE -cuda)
    endif()

    # v0.4 parallel benchmark (batched/axis operations with OpenMP)
    add_executable(bench_v04_parallel bench/bench_v04_parallel.cpp)
    target_link_libraries(bench_v04_parallel hpcs_core OpenMP::OpenMP_CXX)
    target_include_directories(bench_v04_parallel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
        target_link_options(bench_v04_parallel PRIVATE -cuda)
    endif()

    # Anomaly detection benchmark (classical vs robust vs rolling)
    add_executable(bench_anomaly_detection bench/bench_anomaly_detection.cpp)
    target_link_libraries(bench_anomaly_detection hpcs_core)
    target_include_directories(bench_anomaly_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
        target_link_options(bench_anomaly_detection PRIVATE -cuda)
    endif()

    # GPU acceleration benchmark (CPU vs GPU comparison)
    add_executable(bench_gpu_acceleration bench/bench_gpu_acceleration.cpp)
    target_link_libraries(bench_gpu_acceleration hpcs_core)
    target_include_directories(bench_gpu_acceleration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(HPCS_ENABLE_GPU_CUDA AND CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
        target_link_options(bench_gpu_acceleration PRIVATE -cuda)
    endif()

    # Adaptive parallelization demo (CPU detection and auto-tuning)
    add_executable(adaptive_demo examples/adaptive_demo.c)
    target_link_libraries(adaptive_demo hpcs_core OpenMP::OpenMP_C)
    target_include_directories(adaptive_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # NUMA affinity demo (v0.5 - NUMA topology and thread affinity)
    add_executable(numa_affinity_demo examples/numa_affinity_demo.c)
    target_link_libraries(numa_affinity_demo hpcs_core OpenMP::OpenMP_C)
    target_include_directories(numa_affinity_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Calibration demo (v0.5 - Benchmark-based auto-tuning)
    add_executable(calibration_demo examples/calibration_demo.c)
    target_link_libraries(calibration_demo hpcs_core OpenMP::OpenMP_C)
    target_include_directories(calibration_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Calibration verification (v0.5 - Before/after performance comparison)
    add_executable(verify_calibration examples/verify_calibration.c)
    target_link_libraries(verify_calibration hpcs_core OpenMP::OpenMP_C)
    target_include_directories(verify_calibration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # SIMD benchmark (v0.6 - Vectorization performance comparison)
    add_executable(bench_simd examples/bench_simd.c)
    target_link_libraries(bench_simd hpcs_core OpenMP::OpenMP_C)
    target_include_directories(bench_simd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Fortran-SIMD integration test (v0.6)
    add_executable(test_fortran_simd examples/test_fortran_simd.c)
    target_link_libraries(test_fortran_simd hpcs_core OpenMP::OpenMP_C)
    target_include_directories(test_fortran_simd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Axis operations benchmark: Fortran OpenMP vs C SIMD comparison (v0.6)
    add_executable(bench_axis_comparison examples/bench_axis_comparison.c)
    target_link_libraries(bench_axis_comparison hpcs_core OpenMP::OpenMP_C)
    target_include_directories(bench_axis_comparison PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Prefetch optimization benchmark (v0.6 - Microarchitecture Opt)
    add_executable(bench_prefetch bench/bench_prefetch.c)
    target_link_libraries(bench_prefetch hpcs_core OpenMP::OpenMP_C)
    target_include_directories(bench_prefetch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()