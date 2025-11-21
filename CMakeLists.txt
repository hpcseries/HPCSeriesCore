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
    endif()
elseif(HPCS_ENABLE_GPU_CUDA)
    message(WARNING "HPCSeries: CUDA backend not yet implemented (Phase 1). Falling back to CPU-only.")
    # Future: add CUDA runtime bindings
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
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -march=native ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -Wall -Wextra -pedantic -fbounds-check ${OpenMP_Fortran_FLAGS}")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -xHost -fp-model fast -ipo ${OpenMP_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -warn all -check all ${OpenMP_Fortran_FLAGS}")
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

# Option to build tests
option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS)
    enable_testing()

    # C test
    add_executable(test_core_c tests/test_core_c.c)
    target_link_libraries(test_core_c hpcs_core)
    add_test(NAME test_core_c COMMAND test_core_c)

    # C++ test
    add_executable(test_core_cpp tests/test_core_cpp.cpp)
    target_link_libraries(test_core_cpp hpcs_core)
    add_test(NAME test_core_cpp COMMAND test_core_cpp)

    # Robust anomaly detection tests (v0.3+)
    add_executable(test_robust_anomalies tests/test_robust_anomalies.c)
    target_link_libraries(test_robust_anomalies hpcs_core)
    add_test(NAME test_robust_anomalies COMMAND test_robust_anomalies)

    # GPU infrastructure tests (v0.4 Phase 1)
    add_executable(test_gpu_infrastructure tests/test_gpu_infrastructure.c)
    target_link_libraries(test_gpu_infrastructure hpcs_core)
    add_test(NAME test_gpu_infrastructure COMMAND test_gpu_infrastructure)

    # Phase 2 acceleration tests (v0.4 Phase 2)
    add_executable(test_phase2_accel tests/test_phase2_accel.c)
    target_link_libraries(test_phase2_accel hpcs_core)
    add_test(NAME test_phase2_accel COMMAND test_phase2_accel)
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
