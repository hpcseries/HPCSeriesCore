cmake_minimum_required(VERSION 3.18)

project(HPCSeriesCore
    VERSION 0.2.0
    DESCRIPTION "High-Performance Computing Series Core Library"
    LANGUAGES C CXX Fortran
)

# Find OpenMP (required for parallel kernels)
find_package(OpenMP REQUIRED)

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
    src/fortran/hpcs_constants.f90
    src/fortran/hpcs_core_1d.f90
    src/fortran/hpcs_core_reductions.f90
    src/fortran/hpcs_core_utils.f90
    src/fortran/hpcs_core_prefix.f90
    src/fortran/hpcs_core_parallel.f90
)

target_include_directories(hpcs_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenMP to the library
target_link_libraries(hpcs_core PUBLIC OpenMP::OpenMP_Fortran)

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
endif()

# Option to build benchmarks
option(BUILD_BENCHMARKS "Build benchmark suite" ON)
if(BUILD_BENCHMARKS)
    # Benchmark uses simplified C API wrappers
    add_executable(bench_core bench/bench_core.cpp src/hpc_series.c)
    target_include_directories(bench_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
