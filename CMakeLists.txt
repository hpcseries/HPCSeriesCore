cmake_minimum_required(VERSION 3.18)

project(HPCSeriesCore
    VERSION 0.1.0
    DESCRIPTION "High-Performance Computing Series Core Library"
    LANGUAGES C CXX Fortran
)

# Set C++ and C standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -march=native -ffast-math")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -Wall -Wextra -pedantic -fbounds-check")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -xHost -ipo")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -warn all -check all")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W4")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /W4")
endif()

# Core library target
add_library(hpcs_core STATIC
    src/fortran/hpcs_constants.f90
    src/fortran/hpcs_core_1d.f90
    src/fortran/hpcs_core_reductions.f90
    src/fortran/hpcs_core_utils.f90
)

target_include_directories(hpcs_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

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
    add_executable(bench_core bench/bench_core.cpp)
    target_link_libraries(bench_core hpcs_core)
endif()
