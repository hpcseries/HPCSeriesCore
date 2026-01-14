cmake_minimum_required(VERSION 3.15)
project(SignalCoreExample
    VERSION 1.0.0
    DESCRIPTION "HPCSeries Core C++ Integration Example"
    LANGUAGES CXX
)

# Enable Fortran optionally - needed to resolve OpenMP::OpenMP_Fortran from Core
# (Core library is built with Fortran, so we need Fortran language support to
#  understand its OpenMP dependencies, even though this example is pure C++)
include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
endif()

# C++14 minimum for modern practices
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Find HPCSeries Core (installed via cmake --install)
# ============================================================================
# This demonstrates how SignalCore will integrate with HPCSeries Core.
#
# Usage:
#   cmake -DCMAKE_PREFIX_PATH=/path/to/installed/core ..
#
# The find_package() call locates:
#   - hpcs::core target (links library + sets include paths)
#   - Version information for compatibility checking
#   - All dependencies (OpenMP, etc.)
# ============================================================================

find_package(hpcs_core 0.8 CONFIG REQUIRED)

# Display what was found
message(STATUS "Found HPCSeries Core ${hpcs_core_VERSION}")
message(STATUS "  Target: hpcs::hpcs_core")

# ============================================================================
# Build Example Executable
# ============================================================================

add_executable(signalcore_example signalcore_example.cpp)

# Link against hpcs::hpcs_core target
# This automatically handles:
#   - Library linking
#   - Include directories
#   - Dependencies (OpenMP, etc.)
target_link_libraries(signalcore_example PRIVATE hpcs::hpcs_core)

# Compiler warnings (NASA-level quality)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(signalcore_example PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(signalcore_example PRIVATE /W4)
endif()

# ============================================================================
# Alternative: Using pkg-config (if available)
# ============================================================================
# Uncomment below to use pkg-config instead of CMake package config:
#
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(HPCS REQUIRED hpcs_core)
#
# add_executable(signalcore_example signalcore_example.cpp)
# target_include_directories(signalcore_example PRIVATE ${HPCS_INCLUDE_DIRS})
# target_link_libraries(signalcore_example PRIVATE ${HPCS_LIBRARIES})
# ============================================================================

# Installation (optional)
install(TARGETS signalcore_example
    RUNTIME DESTINATION bin
)

# ============================================================================
# Post-build verification
# ============================================================================
add_custom_command(TARGET signalcore_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "✓ SignalCore example built successfully"
    COMMAND ${CMAKE_COMMAND} -E echo "  Run: ./signalcore_example"
    VERBATIM
)
