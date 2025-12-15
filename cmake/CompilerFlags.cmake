# CompilerFlags.cmake - Architecture-Aware Compiler Flags for HPCSeries Core
#
# Configures compilation flags based on:
#   1. Detected CPU architecture (HPCS_ARCH from DetectArchitecture.cmake)
#   2. Compilation profile (HPCS_PROFILE: SAFE or FAST)
#
# Output Variables:
#   HPCS_COMPILER_FLAGS_FORTRAN - Fortran compiler flags
#   HPCS_COMPILER_FLAGS_CXX     - C++ compiler flags
#   HPCS_COMPILER_FLAGS_C       - C compiler flags
#
# Environment Variables:
#   HPCS_PROFILE - Compilation profile (SAFE or FAST), default: SAFE
#

# Require HPCS_ARCH to be set (should be set by DetectArchitecture.cmake)
if(NOT DEFINED HPCS_ARCH)
    message(FATAL_ERROR "HPCSeries: HPCS_ARCH not set. Include DetectArchitecture.cmake first.")
endif()

# Determine compilation profile
if(DEFINED ENV{HPCS_PROFILE})
    set(HPCS_PROFILE "$ENV{HPCS_PROFILE}" CACHE STRING "Compilation profile (environment override)")
else()
    set(HPCS_PROFILE "SAFE" CACHE STRING "Compilation profile (SAFE or FAST)")
endif()

# Validate profile
if(NOT HPCS_PROFILE MATCHES "^(SAFE|FAST)$")
    message(WARNING "HPCSeries: Invalid HPCS_PROFILE '${HPCS_PROFILE}', using SAFE")
    set(HPCS_PROFILE "SAFE" CACHE STRING "Compilation profile" FORCE)
endif()

message(STATUS "HPCSeries: Compilation profile: ${HPCS_PROFILE}")

# ============================================================================
# Base Flags (Common to All Architectures)
# ============================================================================

set(HPCS_BASE_FLAGS "-O3 -funroll-loops -fomit-frame-pointer")

# SAFE profile: No fast-math (IEEE 754 compliant)
set(HPCS_SAFE_FLAGS "${HPCS_BASE_FLAGS} -fno-unsafe-math-optimizations -fno-fast-math")

# FAST profile: Add fast-math flags (5-10% faster, relaxed IEEE 754)
set(HPCS_FAST_FLAGS "${HPCS_BASE_FLAGS} -ffast-math -fno-math-errno -funsafe-math-optimizations -fno-trapping-math")

# Select profile
if(HPCS_PROFILE STREQUAL "FAST")
    set(HPCS_SELECTED_FLAGS "${HPCS_FAST_FLAGS}")
    message(STATUS "HPCSeries: Using FAST profile (performance-optimized, relaxed IEEE 754)")
else()
    set(HPCS_SELECTED_FLAGS "${HPCS_SAFE_FLAGS}")
    message(STATUS "HPCSeries: Using SAFE profile (IEEE 754 compliant)")
endif()

# ============================================================================
# Architecture-Specific Flags
# ============================================================================

# Determine architecture-specific flags based on HPCS_ARCH
if(HPCS_ARCH_FAMILY STREQUAL "x86")
    # x86/x86_64 architectures (Intel, AMD)
    set(HPCS_ARCH_FLAGS "-march=native -mtune=native")
    message(STATUS "HPCSeries: Using x86 flags: -march=native -mtune=native")

elseif(HPCS_ARCH_FAMILY STREQUAL "arm")
    # ARM architectures (Graviton2, Graviton3, etc.)
    set(HPCS_ARCH_FLAGS "-mcpu=native -mtune=native")
    message(STATUS "HPCSeries: Using ARM flags: -mcpu=native -mtune=native")

else()
    # Unknown architecture, use conservative flags
    set(HPCS_ARCH_FLAGS "")
    message(WARNING "HPCSeries: Unknown architecture family '${HPCS_ARCH_FAMILY}', using generic flags")
endif()

# ============================================================================
# Combine Flags
# ============================================================================

set(HPCS_COMPILER_FLAGS_BASE "${HPCS_SELECTED_FLAGS} ${HPCS_ARCH_FLAGS}")

# Fortran-specific flags
set(HPCS_COMPILER_FLAGS_FORTRAN "${HPCS_COMPILER_FLAGS_BASE}")

# C++-specific flags
set(HPCS_COMPILER_FLAGS_CXX "${HPCS_COMPILER_FLAGS_BASE}")

# C-specific flags
set(HPCS_COMPILER_FLAGS_C "${HPCS_COMPILER_FLAGS_BASE}")

# ============================================================================
# Debug Information
# ============================================================================

message(STATUS "HPCSeries: Compiler flags configured:")
message(STATUS "  Architecture: ${HPCS_ARCH}")
message(STATUS "  Profile:      ${HPCS_PROFILE}")
message(STATUS "  Fortran:      ${HPCS_COMPILER_FLAGS_FORTRAN}")
message(STATUS "  C++:          ${HPCS_COMPILER_FLAGS_CXX}")
message(STATUS "  C:            ${HPCS_COMPILER_FLAGS_C}")

# ============================================================================
# Cache Variables for User Override
# ============================================================================

set(HPCS_COMPILER_FLAGS_FORTRAN "${HPCS_COMPILER_FLAGS_FORTRAN}"
    CACHE STRING "HPCSeries Fortran compiler flags")
set(HPCS_COMPILER_FLAGS_CXX "${HPCS_COMPILER_FLAGS_CXX}"
    CACHE STRING "HPCSeries C++ compiler flags")
set(HPCS_COMPILER_FLAGS_C "${HPCS_COMPILER_FLAGS_C}"
    CACHE STRING "HPCSeries C compiler flags")

mark_as_advanced(HPCS_COMPILER_FLAGS_FORTRAN HPCS_COMPILER_FLAGS_CXX HPCS_COMPILER_FLAGS_C)
