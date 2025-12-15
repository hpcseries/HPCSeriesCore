# DetectArchitecture.cmake - CPU Architecture Detection for HPCSeries Core
#
# Executes detect_architecture.sh to classify CPU architecture and sets HPCS_ARCH variable.
#
# Output Variables:
#   HPCS_ARCH - Architecture classification (e.g., "x86_sapphire_rapids", "arm_graviton3")
#
# Environment Variables (can override detection):
#   HPCS_ARCH - Pre-set architecture (bypasses detection script)
#

# Check if HPCS_ARCH is already set via environment variable
if(DEFINED ENV{HPCS_ARCH})
    set(HPCS_ARCH "$ENV{HPCS_ARCH}" CACHE STRING "CPU Architecture (environment override)")
    message(STATUS "HPCSeries: Architecture override from environment: ${HPCS_ARCH}")
else()
    # Execute detection script
    set(DETECT_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/detect_architecture.sh")

    if(NOT EXISTS "${DETECT_SCRIPT}")
        message(WARNING "HPCSeries: detect_architecture.sh not found, defaulting to x86_generic")
        set(HPCS_ARCH "x86_generic" CACHE STRING "CPU Architecture (default)")
    else()
        # Make script executable (in case permissions are lost)
        if(UNIX)
            execute_process(
                COMMAND chmod +x "${DETECT_SCRIPT}"
                OUTPUT_QUIET
                ERROR_QUIET
            )
        endif()

        # Run detection script
        execute_process(
            COMMAND "${DETECT_SCRIPT}"
            OUTPUT_VARIABLE HPCS_ARCH_DETECTED
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE DETECT_RESULT
            ERROR_QUIET
        )

        if(DETECT_RESULT EQUAL 0 AND HPCS_ARCH_DETECTED)
            set(HPCS_ARCH "${HPCS_ARCH_DETECTED}" CACHE STRING "CPU Architecture (auto-detected)")
            message(STATUS "HPCSeries: Detected architecture: ${HPCS_ARCH}")
        else()
            message(WARNING "HPCSeries: Architecture detection failed, defaulting to x86_generic")
            set(HPCS_ARCH "x86_generic" CACHE STRING "CPU Architecture (fallback)")
        endif()
    endif()
endif()

# Validate architecture
set(VALID_ARCHITECTURES
    "x86_sapphire_rapids"
    "x86_emerald_rapids"
    "x86_icelake"
    "x86_cascadelake"
    "x86_amd_epyc"
    "x86_generic"
    "arm_graviton3"
    "arm_graviton2"
    "arm_generic"
)

if(NOT HPCS_ARCH IN_LIST VALID_ARCHITECTURES)
    message(WARNING "HPCSeries: Unknown architecture '${HPCS_ARCH}', using x86_generic")
    set(HPCS_ARCH "x86_generic" CACHE STRING "CPU Architecture (fallback)" FORCE)
endif()

# Set architecture family for conditional compilation
if(HPCS_ARCH MATCHES "^x86_")
    set(HPCS_ARCH_FAMILY "x86" CACHE STRING "Architecture family")
elseif(HPCS_ARCH MATCHES "^arm_")
    set(HPCS_ARCH_FAMILY "arm" CACHE STRING "Architecture family")
else()
    set(HPCS_ARCH_FAMILY "unknown" CACHE STRING "Architecture family")
endif()

message(STATUS "HPCSeries: Architecture family: ${HPCS_ARCH_FAMILY}")
