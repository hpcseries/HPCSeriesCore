#!/bin/bash
# detect_architecture.sh - CPU Architecture Detection for HPCSeries Core
#
# Detects CPU architecture and outputs classification for CMake configuration.
# Supports Intel x86 (Sapphire Rapids, Ice Lake, generic) and ARM (Graviton3, generic).
#
# Usage:
#   ./detect_architecture.sh
#
# Output:
#   Architecture classification string (e.g., "x86_sapphire_rapids", "arm_graviton3")
#
# Environment Variables:
#   HPCS_ARCH - Override auto-detection (e.g., HPCS_ARCH=x86_icelake)
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for environment override
if [ -n "${HPCS_ARCH:-}" ]; then
    echo "$HPCS_ARCH"
    exit 0
fi

# Check if /proc/cpuinfo exists
if [ ! -f /proc/cpuinfo ]; then
    echo "x86_generic"
    >&2 echo -e "${YELLOW}Warning: /proc/cpuinfo not found, defaulting to x86_generic${NC}"
    exit 0
fi

# Parse CPU information
VENDOR=$(grep -m1 "vendor_id" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "")
MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "")
FLAGS=$(grep -m1 "^flags" /proc/cpuinfo 2>/dev/null | cut -d: -f2 || echo "")

# ARM systems use different field names
if [ -z "$VENDOR" ]; then
    # Try ARM-specific fields
    VENDOR=$(grep -m1 "CPU implementer" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "")
    MODEL=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "")
    # Check for ARM in architecture field
    ARCH=$(uname -m)
    if [[ "$ARCH" =~ ^(aarch64|arm64) ]]; then
        VENDOR="ARM"
    fi
fi

# Function to detect Intel generation
detect_intel_generation() {
    local model="$1"
    local flags="$2"

    # Check for Sapphire Rapids (4th Gen Xeon Scalable)
    if echo "$model" | grep -qi "Sapphire Rapids"; then
        echo "x86_sapphire_rapids"
        return
    fi

    # Check for Emerald Rapids (5th Gen Xeon Scalable - newer than Sapphire)
    if echo "$model" | grep -qi "Emerald Rapids"; then
        echo "x86_emerald_rapids"
        return
    fi

    # Check for Ice Lake (3rd Gen Xeon Scalable)
    if echo "$model" | grep -qi "Ice Lake"; then
        echo "x86_icelake"
        return
    fi

    # Check for Cascade Lake (2nd Gen Xeon Scalable)
    if echo "$model" | grep -qi "Cascade Lake"; then
        echo "x86_cascadelake"
        return
    fi

    # Fallback: Use AVX-512 detection
    if echo "$flags" | grep -q "avx512"; then
        # Has AVX-512, likely Ice Lake or newer
        echo "x86_icelake"
        return
    fi

    # Fallback: Use AVX2 detection
    if echo "$flags" | grep -q "avx2"; then
        # Has AVX2 but not AVX-512, likely Cascade Lake or older
        echo "x86_cascadelake"
        return
    fi

    # Final fallback: generic x86
    echo "x86_generic"
}

# Function to detect ARM generation
detect_arm_generation() {
    local model="$1"

    # Check for AWS Graviton3 (Neoverse V1)
    if echo "$model" | grep -qi "Neoverse.*V1"; then
        echo "arm_graviton3"
        return
    fi

    # Check for AWS Graviton2 (Neoverse N1)
    if echo "$model" | grep -qi "Neoverse.*N1"; then
        echo "arm_graviton2"
        return
    fi

    # Generic ARM with NEON
    echo "arm_generic"
}

# Classify architecture based on vendor
case "$VENDOR" in
    GenuineIntel)
        ARCH=$(detect_intel_generation "$MODEL" "$FLAGS")
        echo "$ARCH"
        ;;

    AuthenticAMD)
        # AMD detection (for c6a instances)
        if echo "$MODEL" | grep -qi "EPYC"; then
            # AMD EPYC processors
            if echo "$FLAGS" | grep -q "avx2"; then
                echo "x86_amd_epyc"
            else
                echo "x86_generic"
            fi
        else
            echo "x86_generic"
        fi
        ;;

    ARM|*ARM*|0x41)
        # ARM architecture
        ARCH=$(detect_arm_generation "$MODEL")
        echo "$ARCH"
        ;;

    *)
        # Unknown vendor, check architecture
        MACHINE=$(uname -m)
        if [[ "$MACHINE" =~ ^(aarch64|arm64) ]]; then
            echo "arm_generic"
        else
            echo "x86_generic"
        fi
        ;;
esac
