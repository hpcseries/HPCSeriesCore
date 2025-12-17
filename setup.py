"""
HPCSeries Core - Python Package Build Configuration
====================================================

Builds Cython extensions linking to the native libhpcs_core library.

Prerequisites:
    - Build libhpcs_core.a first: cmake -S . -B build && cmake --build build
    - Requires: gcc, gfortran, cmake, numpy, Cython>=3.0

Build commands:
    python -m build              # Build wheel (recommended)
    pip install -e .             # Editable install (development)
    python setup.py build_ext    # Build extensions only

Configuration is in pyproject.toml (PEP 517/518).
"""

import os
import sys
from pathlib import Path
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# Detect platform
IS_LINUX = sys.platform.startswith('linux')
IS_MACOS = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

# Project root
ROOT_DIR = Path(__file__).parent.absolute()
BUILD_DIR = ROOT_DIR / "build"
LIB_DIR = BUILD_DIR

# Find the compiled library
if IS_LINUX:
    LIB_NAME = "libhpcs_core.a"
elif IS_MACOS:
    LIB_NAME = "libhpcs_core.a"
elif IS_WINDOWS:
    raise RuntimeError(
        "Windows is not officially supported in v0.7.0.\n"
        "Consider using WSL2 (Windows Subsystem for Linux) or Docker."
    )
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

# Library path
lib_path = LIB_DIR / LIB_NAME
if not lib_path.exists():
    error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║ ERROR: Native library not found                                         ║
╚══════════════════════════════════════════════════════════════════════════╝

Missing: {lib_path}

The HPCSeries Core Python package requires building the native library first.

Build instructions:
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)
    cd ..

Then retry: pip install -e .

See: https://github.com/hpcseries/HPCSeriesCore#installation
"""
    print(error_msg, file=sys.stderr)

    if BUILD_DIR.exists():
        import subprocess
        print(f"\nFiles found in {BUILD_DIR}:")
        subprocess.run(["ls", "-lh", str(BUILD_DIR)])

    raise FileNotFoundError(f"Required library not found: {lib_path}")

# Compiler flags for Cython extensions
# Note: These flags are used when compiling the Cython-generated C code,
#       not the native library (which is pre-built via CMake)
extra_compile_args = [
    "-O3",                # Maximum optimization
    "-march=native",      # CPU-specific optimizations (matches CMake)
    "-fopenmp",           # OpenMP support
    "-std=c11",           # C11 standard
]

extra_link_args = [
    "-fopenmp",           # Link OpenMP runtime
]

if IS_MACOS:
    # macOS-specific: Suppress Cython-generated warnings
    extra_compile_args.extend([
        "-Wno-unreachable-code-fallthrough",
        "-Wno-deprecated-declarations",
    ])

# Include directories
include_dirs = [
    str(ROOT_DIR / "include"),
    str(ROOT_DIR / "src" / "fortran"),
    np.get_include(),
]

# Library directories
library_dirs = [
    str(LIB_DIR),
]

# Libraries to link
libraries = [
    "hpcs_core",  # Our static library
    "gfortran",   # Fortran runtime
    "stdc++",     # C++ standard library (needed for SIMD kernels)
    "m",          # Math library
]

if IS_MACOS:
    libraries.extend(["quadmath"])  # Required for gfortran on macOS

# Define Cython extensions
extensions = [
    Extension(
        name="hpcs._core",
        sources=["python/hpcs/_core.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
    Extension(
        name="hpcs._simd",
        sources=["python/hpcs/_simd.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
]

# Cythonize with compiler directives
ext_modules = cythonize(
    extensions,
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,  # Disable bounds checking for performance
        "wraparound": False,   # Disable negative indexing
        "cdivision": True,     # C-style division (faster)
        "initializedcheck": False,  # Skip initialization checks
        "embedsignature": True,  # Include signatures in docstrings
    },
    annotate=False,  # Set to True to generate HTML annotation files
)

setup(
    ext_modules=ext_modules,
)
