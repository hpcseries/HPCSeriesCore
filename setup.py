"""
HPCSeries Core v0.7 - Python Bindings Setup
=============================================

Builds Cython extensions and packages the native libhpcs_core library.

Build commands:
  python setup.py build_ext --inplace  # Development build
  pip install -e .                     # Editable install
  python -m build                      # Build wheel
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
else:
    raise RuntimeError("Windows not yet supported in v0.7")

# Library path
lib_path = LIB_DIR / LIB_NAME
if not lib_path.exists():
    print(f"ERROR: {lib_path} not found!")
    print("Build the C library first with:")
    print("  cmake -S . -B build && cmake --build build")
    print(f"\nSearching for library in: {BUILD_DIR}")
    if BUILD_DIR.exists():
        print("Files in build directory:")
        import subprocess
        subprocess.run(["ls", "-lh", str(BUILD_DIR)])
    raise FileNotFoundError(f"Required library not found: {lib_path}")

# Compiler flags
extra_compile_args = [
    "-O3",
    "-march=native",
    "-fopenmp",
    "-std=c11",
]

extra_link_args = [
    "-fopenmp",
]

if IS_MACOS:
    # macOS-specific flags
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
