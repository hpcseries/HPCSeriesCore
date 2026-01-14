"""
HPCSeries Core - Python Package Build Configuration
====================================================

Builds Cython extensions linking to the native libhpcs_core library.

Prerequisites:
    - gcc, gfortran, cmake (automatically builds native library if missing)
    - numpy, Cython>=3.0

Build commands:
    python -m build              # Build wheel (recommended)
    pip install .                # Install from source (auto-builds native lib)
    pip install -e .             # Editable install (development)

Configuration is in pyproject.toml (PEP 517/518).
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildNativeLibrary(build_ext):
    """Custom build command that builds the native library first."""

    def run(self):
        """Build native library, then build Python extensions."""
        # Check if native library exists (shared or static), build if not
        lib_path_shared = Path("build/libhpcs_core.so")
        lib_path_static = Path("build/libhpcs_core.a")

        lib_exists = lib_path_shared.exists() or lib_path_static.exists()
        lib_path = lib_path_shared if lib_path_shared.exists() else lib_path_static

        if not lib_exists:
            print("\n" + "="*80)
            print("Building native library (libhpcs_core)...")
            print("="*80 + "\n")

            # Check for required tools
            self._check_build_tools()

            # Build native library with CMake
            self._build_native_library()

            # Verify library was built (check for either shared or static)
            lib_path_shared = Path("build/libhpcs_core.so")
            lib_path_static = Path("build/libhpcs_core.a")
            lib_exists = lib_path_shared.exists() or lib_path_static.exists()
            lib_path = lib_path_shared if lib_path_shared.exists() else lib_path_static

            if not lib_exists:
                raise RuntimeError(
                    "Native library build succeeded but libhpcs_core not found!"
                )

            print("\n" + "="*80)
            print(f"✓ Native library built successfully: {lib_path}")
            print("="*80 + "\n")
        else:
            print(f"✓ Native library found: {lib_path}")

        # Continue with normal extension building
        super().run()

    def _check_build_tools(self):
        """Check that cmake, gcc, and gfortran are available."""
        required_tools = {
            'cmake': 'CMake (https://cmake.org/)',
            'gcc': 'GCC compiler',
            'gfortran': 'GNU Fortran compiler'
        }

        missing = []
        for tool, description in required_tools.items():
            try:
                subprocess.run(
                    [tool, '--version'],
                    capture_output=True,
                    check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(f"  - {tool}: {description}")

        if missing:
            error_msg = "\n".join([
                "",
                "╔══════════════════════════════════════════════════════════════════════════╗",
                "║ ERROR: Missing required build tools                                     ║",
                "╚══════════════════════════════════════════════════════════════════════════╝",
                "",
                "The following tools are required to build HPCSeries Core:",
                "",
                *missing,
                "",
                "Installation instructions:",
                "",
                "  Ubuntu/Debian:",
                "    sudo apt-get install cmake gcc gfortran",
                "",
                "  Fedora/RHEL:",
                "    sudo dnf install cmake gcc gcc-gfortran",
                "",
                "  macOS (Homebrew):",
                "    brew install cmake gcc",
                "",
                "  Windows:",
                "    Install Visual Studio with C++ support and CMake",
                "    Or use WSL with Linux instructions",
                "",
            ])
            raise RuntimeError(error_msg)

    def _build_native_library(self):
        """Build the native library using CMake."""
        build_dir = Path("build")
        build_dir.mkdir(exist_ok=True)

        # Configure with CMake
        cmake_args = [
            'cmake',
            '..',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_TESTS=OFF',
            '-DBUILD_BENCHMARKS=OFF'
        ]

        # Add Ninja generator if available (faster builds)
        try:
            subprocess.run(['ninja', '--version'], capture_output=True, check=True)
            cmake_args.insert(1, '-G')
            cmake_args.insert(2, 'Ninja')
            print("Using Ninja build system (faster)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Using default build system (Ninja not available)")

        print(f"Configuring: {' '.join(cmake_args)}")
        subprocess.run(
            cmake_args,
            cwd=build_dir,
            check=True
        )

        # Build
        build_args = [
            'cmake',
            '--build', '.',
            '--config', 'Release'
        ]

        # Use parallel build
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
        build_args.extend(['--parallel', str(n_jobs)])

        print(f"Building: {' '.join(build_args)}")
        subprocess.run(
            build_args,
            cwd=build_dir,
            check=True
        )


# Project root directory
project_root = Path(__file__).parent.resolve()

# Define extensions
extensions = [
    Extension(
        "hpcs._core",
        sources=["python/hpcs/_core.pyx"],
        include_dirs=[
            str(project_root / "include"),
            str(project_root / "src" / "fortran"),
            numpy.get_include(),
        ],
        library_dirs=[str(project_root / "build")],
        libraries=["hpcs_core", "gfortran", "stdc++", "m"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp", "-std=c11"],
        extra_link_args=["-fopenmp"],
        language="c",
    ),
    Extension(
        "hpcs._simd",
        sources=["python/hpcs/_simd.pyx"],
        include_dirs=[
            str(project_root / "include"),
            str(project_root / "src" / "fortran"),
            numpy.get_include(),
        ],
        library_dirs=[str(project_root / "build")],
        libraries=["hpcs_core", "gfortran", "stdc++", "m"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp", "-std=c11"],
        extra_link_args=["-fopenmp"],
        language="c",
    ),
]

# Cythonize extensions
cythonized_extensions = cythonize(
    extensions,
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    },
)

# Setup
setup(
    ext_modules=cythonized_extensions,
    cmdclass={
        'build_ext': BuildNativeLibrary,
    },
)
