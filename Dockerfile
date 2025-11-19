# HPCSeriesCore Build Environment
# Based on Ubuntu 22.04 with GFortran, GCC, and CMake

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials, Fortran compiler and Python with NumPy
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    gcc \
    g++ \
    cmake \
    make \
    git \
    vim \
    nano \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies. Using pip avoids pulling in unnecessary
# packages and ensures the benchmark script can run inside the
# container without requiring system‑wide NumPy.
RUN pip3 install --no-cache-dir numpy

# Set working directory
WORKDIR /workspace

# Verify installations
RUN gfortran --version && \
    gcc --version && \
    cmake --version

# Default command: bash shell
CMD ["/bin/bash"]
