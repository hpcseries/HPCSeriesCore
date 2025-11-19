# HPCSeriesCore Build Environment
# Based on Ubuntu 22.04 with GFortran, GCC, and CMake

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials and Fortran compiler
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Verify installations
RUN gfortran --version && \
    gcc --version && \
    cmake --version

# Default command: bash shell
CMD ["/bin/bash"]
