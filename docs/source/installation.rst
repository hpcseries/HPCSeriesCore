Installation
============

Requirements
------------

- Python 3.8 or higher
- NumPy >= 1.20
- C compiler (GCC, Clang, or MSVC)
- Fortran compiler (gfortran)
- OpenMP support (usually included with GCC/Clang)

System Requirements
~~~~~~~~~~~~~~~~~~~

**Linux/macOS**:
  - GCC 7.0+ or Clang 10.0+
  - gfortran 7.0+
  - OpenMP support

**Windows**:
  - Visual Studio 2019+ or MinGW-w64
  - gfortran (via MinGW-w64 or MSYS2)

Installation Methods
--------------------

Via pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install hpcs

This will install the pre-built binary wheel for your platform if available.

From Source
~~~~~~~~~~~

**1. Clone the repository:**

.. code-block:: bash

   git clone https://github.com/your-org/HPCSeriesCore.git
   cd HPCSeriesCore

**2. Install build dependencies:**

.. code-block:: bash

   pip install -r requirements-dev.txt

**3. Build and install:**

.. code-block:: bash

   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development with all optional dependencies:

.. code-block:: bash

   pip install -e ".[dev,docs,examples]"

This installs:
  - ``dev``: Testing and linting tools (pytest, ruff)
  - ``docs``: Documentation building (sphinx, nbsphinx)
  - ``examples``: Jupyter and plotting (jupyter, matplotlib, pandas)

Using Docker
~~~~~~~~~~~~

A Docker environment is provided for development and testing:

.. code-block:: bash

   # Build the Docker image
   docker compose -f docker-compose.python.yml build

   # Run container
   docker compose -f docker-compose.python.yml up -d

   # Execute commands inside container
   docker compose -f docker-compose.python.yml run --rm hpcs-python bash

Verification
------------

Verify your installation:

.. code-block:: python

   import hpcs
   print(hpcs.__version__)  # Should print 0.7.0

   # Check SIMD capabilities
   print(hpcs.simd_info())
   # {'isa': 'AVX2', 'width_bytes': 32, 'width_doubles': 4}

   # Run a simple test
   import numpy as np
   x = np.random.randn(1000)
   result = hpcs.mean(x)
   print(f"Mean: {result}")

CPU Information
~~~~~~~~~~~~~~~

Check your CPU topology and SIMD capabilities:

.. code-block:: bash

   hpcs cpuinfo

This will display:
  - Physical and logical core counts
  - Cache hierarchy (L1/L2/L3)
  - NUMA topology
  - SIMD capabilities (SSE2, AVX, AVX2, AVX-512)

Performance Calibration
-----------------------

For optimal performance, run calibration on your hardware:

.. code-block:: python

   import hpcs

   # Full calibration (takes ~30 seconds)
   hpcs.calibrate()

   # Save configuration
   hpcs.save_calibration_config()

This creates a configuration file at ``~/.hpcs/config.json`` with optimal settings for your CPU.

Troubleshooting
---------------

Import Error: No module named 'hpcs'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure the package is installed:

.. code-block:: bash

   pip list | grep hpcs

If not listed, reinstall with ``pip install hpcs``.

Compilation Errors
~~~~~~~~~~~~~~~~~~

**Missing compiler:**

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install gcc gfortran

   # macOS (via Homebrew)
   brew install gcc

   # Windows (via MSYS2)
   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran

**OpenMP not found:**

Ensure your compiler supports OpenMP. GCC and Clang typically include it by default.

Performance Issues
~~~~~~~~~~~~~~~~~~

1. **Run calibration**: ``hpcs.calibrate()``
2. **Check SIMD support**: ``hpcs.simd_info()``
3. **Verify OpenMP threads**: Set ``OMP_NUM_THREADS`` environment variable

.. code-block:: bash

   export OMP_NUM_THREADS=8  # Use 8 threads
   python your_script.py

Getting Help
------------

- **Documentation**: https://hpcseries.readthedocs.io
- **GitHub Issues**: https://github.com/your-org/HPCSeriesCore/issues
- **Examples**: See the ``notebooks/`` directory
