Contributing
============

We welcome contributions to HPCSeries Core!

Development Setup
-----------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/your-org/HPCSeriesCore.git
      cd HPCSeriesCore

2. **Install development dependencies**:

   .. code-block:: bash

      pip install -e ".[dev,docs,examples]"

3. **Build the library**:

   .. code-block:: bash

      # Build C/Fortran library
      mkdir build && cd build
      cmake .. -DCMAKE_BUILD_TYPE=Release
      make -j$(nproc)
      cd ..

      # Build Python extensions
      python setup.py build_ext --inplace
      pip install -e .

Running Tests
-------------

.. code-block:: bash

   # Run Python tests
   pytest tests/

   # Run with coverage
   pytest --cov=hpcs --cov-report=html tests/

Code Style
----------

We use Ruff for linting and formatting:

.. code-block:: bash

   # Check code style
   ruff check python/

   # Auto-fix issues
   ruff check --fix python/

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   make html
   # Open build/html/index.html

Write NumPy-style docstrings:

.. code-block:: python

   def my_function(x, threshold=3.0):
       """
       Short description.

       Parameters
       ----------
       x : ndarray
           Input array of shape (n,)
       threshold : float, optional
           Threshold value, by default 3.0

       Returns
       -------
       result : ndarray
           Output array of shape (n,)

       Examples
       --------
       >>> import hpcs
       >>> x = np.array([1, 2, 3])
       >>> hpcs.my_function(x)
       array([...])
       """
       pass

Pull Request Process
--------------------

1. **Fork the repository** and create a feature branch
2. **Make your changes** with clear commit messages
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Ensure tests pass**: ``pytest tests/``
6. **Submit a pull request** with a clear description

Commit Messages
~~~~~~~~~~~~~~~

Use conventional commits:

.. code-block:: text

   feat: Add new rolling_quantile function
   fix: Correct edge case in anomaly detection
   docs: Update installation guide
   perf: Optimize SIMD dispatch
   test: Add tests for masked operations

Areas for Contribution
----------------------

High Priority
~~~~~~~~~~~~~

- Additional SIMD kernels (ARM NEON support)
- GPU acceleration (CUDA/ROCm)
- Additional statistical functions
- Performance benchmarks
- Documentation improvements

Good First Issues
~~~~~~~~~~~~~~~~~

- Adding examples to notebooks
- Improving error messages
- Adding type hints
- Documentation typos
- Test coverage improvements

Contact
-------

- **GitHub Issues**: https://github.com/your-org/HPCSeriesCore/issues
- **Discussions**: https://github.com/your-org/HPCSeriesCore/discussions
