"""
HPCSeries Core v0.7 - Pytest Configuration
===========================================

Configuration and fixtures for Python tests.
"""

import pytest
import numpy as np


# Configure pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_array():
    """Small test array (5 elements)."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def medium_array():
    """Medium test array (1000 elements)."""
    np.random.seed(42)
    return np.random.randn(1000)


@pytest.fixture
def large_array():
    """Large test array (1M elements)."""
    np.random.seed(42)
    return np.random.randn(1000000)


@pytest.fixture
def array_with_outliers():
    """Array containing outliers for robust statistics tests."""
    # Normal data with a few outliers
    data = np.random.randn(100)
    data[10] = 100.0  # Outlier
    data[50] = -100.0  # Outlier
    return data
