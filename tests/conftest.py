"""Script which contains fixtures for the tests"""

import pytest

import numpes as pes

@pytest.fixture(autouse=True)
def reset_config():
    """Reset the global configuration before each test"""
    pes.reset_config()