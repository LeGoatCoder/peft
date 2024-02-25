# Copyright notice and license information

import pytest  # Import the pytest module for testing

def pytest_addoption(parser):
    """
    Add a command-line option for running regression tests.

    Args:
        parser (pytest.Parser): The parser object for defining command-line options.
    """
    parser.addoption(
        "--regression",  # Option name
        action="store_true",  # Store as a boolean flag
        default=False,  # Default value
        help="run regression tests"  # Help text
    )

def pytest_configure(config):
    """
    Add a mark to identify regression tests.

    Args:
        config (pytest.Config): The pytest configuration object.
    """
    config.addinivalue_line(
        "markers",  # Section to add the value to
        "regression: mark regression tests"  # Value to add
    )

def pytest_collection_modifyitems(config, items):
    """

