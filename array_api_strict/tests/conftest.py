import os

import pytest

from .._flags import ENVIRONMENT_VARIABLES, reset_array_api_strict_flags


def pytest_configure(config):
    for env_var in ENVIRONMENT_VARIABLES:
        if env_var in os.environ:
            pytest.exit(f"ERROR: {env_var} is set. array-api-strict environment variables must not be set when the tests are run.")

@pytest.fixture(autouse=True)
def reset_flags():
    reset_array_api_strict_flags()
    yield
    reset_array_api_strict_flags()
