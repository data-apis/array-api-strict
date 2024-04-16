from .._flags import reset_array_api_strict_flags

import pytest

@pytest.fixture(autouse=True)
def reset_flags():
    reset_array_api_strict_flags()
    yield
    reset_array_api_strict_flags()
