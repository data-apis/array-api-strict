from .._flags import (set_array_api_strict_flags, get_array_api_strict_flags,
                      reset_array_api_strict_flags)

from .. import (asarray, unique_all, unique_counts, unique_inverse,
                unique_values, nonzero)

import pytest

@pytest.fixture(autouse=True)
def reset_flags():
    reset_array_api_strict_flags()
    yield
    reset_array_api_strict_flags()

def test_flags():
    # Test defaults
    flags = get_array_api_strict_flags()
    assert flags == {
        'standard_version': '2022.12',
        'data_dependent_shapes': True,
        'enabled_extensions': ('linalg', 'fft'),
    }

    # Test setting flags
    set_array_api_strict_flags(data_dependent_shapes=False)
    flags = get_array_api_strict_flags()
    assert flags == {
        'standard_version': '2022.12',
        'data_dependent_shapes': False,
        'enabled_extensions': ('linalg', 'fft'),
    }
    set_array_api_strict_flags(enabled_extensions=('fft',))
    flags = get_array_api_strict_flags()
    assert flags == {
        'standard_version': '2022.12',
        'data_dependent_shapes': False,
        'enabled_extensions': ('fft',),
    }
    # Make sure setting the version to 2021.12 disables fft
    set_array_api_strict_flags(standard_version='2021.12')
    flags = get_array_api_strict_flags()
    assert flags == {
        'standard_version': '2021.12',
        'data_dependent_shapes': False,
        'enabled_extensions': ('linalg',),
    }

    # Test setting flags with invalid values
    pytest.raises(ValueError, lambda:
                  set_array_api_strict_flags(standard_version='2020.12'))
    pytest.raises(ValueError, lambda: set_array_api_strict_flags(
                      enabled_extensions=('linalg', 'fft', 'invalid')))
    pytest.raises(ValueError, lambda: set_array_api_strict_flags(
        standard_version='2021.12',
        enabled_extensions=('linalg', 'fft')))


def test_data_dependent_shapes():
    a = asarray([0, 0, 1, 2, 2])
    mask = asarray([True, False, True, False, True])

    # Should not error
    unique_all(a)
    unique_counts(a)
    unique_inverse(a)
    unique_values(a)
    nonzero(a)
    a[mask]
    # TODO: add repeat when it is implemented

    set_array_api_strict_flags(data_dependent_shapes=False)

    pytest.raises(RuntimeError, lambda: unique_all(a))
    pytest.raises(RuntimeError, lambda: unique_counts(a))
    pytest.raises(RuntimeError, lambda: unique_inverse(a))
    pytest.raises(RuntimeError, lambda: unique_values(a))
    pytest.raises(RuntimeError, lambda: nonzero(a))
    pytest.raises(RuntimeError, lambda: a[mask])
