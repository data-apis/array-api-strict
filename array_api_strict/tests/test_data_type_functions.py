import warnings

import pytest

from numpy.testing import assert_raises
import numpy as np

from .._creation_functions import asarray
from .._data_type_functions import astype, can_cast, isdtype
from .._dtypes import (
    bool, int8, int16, uint8, float64,
)
from .._flags import set_array_api_strict_flags


@pytest.mark.parametrize(
    "from_, to, expected",
    [
        (int8, int16, True),
        (int16, int8, False),
        (bool, int8, False),
        (asarray(0, dtype=uint8), int8, False),
    ],
)
def test_can_cast(from_, to, expected):
    """
    can_cast() returns correct result
    """
    assert can_cast(from_, to) == expected

def test_isdtype_strictness():
    assert_raises(TypeError, lambda: isdtype(float64, 64))
    assert_raises(ValueError, lambda: isdtype(float64, 'f8'))

    assert_raises(TypeError, lambda: isdtype(float64, (('integral',),)))
    with assert_raises(TypeError), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        isdtype(float64, np.object_)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)

    assert_raises(TypeError, lambda: isdtype(float64, None))
    with assert_raises(TypeError), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        isdtype(float64, np.float64)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)


@pytest.mark.parametrize("api_version", ['2021.12', '2022.12', '2023.12'])
def astype_device(api_version):
    if api_version != '2022.12':
        with pytest.warns(UserWarning):
            set_array_api_strict_flags(api_version=api_version)
    else:
        set_array_api_strict_flags(api_version=api_version)

    a = asarray([1, 2, 3], dtype=int8)
    # Never an error
    astype(a, int16)

    # Always an error
    astype(a, int16, device="cpu")

    if api_version >= '2023.12':
        astype(a, int8, device=None)
        astype(a, int8, device=a.device)
    else:
        pytest.raises(TypeError, lambda: astype(a, int8, device=None))
        pytest.raises(TypeError, lambda: astype(a, int8, device=a.device))
