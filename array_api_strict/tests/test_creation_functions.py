import sys
import warnings

from numpy.testing import assert_raises
import numpy as np

import pytest

from .. import all
from .._creation_functions import (
    asarray,
    arange,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .._dtypes import int16, float32, float64
from .._array_object import Array, CPU_DEVICE, Device
from .._flags import set_array_api_strict_flags

def test_asarray_errors():
    # Test various protections against incorrect usage
    assert_raises(TypeError, lambda: Array([1]))
    assert_raises(TypeError, lambda: asarray(["a"]))
    with assert_raises(ValueError), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        asarray([1.0], dtype=np.float16)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
    assert_raises(OverflowError, lambda: asarray(2**100))
    # Preferably this would be OverflowError
    # assert_raises(OverflowError, lambda: asarray([2**100]))
    assert_raises(TypeError, lambda: asarray([2**100]))
    asarray([1], device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: asarray([1], device="cpu"))
    assert_raises(ValueError, lambda: asarray([1], device="gpu"))

    assert_raises(ValueError, lambda: asarray([1], dtype=int))
    assert_raises(ValueError, lambda: asarray([1], dtype="i"))


def test_asarray_copy():
    a = asarray([1])
    b = asarray(a, copy=True)
    a[0] = 0
    assert all(b[0] == 1)
    assert all(a[0] == 0)

    a = asarray([1])
    b = asarray(a, copy=False)
    a[0] = 0
    assert all(b[0] == 0)

    a = asarray([1])
    assert_raises(ValueError, lambda: asarray(a, copy=False, dtype=float64))

    a = asarray([1])
    b = asarray(a, copy=None)
    a[0] = 0
    assert all(b[0] == 0)

    a = asarray([1])
    b = asarray(a, dtype=float64, copy=None)
    a[0] = 0
    assert all(b[0] == 1.0)

    # Python built-in types
    for obj in [True, 0, 0.0, 0j, [0], [[0]]]:
        asarray(obj, copy=True) # No error
        asarray(obj, copy=None) # No error
        assert_raises(ValueError, lambda: asarray(obj, copy=False))

    # Buffer protocol
    a = np.array([1])
    b = asarray(a, copy=True)
    assert isinstance(b, Array)
    a[0] = 0
    assert all(b[0] == 1)

    a = np.array([1])
    b = asarray(a, copy=False)
    assert isinstance(b, Array)
    a[0] = 0
    assert all(b[0] == 0)

    a = np.array([1])
    b = asarray(a, copy=None)
    assert isinstance(b, Array)
    a[0] = 0
    assert all(b[0] == 0)


@pytest.mark.xfail(sys.version_info.major*100 + sys.version_info.minor < 312,
                   reason="array conversion relies on buffer protocol, and "
                          "requires python >= 3.12"
)
def test_asarray_list_of_arrays():
    a = asarray(1, dtype=int16)
    b = asarray([1], dtype=int16)
    res = asarray([a, a])
    assert res.shape == (2,)
    assert res.dtype == int16
    assert all(res == asarray([1, 1]))

    res = asarray([b, b])
    assert res.shape == (2, 1)
    assert res.dtype == int16
    assert all(res == asarray([[1], [1]]))


def test_asarray_device_inference():
    assert asarray([1, 2, 3]).device == CPU_DEVICE

    x = asarray([1, 2, 3])
    assert asarray(x).device == CPU_DEVICE

    device1 = Device("device1")
    x = asarray([1, 2, 3], device=device1)
    assert asarray(x).device == device1

def test_arange_errors():
    arange(1, device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: arange(1, device="cpu"))
    assert_raises(ValueError, lambda: arange(1, device="gpu"))
    assert_raises(ValueError, lambda: arange(1, dtype=int))
    assert_raises(ValueError, lambda: arange(1, dtype="i"))


def test_empty_errors():
    empty((1,), device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: empty((1,), device="cpu"))
    assert_raises(ValueError, lambda: empty((1,), device="gpu"))
    assert_raises(ValueError, lambda: empty((1,), dtype=int))
    assert_raises(ValueError, lambda: empty((1,), dtype="i"))


def test_empty_like_errors():
    empty_like(asarray(1), device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: empty_like(asarray(1), device="cpu"))
    assert_raises(ValueError, lambda: empty_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: empty_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: empty_like(asarray(1), dtype="i"))


def test_eye_errors():
    eye(1, device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: eye(1, device="cpu"))
    assert_raises(ValueError, lambda: eye(1, device="gpu"))
    assert_raises(ValueError, lambda: eye(1, dtype=int))
    assert_raises(ValueError, lambda: eye(1, dtype="i"))


def test_full_errors():
    full((1,), 0, device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: full((1,), 0, device="cpu"))
    assert_raises(ValueError, lambda: full((1,), 0, device="gpu"))
    assert_raises(ValueError, lambda: full((1,), 0, dtype=int))
    assert_raises(ValueError, lambda: full((1,), 0, dtype="i"))


def test_full_like_errors():
    full_like(asarray(1), 0, device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, device="cpu"))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, device="gpu"))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, dtype=int))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, dtype="i"))


def test_linspace_errors():
    linspace(0, 1, 10, device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: linspace(0, 1, 10, device="cpu"))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, device="gpu"))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, dtype=float))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, dtype="f"))


def test_ones_errors():
    ones((1,), device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: ones((1,), device="cpu"))
    assert_raises(ValueError, lambda: ones((1,), device="gpu"))
    assert_raises(ValueError, lambda: ones((1,), dtype=int))
    assert_raises(ValueError, lambda: ones((1,), dtype="i"))


def test_ones_like_errors():
    ones_like(asarray(1), device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: ones_like(asarray(1), device="cpu"))
    assert_raises(ValueError, lambda: ones_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: ones_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: ones_like(asarray(1), dtype="i"))


def test_zeros_errors():
    zeros((1,), device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: zeros((1,), device="cpu"))
    assert_raises(ValueError, lambda: zeros((1,), device="gpu"))
    assert_raises(ValueError, lambda: zeros((1,), dtype=int))
    assert_raises(ValueError, lambda: zeros((1,), dtype="i"))


def test_zeros_like_errors():
    zeros_like(asarray(1), device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: zeros_like(asarray(1), device="cpu"))
    assert_raises(ValueError, lambda: zeros_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: zeros_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: zeros_like(asarray(1), dtype="i"))

def test_meshgrid_dtype_errors():
    # Doesn't raise
    meshgrid()
    meshgrid(asarray([1.], dtype=float32))
    meshgrid(asarray([1.], dtype=float32), asarray([1.], dtype=float32))

    assert_raises(ValueError, lambda: meshgrid(asarray([1.], dtype=float32), asarray([1.], dtype=float64)))


@pytest.mark.parametrize("api_version", ['2021.12', '2022.12', '2023.12'])
def from_dlpack_2023_12(api_version):
    if api_version != '2022.12':
        with pytest.warns(UserWarning):
            set_array_api_strict_flags(api_version=api_version)
    else:
        set_array_api_strict_flags(api_version=api_version)

    a = asarray([1., 2., 3.], dtype=float64)
    # Never an error
    capsule = a.__dlpack__()
    from_dlpack(capsule)

    exception = NotImplementedError if api_version >= '2023.12' else ValueError
    pytest.raises(exception, lambda: from_dlpack(capsule, device=CPU_DEVICE))
    pytest.raises(exception, lambda: from_dlpack(capsule, device=None))
    pytest.raises(exception, lambda: from_dlpack(capsule, copy=False))
    pytest.raises(exception, lambda: from_dlpack(capsule, copy=True))
    pytest.raises(exception, lambda: from_dlpack(capsule, copy=None))


def test_from_dlpack_default_device():
    x = asarray([1, 2, 3])
    y = from_dlpack(x)
    z = from_dlpack(np.asarray([1, 2, 3]))
    assert x.device == y.device == z.device == CPU_DEVICE
