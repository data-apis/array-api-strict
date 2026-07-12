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
from .._dtypes import float32, float64, complex64, int32, int64, bool as xp_bool
from .._array_object import Array
from .._devices import CPU_DEVICE, ALL_DEVICES, Device
from .._info import __array_namespace_info__
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


def test_asarray_list_of_lists():
    lst = [[1, 2, 3], [4, 5, 6]]
    res = asarray(lst)
    assert res.shape == (2, 3)


def test_asarray_nested_arrays():
    # do not allow arrays in nested sequences
    with pytest.raises(TypeError):
        asarray([[1, 2, 3], asarray([4, 5, 6])])

    with pytest.raises(TypeError):
        asarray([1, asarray(1)])


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
    assert_raises(TypeError, lambda: full((1,), asarray(0)))


def test_full_like_errors():
    full_like(asarray(1), 0, device=CPU_DEVICE)  # Doesn't error
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, device="cpu"))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, device="gpu"))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, dtype=int))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, dtype="i"))
    assert_raises(TypeError, lambda: full(asarray(1), asarray(0)))


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



def _full(a, *args, **kwds):
    return full(a, fill_value=42.0, *args, **kwds)


def _full_like(a, *args, **kwds):
    return full_like(a, fill_value=42.0, *args, **kwds)


class TestDefaultDType:

    info = __array_namespace_info__()

    @pytest.mark.parametrize("device", ALL_DEVICES)
    @pytest.mark.parametrize("func", [empty, zeros, ones, _full])
    def test_ones_etc(self, func, device):
        a = func(1, device=device)
        assert a.dtype == self.info.default_dtypes(device=device)["real floating"]

    @pytest.mark.parametrize("func", [empty_like, zeros_like, ones_like, _full_like])
    def test_ones_like_etc_correct(self, func):
        # float32 is preserved
        a = ones(2, dtype=float32)
        device = Device('no_float64')
        b = func(a, device=device)
        assert b.dtype == self.info.default_dtypes(device=device)["real floating"]
        assert b.device == device

    @pytest.mark.parametrize("func", [empty_like, zeros_like, ones_like, _full_like])
    def test_ones_like_etc_incorrect(self, func):
        a = ones(2)
        assert a.dtype == float64
        assert a.device == Device()

        # XXX: a.dtype not supported by the device: ValueError or TypeError?

        # >>> a = torch.ones(3, dtype=torch.float64, device='cpu')
        # >>> torch.ones_like(a, device='mps')
        # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework
        # doesn't support float64.

        # incompatible dtype inferred from `a.dtype`
        with pytest.raises((TypeError, ValueError)):
            func(a, device=Device('no_float64'))

        # `a.dtype` is compatible but the explicit dtype= argument is incompatible
        a = ones(2, dtype=float32)
        with pytest.raises((TypeError, ValueError)):
            func(a, device=Device('no_float64'), dtype=float64)

    def test_eye(self):
        device = Device('no_float64')
        a = eye(3, device=device)
        assert a.dtype == self.info.default_dtypes(device=device)["real floating"]
        assert a.device == device

        with pytest.raises((TypeError, ValueError)):
            eye(3, device=device, dtype=float64)

    def test_linspace(self):
        device = Device('no_float64')

        a = linspace(1, 10, 11, device=device)
        assert a.dtype == self.info.default_dtypes(device=device)["real floating"]
        assert a.device == device

        a = linspace(1+0j, 10, 11, device=device)
        assert a.dtype == self.info.default_dtypes(device=device)["complex floating"]

        with pytest.raises((TypeError, ValueError)):
            linspace(1, 10, 11, device=device, dtype=float64)

    def test_arange(self):
        device = Device('no_float64')

        a = arange(0, 10, 1, device=device)
        assert a.dtype == self.info.default_dtypes(device=device)["integral"]
        assert a.device == device

        a = arange(0.0, 10, 1, device=device)
        assert a.dtype == self.info.default_dtypes(device=device)["real floating"]
        assert a.device == device

        with pytest.raises((TypeError, ValueError)):
            arange(0, 10, 1, device=device, dtype=float64)

        with pytest.raises((TypeError, ValueError)):
            arange(0.0, 10, 1, device=device, dtype=float64)

    def test_asarray(self):
        device = Device('no_float64')

        ### asarray(python_object)
        for x in (True, [False,]):
            arr = asarray(x, device=device)
            assert arr.dtype == xp_bool
            assert arr.device == device

        for x in [1, [1,]]:
            arr = asarray(x, device=device)
            assert arr.dtype == self.info.default_dtypes(device=device)['integral']
            assert arr.device == device

        for x in [1.0, [1.0,]]:
            arr = asarray(x, device=device)
            assert arr.dtype == self.info.default_dtypes(device=device)['real floating']
            assert arr.device == device

        for x in [1j, [1j,]]:
            arr = asarray(x, device=device)
            assert arr.dtype == self.info.default_dtypes(device=device)['complex floating']
            assert arr.device == device

        # asarray(python_object, dtype=unsupported_by_device)
        with pytest.raises(ValueError, match="Device"):
            asarray(1, dtype=float64, device=device)

        ### asarray(array)

        # compatible dtypes, device transfer
        src = asarray(1, dtype=float32, device=Device('device1'))
        dst = asarray(src, device=device)
        assert dst.device == device
        assert dst.dtype == float32

        # incompatible dtypes, device transfer
        src = asarray(1, dtype=float64, device=Device('device1'))

        with pytest.raises(ValueError, match="Device"):
            asarray(src, device=device)


def test_asarray_device_2():
    # device2 allows float64 but defaults to float32
    x = asarray([1.0], device=Device('device2'))
    assert x.dtype == float32

    x = asarray([1j], device=Device('device2'))
    assert x.dtype == complex64

    y = asarray([1.0], device=Device('device2'), dtype=float64)
    assert y.dtype == float64


def test_asarray_device_1():
    # regression test for https://github.com/data-apis/array-api-strict/issues/222
    x = np.ones(3, dtype=np.float32)
    y = asarray(x, device=Device('device1'))
    assert y.dtype == float32


def test_asarray_no_x64_device():
    x = asarray(3, device=Device("no_x64"))
    assert x.dtype == int32

    with pytest.raises(ValueError):
        asarray(3, device=Device("no_x64"), dtype=int64)

    y = zeros_like(ones(3, dtype=int32), device=Device("no_x64"))
    assert y.dtype == int32


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


@pytest.mark.parametrize(
    "device",
    [Device("device1"), Device("device2")],
)
def test_from_dlpack_preserves_device(device):
    x = asarray([1, 2, 3], device=device)
    y = from_dlpack(x)
    assert y.device == device
