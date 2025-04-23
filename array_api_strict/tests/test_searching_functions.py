import pytest

import array_api_strict as xp

from array_api_strict import ArrayAPIStrictFlags
from .._array_object import ALL_DEVICES, CPU_DEVICE, Device
from .._dtypes import _all_dtypes


def test_where_with_scalars():
    x = xp.asarray([1, 2, 3, 1])

    # Versions up to and including 2023.12 don't support scalar arguments
    with ArrayAPIStrictFlags(api_version='2023.12'):
        with pytest.raises(AttributeError, match="object has no attribute 'dtype'"):
            xp.where(x == 1, 42, 44)

    # Versions after 2023.12 support scalar arguments
    x_where = xp.where(x == 1, xp.asarray(42), 44)

    expected = xp.asarray([42, 44, 44, 42])
    assert xp.all(x_where == expected)

    # The spec does not allow both x1 and x2 to be scalars
    with pytest.raises(TypeError, match="Two scalars"):
        xp.where(x == 1, 42, 44)

    # The spec does not allow for condition to be scalar
    with pytest.raises(TypeError, match="Array"):
        xp.where(True, x, x)


def test_where_mixed_dtypes():
    # https://github.com/data-apis/array-api-strict/issues/131
    x =  xp.asarray([1., 2.])
    res = xp.where(x > 1.5, x, 0)
    assert res.dtype == x.dtype
    assert all(res == xp.asarray([0., 2.]))

    # retry with boolean x1, x2
    c = x > 1.5
    res = xp.where(c, False, c)
    assert all(res == xp.asarray([False, False]))


def test_where_f32():
    # https://github.com/data-apis/array-api-strict/issues/131#issuecomment-2723016294
    res = xp.where(xp.asarray([True, False]), 1., xp.asarray([2, 2], dtype=xp.float32))
    assert res.dtype == xp.float32


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_where_device_persists(device):
    """Test that the device of the input and output array are the same"""

    cond = xp.asarray([True, False], device=device)
    x1 = xp.asarray([1, 2], device=device)
    x2 = xp.asarray([3, 4], device=device)
    res = xp.where(cond, x1, x2)
    assert res.device == device
    res = xp.where(cond, 1, x2)
    assert res.device == device
    res = xp.where(cond, x1, 2)
    assert res.device == device


@pytest.mark.parametrize(
    "cond_device,x1_device,x2_device",
    [
        (CPU_DEVICE, CPU_DEVICE, Device("device1")),
        (CPU_DEVICE, Device("device1"), CPU_DEVICE),
        (Device("device1"), CPU_DEVICE, CPU_DEVICE),
    ]
)
def test_where_device_mismatch(cond_device, x1_device, x2_device):
    cond = xp.asarray([True, False], device=cond_device)
    x1 = xp.asarray([1, 2], device=x1_device)
    x2 = xp.asarray([3, 4], device=x2_device)
    with pytest.raises(ValueError, match="device"):
        xp.where(cond, x1, x2)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_where_numpy_generics(dtype):
    """
    Test that NumPy generics are explicitly disallowed.

    This must notably includes np.float64 and np.complex128, which are
    subclasses of float and complex respectively.
    """
    cond = xp.asarray(True)
    x1 = xp.asarray(1, dtype=dtype)
    x2 = xp.asarray(1, dtype=dtype)
    _ = xp.where(cond, x1, x2)

    with pytest.raises(TypeError, match="neither Array nor Python scalars"):
        xp.where(cond, x1, x2._array[()])
    with pytest.raises(TypeError, match="neither Array nor Python scalars"):
        xp.where(cond, x1._array[()], x2)
    with pytest.raises(TypeError, match="must be an Array"):
        xp.where(cond._array[()], x1, x2)
