from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._dtypes import DType, _all_dtypes, _np_dtype
from ._flags import get_array_api_strict_flags
from ._typing import NestedSequence, SupportsBufferProtocol, SupportsDLPack

if TYPE_CHECKING:
    # TODO import from typing (requires Python >=3.13)
    from typing_extensions import TypeIs

    # Circular import
    from ._array_object import Array, Device


class Undef(Enum):
    UNDEF = 0


_undef = Undef.UNDEF


def _check_valid_dtype(dtype: DType | None) -> None:
    # Note: Only spelling dtypes as the dtype objects is supported.
    if dtype not in (None,) + _all_dtypes:
        raise ValueError(f"dtype must be one of the supported dtypes, got {dtype!r}")


def _supports_buffer_protocol(obj: object) -> TypeIs[SupportsBufferProtocol]:
    try:
        memoryview(obj)  # type: ignore[arg-type]
    except TypeError:
        return False
    return True


def _check_device(device: Device | None) -> None:
    # _array_object imports in this file are inside the functions to avoid
    # circular imports
    from ._array_object import ALL_DEVICES, Device

    if device is not None and not isinstance(device, Device):
        raise ValueError(f"Unsupported device {device!r}")

    if device is not None and device not in ALL_DEVICES:
        raise ValueError(f"Unsupported device {device!r}")


def asarray(
    obj: Array | complex | NestedSequence[complex] | SupportsBufferProtocol,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: bool | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.asarray <numpy.asarray>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _np_dtype = None
    if dtype is not None:
        _np_dtype = dtype._np_dtype
    _check_device(device)
    if isinstance(obj, Array) and device is None:
        device = obj.device

    if np.lib.NumpyVersion(np.__version__) < '2.0.0':
        if copy is False:
            # Note: copy=False is not yet implemented in np.asarray for
            # NumPy 1

            # Work around it by creating the new array and seeing if NumPy
            # copies it.
            if isinstance(obj, Array):
                new_array = np.array(obj._array, copy=copy, dtype=_np_dtype)
                if new_array is not obj._array:
                    raise ValueError("Unable to avoid copy while creating an array from given array.")
                return Array._new(new_array, device=device)
            elif _supports_buffer_protocol(obj):
                # Buffer protocol will always support no-copy
                return Array._new(np.array(obj, copy=copy, dtype=_np_dtype), device=device)
            else:
                # No-copy is unsupported for Python built-in types.
                raise ValueError("Unable to avoid copy while creating an array from given object.")

        if copy is None:
            # NumPy 1 treats copy=False the same as the standard copy=None
            copy = False

    if isinstance(obj, Array):
        return Array._new(np.array(obj._array, copy=copy, dtype=_np_dtype), device=device)
    elif isinstance(obj, list | tuple):
        if any(isinstance(x, Array) for x in obj):
            raise TypeError("Nested Arrays are not allowed. Use `stack` instead.")

    if dtype is None and isinstance(obj, int) and (obj > 2 ** 64 or obj < -(2 ** 63)):
        # Give a better error message in this case. NumPy would convert this
        # to an object array. TODO: This won't handle large integers in lists.
        raise OverflowError("Integer out of bounds for array dtypes")

    res = np.array(obj, dtype=_np_dtype, copy=copy)
    return Array._new(res, device=device)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arange <numpy.arange>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)

    return Array._new(
        np.arange(start, stop, step, dtype=_np_dtype(dtype)),
        device=device,
    )


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.empty <numpy.empty>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)

    return Array._new(np.empty(shape, dtype=_np_dtype(dtype)), device=device)


def empty_like(
    x: Array, /, *, dtype: DType | None = None, device: Device | None = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.empty_like <numpy.empty_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)
    if device is None:
        device = x.device

    return Array._new(np.empty_like(x._array, dtype=_np_dtype(dtype)), device=device)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.eye <numpy.eye>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)

    return Array._new(
        np.eye(n_rows, M=n_cols, k=k, dtype=_np_dtype(dtype)), device=device
    )


def from_dlpack(
    x: SupportsDLPack,
    /,
    *,
    device: Device | Undef | None = _undef,
    copy: bool | Undef | None = _undef,
) -> Array:
    from ._array_object import Array

    if get_array_api_strict_flags()['api_version'] < '2023.12':
        if device is not _undef:
            raise ValueError("The device argument to from_dlpack requires at least version 2023.12 of the array API")
        if copy is not _undef:
            raise ValueError("The copy argument to from_dlpack requires at least version 2023.12 of the array API")

    # Going to wait for upstream numpy support
    if device is not _undef:
        _check_device(device)
    else:
        device = None
    if copy not in [_undef, None]:
        raise NotImplementedError("The copy argument to from_dlpack is not yet implemented")

    return Array._new(np.from_dlpack(x), device=device)


def full(
    shape: int | tuple[int, ...],
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.full <numpy.full>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)

    if isinstance(fill_value, Array) and fill_value.ndim == 0:
        fill_value = fill_value._array
    res = np.full(shape, fill_value, dtype=_np_dtype(dtype))
    if DType(res.dtype) not in _all_dtypes:
        # This will happen if the fill value is not something that NumPy
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full")
    return Array._new(res, device=device)


def full_like(
    x: Array,
    /,
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.full_like <numpy.full_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)
    if device is None:
        device = x.device

    res = np.full_like(x._array, fill_value, dtype=_np_dtype(dtype))
    if DType(res.dtype) not in _all_dtypes:
        # This will happen if the fill value is not something that NumPy
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full_like")
    return Array._new(res, device=device)


def linspace(
    start: complex,
    stop: complex,
    /,
    num: int,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: bool = True,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linspace <numpy.linspace>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)

    return Array._new(
        np.linspace(start, stop, num, dtype=_np_dtype(dtype), endpoint=endpoint),
        device=device,
    )


def meshgrid(*arrays: Array, indexing: Literal["xy", "ij"] = "xy") -> list[Array]:
    """
    Array API compatible wrapper for :py:func:`np.meshgrid <numpy.meshgrid>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    # Note: unlike np.meshgrid, only inputs with all the same dtype are
    # allowed

    if len({a.dtype for a in arrays}) > 1:
        raise ValueError("meshgrid inputs must all have the same dtype")

    if len({a.device for a in arrays}) > 1:
        raise ValueError("meshgrid inputs must all be on the same device")

    # arrays is allowed to be empty
    if arrays:
        device = arrays[0].device
    else:
        device = None

    return [
        Array._new(array, device=device)
        for array in np.meshgrid(*[a._array for a in arrays], indexing=indexing)
    ]


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.ones <numpy.ones>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)

    return Array._new(np.ones(shape, dtype=_np_dtype(dtype)), device=device)


def ones_like(
    x: Array, /, *, dtype: DType | None = None, device: Device | None = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.ones_like <numpy.ones_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)
    if device is None:
        device = x.device

    return Array._new(np.ones_like(x._array, dtype=_np_dtype(dtype)), device=device)


def tril(x: Array, /, *, k: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.tril <numpy.tril>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    if x.ndim < 2:
        # Note: Unlike np.tril, x must be at least 2-D
        raise ValueError("x must be at least 2-dimensional for tril")
    return Array._new(np.tril(x._array, k=k), device=x.device)


def triu(x: Array, /, *, k: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.triu <numpy.triu>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    if x.ndim < 2:
        # Note: Unlike np.triu, x must be at least 2-D
        raise ValueError("x must be at least 2-dimensional for triu")
    return Array._new(np.triu(x._array, k=k), device=x.device)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.zeros <numpy.zeros>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)

    return Array._new(np.zeros(shape, dtype=_np_dtype(dtype)), device=device)


def zeros_like(
    x: Array, /, *, dtype: DType | None = None, device: Device | None = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.zeros_like <numpy.zeros_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    _check_device(device)
    if device is None:
        device = x.device

    return Array._new(np.zeros_like(x._array, dtype=_np_dtype(dtype)), device=device)
