from dataclasses import dataclass

import numpy as np

from ._array_object import Array, Device
from ._creation_functions import Undef, _check_device, _undef
from ._dtypes import (
    DType,
    _all_dtypes,
    _boolean_dtypes,
    _complex_floating_dtypes,
    _integer_dtypes,
    _numeric_dtypes,
    _real_floating_dtypes,
    _result_type,
    _signed_integer_dtypes,
    _unsigned_integer_dtypes,
)
from ._flags import get_array_api_strict_flags


# Note: astype is a function, not an array method as in NumPy.
def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: bool = True,
    # _default is used to emulate the device argument not existing in 2022.12
    device: Device | Undef | None = _undef,
) -> Array:
    if device is not _undef:
        if get_array_api_strict_flags()['api_version'] >= '2023.12':
            _check_device(device)
        else:
            raise TypeError("The device argument to astype requires at least version 2023.12 of the array API")
    else:
        device = x.device

    if not copy and dtype == x.dtype:
        return x

    if isdtype(x.dtype, 'complex floating') and not isdtype(dtype, 'complex floating'):
        raise TypeError(
            f'The Array API standard stipulates that casting {x.dtype} to {dtype} should not be permitted. '
             'array-api-strict thus prohibits this conversion.'
        )

    return Array._new(x._array.astype(dtype=dtype._np_dtype, copy=copy), device=device)


def broadcast_arrays(*arrays: Array) -> list[Array]:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    return [
        Array._new(array, device=arrays[0].device) for array in np.broadcast_arrays(*[a._array for a in arrays])
    ]


def broadcast_to(x: Array, /, shape: tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_to <numpy.broadcast_to>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    return Array._new(np.broadcast_to(x._array, shape), device=x.device)


def can_cast(from_: DType | Array, to: DType, /) -> bool:
    """
    Array API compatible wrapper for :py:func:`np.can_cast <numpy.can_cast>`.

    See its docstring for more information.
    """
    if isinstance(from_, Array):
        from_ = from_.dtype
    elif from_ not in _all_dtypes:
        raise TypeError(f"{from_=}, but should be an array_api array or dtype")
    if to not in _all_dtypes:
        raise TypeError(f"{to=}, but should be a dtype")
    # Note: We avoid np.can_cast() as it has discrepancies with the array API,
    # since NumPy allows cross-kind casting (e.g., NumPy allows bool -> int8).
    # See https://github.com/numpy/numpy/issues/20870
    try:
        # We promote `from_` and `to` together. We then check if the promoted
        # dtype is `to`, which indicates if `from_` can (up)cast to `to`.
        dtype = _result_type(from_, to)
        return to == dtype
    except TypeError:
        # _result_type() raises if the dtypes don't promote together
        return False


# These are internal objects for the return types of finfo and iinfo, since
# the NumPy versions contain extra data that isn't part of the spec.
@dataclass
class finfo_object:
    bits: int
    # Note: The types of the float data here are float, whereas in NumPy they
    # are scalars of the corresponding float dtype.
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: DType


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: DType


def finfo(type: DType | Array, /) -> finfo_object:
    """
    Array API compatible wrapper for :py:func:`np.finfo <numpy.finfo>`.

    See its docstring for more information.
    """
    np_type = type._array if isinstance(type, Array) else type._np_dtype
    fi = np.finfo(np_type)
    # Note: The types of the float data here are float, whereas in NumPy they
    # are scalars of the corresponding float dtype.
    return finfo_object(
        fi.bits,
        float(fi.eps),
        float(fi.max),
        float(fi.min),
        float(fi.smallest_normal),
        DType(fi.dtype),
    )


def iinfo(type: DType | Array, /) -> iinfo_object:
    """
    Array API compatible wrapper for :py:func:`np.iinfo <numpy.iinfo>`.

    See its docstring for more information.
    """
    np_type = type._array if isinstance(type, Array) else type._np_dtype
    ii = np.iinfo(np_type)
    return iinfo_object(ii.bits, ii.max, ii.min, DType(ii.dtype))


# Note: isdtype is a new function from the 2022.12 array API specification.
def isdtype(dtype: DType, kind: DType | str | tuple[DType | str, ...]) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified
    data type ``kind``.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    for more details
    """
    if not isinstance(dtype, DType):
        raise TypeError(f"'dtype' must be a dtype, not a {type(dtype)!r}")

    if isinstance(kind, tuple):
        # Disallow nested tuples
        if any(isinstance(k, tuple) for k in kind):
            raise TypeError("'kind' must be a dtype, str, or tuple of dtypes and strs")
        return any(isdtype(dtype, k) for k in kind)
    elif isinstance(kind, str):
        if kind == 'bool':
            return dtype in _boolean_dtypes
        elif kind == 'signed integer':
            return dtype in _signed_integer_dtypes
        elif kind == 'unsigned integer':
            return dtype in _unsigned_integer_dtypes
        elif kind == 'integral':
            return dtype in _integer_dtypes
        elif kind == 'real floating':
            return dtype in _real_floating_dtypes
        elif kind == 'complex floating':
            return dtype in _complex_floating_dtypes
        elif kind == 'numeric':
            return dtype in _numeric_dtypes
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    elif kind in _all_dtypes:
        return dtype == kind
    else:
        raise TypeError(f"'kind' must be a dtype, str, or tuple of dtypes and strs, not {type(kind).__name__}")


def result_type(
    *arrays_and_dtypes: DType | Array | bool | int | float | complex,
) -> DType:
    """
    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.

    See its docstring for more information.
    """
    # Note: we use a custom implementation that gives only the type promotions
    # required by the spec rather than using np.result_type. NumPy implements
    # too many extra type promotions like int64 + uint64 -> float64, and does
    # value-based casting on scalar arrays.
    A = []
    scalars = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif isinstance(a, (bool, int, float, complex)):
            scalars.append(a)
        elif isinstance(a, np.ndarray) or a not in _all_dtypes:
            raise TypeError("result_type() inputs must be array_api arrays or dtypes")
        A.append(a)

    # remove python scalars
    B = [a for a in A if not isinstance(a, (bool, int, float, complex))]

    if len(B) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(B) == 1:
        result = B[0]
    else:
        t = B[0]
        for t2 in B[1:]:
            t = _result_type(t, t2)
        result = t

    if len(scalars) == 0:
        return result

    if get_array_api_strict_flags()['api_version'] <= '2023.12':
        raise TypeError("result_type() inputs must be array_api arrays or dtypes")

    # promote python scalars given the result_type for all arrays/dtypes
    from ._creation_functions import empty
    arr = empty(1, dtype=result)
    for s in scalars:
        x = arr._promote_scalar(s)
        result = _result_type(x.dtype, result)

    return result
