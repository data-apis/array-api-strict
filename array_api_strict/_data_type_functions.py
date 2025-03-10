from __future__ import annotations

from ._array_object import Array
from ._creation_functions import _check_device
from ._dtypes import (
    _DType,
    _all_dtypes,
    _boolean_dtypes,
    _signed_integer_dtypes,
    _unsigned_integer_dtypes,
    _integer_dtypes,
    _real_floating_dtypes,
    _complex_floating_dtypes,
    _numeric_dtypes,
    _result_type,
)
from ._flags import get_array_api_strict_flags

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Tuple, Union, Optional
    from ._typing import Dtype, Device

import numpy as np

# Use to emulate the asarray(device) argument not existing in 2022.12
_default = object()

# Note: astype is a function, not an array method as in NumPy.
def astype(
    x: Array, dtype: Dtype, /, *, copy: bool = True, device: Optional[Device] = _default
) -> Array:
    if device is not _default:
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


def broadcast_arrays(*arrays: Array) -> List[Array]:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    return [
        Array._new(array, device=arrays[0].device) for array in np.broadcast_arrays(*[a._array for a in arrays])
    ]


def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_to <numpy.broadcast_to>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    return Array._new(np.broadcast_to(x._array, shape), device=x.device)


def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
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
    dtype: Dtype


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: Dtype


def finfo(type: Union[Dtype, Array], /) -> finfo_object:
    """
    Array API compatible wrapper for :py:func:`np.finfo <numpy.finfo>`.

    See its docstring for more information.
    """
    if isinstance(type, _DType):
        type = type._np_dtype
    fi = np.finfo(type)
    # Note: The types of the float data here are float, whereas in NumPy they
    # are scalars of the corresponding float dtype.
    return finfo_object(
        fi.bits,
        float(fi.eps),
        float(fi.max),
        float(fi.min),
        float(fi.smallest_normal),
        fi.dtype,
    )


def iinfo(type: Union[Dtype, Array], /) -> iinfo_object:
    """
    Array API compatible wrapper for :py:func:`np.iinfo <numpy.iinfo>`.

    See its docstring for more information.
    """
    if isinstance(type, _DType):
        type = type._np_dtype
    ii = np.iinfo(type)
    return iinfo_object(ii.bits, ii.max, ii.min, ii.dtype)


# Note: isdtype is a new function from the 2022.12 array API specification.
def isdtype(
    dtype: Dtype, kind: Union[Dtype, str, Tuple[Union[Dtype, str], ...]]
) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    for more details
    """
    if not isinstance(dtype, _DType):
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

def result_type(*arrays_and_dtypes: Union[Array, Dtype, int, float, complex, bool]) -> Dtype:
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
    A = [a for a in A if not isinstance(a, (bool, int, float, complex))]

    if len(A) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(A) == 1:
        result = A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
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
