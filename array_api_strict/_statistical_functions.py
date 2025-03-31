from __future__ import annotations

from typing import Any

import numpy as np

from ._array_object import Array
from ._creation_functions import ones, zeros
from ._dtypes import (
    DType,
    _floating_dtypes,
    _np_dtype,
    _numeric_dtypes,
    _real_floating_dtypes,
    _real_numeric_dtypes,
    complex64,
    float32,
)
from ._flags import get_array_api_strict_flags, requires_api_version
from ._manipulation_functions import concat


@requires_api_version('2023.12')
def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in cumulative_sum")

    # TODO: The standard is not clear about what should happen when x.ndim == 0.
    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified in cumulative_sum for more than one dimension")
        axis = 0
    # np.cumsum does not support include_initial
    if include_initial:
        if axis < 0:
            axis += x.ndim
        x = concat([zeros(x.shape[:axis] + (1,) + x.shape[axis + 1:], dtype=x.dtype), x], axis=axis)
    return Array._new(np.cumsum(x._array, axis=axis, dtype=_np_dtype(dtype)), device=x.device)


@requires_api_version('2024.12')
def cumulative_prod(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in cumulative_prod")
    if x.ndim == 0:
        raise ValueError("Only ndim >= 1 arrays are allowed in cumulative_prod")

    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified in cumulative_prod for more than one dimension")
        axis = 0

    # np.cumprod does not support include_initial
    if include_initial:
        if axis < 0:
            axis += x.ndim
        x = concat([ones(x.shape[:axis] + (1,) + x.shape[axis + 1:], dtype=x.dtype), x], axis=axis)
    return Array._new(np.cumprod(x._array, axis=axis, dtype=_np_dtype(dtype)), device=x.device)


def max(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in max")
    return Array._new(np.max(x._array, axis=axis, keepdims=keepdims), device=x.device)


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:

    allowed_dtypes = (
        _floating_dtypes
        if get_array_api_strict_flags()['api_version'] > '2023.12'
        else _real_floating_dtypes
    )

    if x.dtype not in allowed_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in mean")
    return Array._new(np.mean(x._array, axis=axis, keepdims=keepdims), device=x.device)


def min(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in min")
    return Array._new(np.min(x._array, axis=axis, keepdims=keepdims), device=x.device)


def _np_dtype_sumprod(x: Array, dtype: DType | None) -> np.dtype[Any] | None:
    """In versions prior to 2023.12, sum() and prod() upcast for all
    dtypes when dtype=None. For 2023.12, the behavior is the same as in
    NumPy (only upcast for integral dtypes).
    """
    if dtype is None and get_array_api_strict_flags()['api_version'] < '2023.12':
        if x.dtype == float32:
            return np.float64  # type: ignore[return-value]
        elif x.dtype == complex64:
            return np.complex128  # type: ignore[return-value]
    return _np_dtype(dtype)


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in prod")

    np_dtype = _np_dtype_sumprod(x, dtype)
    return Array._new(
        np.prod(x._array, dtype=np_dtype, axis=axis, keepdims=keepdims),
        device=x.device,
    )


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    # Note: the keyword argument correction is different here
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in std")
    return Array._new(np.std(x._array, axis=axis, ddof=correction, keepdims=keepdims), device=x.device)


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")

    np_dtype = _np_dtype_sumprod(x, dtype)
    return Array._new(
        np.sum(x._array, axis=axis, dtype=np_dtype, keepdims=keepdims),
        device=x.device,
    )


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    # Note: the keyword argument correction is different here
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in var")
    return Array._new(np.var(x._array, axis=axis, ddof=correction, keepdims=keepdims), device=x.device)
