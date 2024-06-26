from __future__ import annotations

from ._dtypes import (
    _real_floating_dtypes,
    _real_numeric_dtypes,
    _numeric_dtypes,
)
from ._array_object import Array
from ._dtypes import float32, complex64
from ._flags import requires_api_version, get_array_api_strict_flags
from ._creation_functions import zeros
from ._manipulation_functions import concat

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Tuple, Union
    from ._typing import Dtype

import numpy as np

@requires_api_version('2023.12')
def cumulative_sum(
    x: Array,
    /,
    *,
    axis: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    include_initial: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in cumulative_sum")
    dt = x.dtype if dtype is None else dtype
    if dtype is not None:
        dtype = dtype._np_dtype

    # TODO: The standard is not clear about what should happen when x.ndim == 0.
    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified in cumulative_sum for more than one dimension")
        axis = 0
    # np.cumsum does not support include_initial
    if include_initial:
        if axis < 0:
            axis += x.ndim
        x = concat([zeros(x.shape[:axis] + (1,) + x.shape[axis + 1:], dtype=dt), x], axis=axis)
    return Array._new(np.cumsum(x._array, axis=axis, dtype=dtype))

def max(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in max")
    return Array._new(np.max(x._array, axis=axis, keepdims=keepdims))


def mean(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in mean")
    return Array._new(np.mean(x._array, axis=axis, keepdims=keepdims))


def min(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in min")
    return Array._new(np.min(x._array, axis=axis, keepdims=keepdims))


def prod(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in prod")

    if dtype is None:
        # Note: In versions prior to 2023.12, sum() and prod() upcast for all
        # dtypes when dtype=None. For 2023.12, the behavior is the same as in
        # NumPy (only upcast for integral dtypes).
        if get_array_api_strict_flags()['api_version'] < '2023.12':
            if x.dtype == float32:
                dtype = np.float64
            elif x.dtype == complex64:
                dtype = np.complex128
    else:
        dtype = dtype._np_dtype
    return Array._new(np.prod(x._array, dtype=dtype, axis=axis, keepdims=keepdims))


def std(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    # Note: the keyword argument correction is different here
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in std")
    return Array._new(np.std(x._array, axis=axis, ddof=correction, keepdims=keepdims))


def sum(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")

    if dtype is None:
        # Note: In versions prior to 2023.12, sum() and prod() upcast for all
        # dtypes when dtype=None. For 2023.12, the behavior is the same as in
        # NumPy (only upcast for integral dtypes).
        if get_array_api_strict_flags()['api_version'] < '2023.12':
            if x.dtype == float32:
                dtype = np.float64
            elif x.dtype == complex64:
                dtype = np.complex128
    else:
        dtype = dtype._np_dtype
    return Array._new(np.sum(x._array, axis=axis, dtype=dtype, keepdims=keepdims))


def var(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    # Note: the keyword argument correction is different here
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in var")
    return Array._new(np.var(x._array, axis=axis, ddof=correction, keepdims=keepdims))
