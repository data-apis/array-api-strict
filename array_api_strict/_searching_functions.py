from __future__ import annotations

from ._array_object import Array
from ._dtypes import _result_type, _real_numeric_dtypes, bool as _bool
from ._flags import requires_data_dependent_shapes, requires_api_version, get_array_api_strict_flags

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Optional, Tuple, Union

import numpy as np


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmax")
    return Array._new(np.asarray(np.argmax(x._array, axis=axis, keepdims=keepdims)), device=x.device)


def argmin(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmin")
    return Array._new(np.asarray(np.argmin(x._array, axis=axis, keepdims=keepdims)), device=x.device)


@requires_data_dependent_shapes
def nonzero(x: Array, /) -> Tuple[Array, ...]:
    """
    Array API compatible wrapper for :py:func:`np.nonzero <numpy.nonzero>`.

    See its docstring for more information.
    """
    # Note: nonzero is disallowed on 0-dimensional arrays
    if x.ndim == 0:
        raise ValueError("nonzero is not allowed on 0-dimensional arrays")
    return tuple(Array._new(i, device=x.device) for i in np.nonzero(x._array))


@requires_api_version('2024.12')
def count_nonzero(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.count_nonzero <numpy.count_nonzero>`

    See its docstring for more information.
    """
    arr = np.count_nonzero(x._array, axis=axis, keepdims=keepdims)
    return Array._new(np.asarray(arr), device=x.device)


@requires_api_version('2023.12')
def searchsorted(
    x1: Array,
    x2: Array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Optional[Array] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.searchsorted <numpy.searchsorted>`.

    See its docstring for more information.
    """
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in searchsorted")

    if x1.device != x2.device:
        raise ValueError(f"Arrays from two different devices ({x1.device} and {x2.device}) can not be combined.")

    sorter = sorter._array if sorter is not None else None
    # TODO: The sort order of nans and signed zeros is implementation
    # dependent. Should we error/warn if they are present?

    # x1 must be 1-D, but NumPy already requires this.
    return Array._new(np.searchsorted(x1._array, x2._array, side=side, sorter=sorter), device=x1.device)

def where(
    condition: Array,
    x1: bool | int | float | complex | Array,
    x2: bool | int | float | complex | Array, /
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.where <numpy.where>`.

    See its docstring for more information.
    """
    if get_array_api_strict_flags()['api_version'] > '2023.12':
        num_scalars = 0

        if isinstance(x1, (bool, float, complex, int)):
            x1 = Array._new(np.asarray(x1), device=condition.device)
            num_scalars += 1

        if isinstance(x2, (bool, float, complex, int)):
            x2 = Array._new(np.asarray(x2), device=condition.device)
            num_scalars += 1

        if num_scalars == 2:
            raise ValueError("One of x1, x2 arguments must be an array.")

    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    
    if condition.dtype != _bool:
        raise TypeError("`condition` must be have a boolean data type")

    if len({a.device for a in (condition, x1, x2)}) > 1:
        raise ValueError("Inputs to `where` must all use the same device")

    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.where(condition._array, x1._array, x2._array), device=x1.device)
