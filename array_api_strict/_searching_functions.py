from __future__ import annotations

from ._array_object import Array
from ._dtypes import _result_type, _real_numeric_dtypes
from ._flags import requires_data_dependent_shapes, requires_api_version

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Optional, Tuple

import numpy as np


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmax")
    return Array._new(np.asarray(np.argmax(x._array, axis=axis, keepdims=keepdims)))


def argmin(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmin")
    return Array._new(np.asarray(np.argmin(x._array, axis=axis, keepdims=keepdims)))


@requires_data_dependent_shapes
def nonzero(x: Array, /) -> Tuple[Array, ...]:
    """
    Array API compatible wrapper for :py:func:`np.nonzero <numpy.nonzero>`.

    See its docstring for more information.
    """
    # Note: nonzero is disallowed on 0-dimensional arrays
    if x.ndim == 0:
        raise ValueError("nonzero is not allowed on 0-dimensional arrays")
    return tuple(Array._new(i) for i in np.nonzero(x._array))

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
    sorter = sorter._array if sorter is not None else None
    # TODO: The sort order of nans and signed zeros is implementation
    # dependent. Should we error/warn if they are present?

    # x1 must be 1-D, but NumPy already requires this.
    return Array._new(np.searchsorted(x1._array, x2._array, side=side, sorter=sorter))

def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.where <numpy.where>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.where(condition._array, x1._array, x2._array))
