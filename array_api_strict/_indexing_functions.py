import numpy as np

from ._array_object import Array
from ._dtypes import _integer_dtypes
from ._flags import requires_api_version


def take(x: Array, indices: Array, /, *, axis: int | None = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.take <numpy.take>`.

    See its docstring for more information.
    """
    if axis is None and x.ndim != 1:
        raise ValueError("axis must be specified when ndim > 1")
    if indices.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in indexing")
    if indices.ndim != 1:
        raise ValueError("Only 1-dim indices array is supported")
    if x.device != indices.device:
        raise ValueError(f"Arrays from two different devices ({x.device} and {indices.device}) can not be combined.")
    return Array._new(np.take(x._array, indices._array, axis=axis), device=x.device)


@requires_api_version('2024.12')
def take_along_axis(x: Array, indices: Array, /, *, axis: int = -1) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.take_along_axis <numpy.take_along_axis>`.

    See its docstring for more information.
    """
    if x.device != indices.device:
        raise ValueError(f"Arrays from two different devices ({x.device} and {indices.device}) can not be combined.")
    return Array._new(np.take_along_axis(x._array, indices._array, axis), device=x.device)
