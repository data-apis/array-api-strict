from __future__ import annotations

from ._array_object import Array
from ._flags import requires_api_version
from ._dtypes import _numeric_dtypes

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Union

import numpy as np


def all(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.all <numpy.all>`.

    See its docstring for more information.
    """
    return Array._new(np.asarray(np.all(x._array, axis=axis, keepdims=keepdims)), device=x.device)


def any(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.any <numpy.any>`.

    See its docstring for more information.
    """
    return Array._new(np.asarray(np.any(x._array, axis=axis, keepdims=keepdims)), device=x.device)

@requires_api_version('2024.12')
def diff(
    x: Array,
    /,
    *,
    axis: int = -1,
    n: int = 1,
    prepend: Optional[Array] = None,
    append: Optional[Array] = None,
) -> Array:
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in diff")

    # TODO: The type promotion behavior for prepend and append is not
    # currently specified.

    # NumPy does not support prepend=None or append=None
    kwargs = dict(axis=axis, n=n)
    if prepend is not None:
        if prepend.device != x.device:
            raise ValueError(f"Arrays from two different devices ({prepend.device} and {x.device}) can not be combined.")
        kwargs['prepend'] = prepend._array
    if append is not None:
        if append.device != x.device:
            raise ValueError(f"Arrays from two different devices ({append.device} and {x.device}) can not be combined.")
        kwargs['append'] = append._array
    return Array._new(np.diff(x._array, **kwargs), device=x.device)
