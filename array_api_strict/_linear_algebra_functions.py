"""
These functions are all also defined in the linalg extension, but we include
them here with wrappers in linalg so that the wrappers can be disabled if the
linalg extension is disabled in the flags.

"""

from __future__ import annotations

from ._dtypes import _numeric_dtypes
from ._array_object import Array
from ._flags import get_array_api_strict_flags

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._typing import Sequence, Tuple, Union

import numpy.linalg
import numpy as np

# Note: matmul is the numpy top-level namespace but not in np.linalg
def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matmul <numpy.matmul>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.matmul.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in matmul')

    return Array._new(np.matmul(x1._array, x2._array))

# Note: tensordot is the numpy top-level namespace but not in np.linalg

# Note: axes must be a tuple, unlike np.tensordot where it can be an array or array-like.
def tensordot(x1: Array, x2: Array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> Array:
    # Note: the restriction to numeric dtypes only is different from
    # np.tensordot.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in tensordot')

    return Array._new(np.tensordot(x1._array, x2._array, axes=axes))

# Note: this function is new in the array API spec. Unlike transpose, it only
# transposes the last two axes.
def matrix_transpose(x: Array, /) -> Array:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return Array._new(np.swapaxes(x._array, -1, -2))

# Note: vecdot is not in NumPy
def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in vecdot')

    if get_array_api_strict_flags()['api_version'] >= '2023.12':
        if axis >= 0:
            raise ValueError("axis must be negative in vecdot")
        elif axis < min(-1, -x1.ndim, -x2.ndim):
            raise ValueError("axis is out of bounds for x1 and x2")

    # In versions of the standard prior to 2023.12, vecdot applied axis after
    # broadcasting. This is different from applying it before broadcasting
    # when axis is nonnegative. The below code keeps this behavior for
    # 2022.12, primarily for backwards compatibility. Note that the behavior
    # is unambiguous when axis is negative, so the below code should work
    # correctly in that case regardless of which version is used.
    ndim = max(x1.ndim, x2.ndim)
    x1_shape = (1,)*(ndim - x1.ndim) + tuple(x1.shape)
    x2_shape = (1,)*(ndim - x2.ndim) + tuple(x2.shape)
    if x1_shape[axis] != x2_shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    x1_, x2_ = np.broadcast_arrays(x1._array, x2._array)
    x1_ = np.moveaxis(x1_, axis, -1)
    x2_ = np.moveaxis(x2_, axis, -1)

    res = x1_[..., None, :] @ x2_[..., None]
    return Array._new(res[..., 0, 0])
