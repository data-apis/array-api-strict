from typing import NamedTuple

import numpy as np

from ._array_object import Array
from ._flags import requires_data_dependent_shapes, requires_api_version
from ._helpers import _maybe_normalize_py_scalars
from ._dtypes import _result_type

# Note: np.unique() is split into four functions in the array API:
# unique_all, unique_counts, unique_inverse, and unique_values (this is done
# to remove polymorphic return types).

# Note: The various unique() functions are supposed to return multiple NaNs.
# This does not match the NumPy behavior, however, this is currently left as a
# TODO in this implementation as this behavior may be reverted in np.unique().
# See https://github.com/numpy/numpy/issues/20326.

# Note: The functions here return a namedtuple (np.unique() returns a normal
# tuple).


class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array


class UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array


@requires_data_dependent_shapes
def unique_all(x: Array, /) -> UniqueAllResult:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    values, indices, inverse_indices, counts = np.unique(
        x._array,
        return_counts=True,
        return_index=True,
        return_inverse=True,
        equal_nan=False,
    )
    # np.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueAllResult(
        Array._new(values, device=x.device),
        Array._new(indices, device=x.device),
        Array._new(inverse_indices, device=x.device),
        Array._new(counts, device=x.device),
    )


@requires_data_dependent_shapes
def unique_counts(x: Array, /) -> UniqueCountsResult:
    res = np.unique(
        x._array,
        return_counts=True,
        return_index=False,
        return_inverse=False,
        equal_nan=False,
    )

    return UniqueCountsResult(*[Array._new(i, device=x.device) for i in res])


@requires_data_dependent_shapes
def unique_inverse(x: Array, /) -> UniqueInverseResult:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    values, inverse_indices = np.unique(
        x._array,
        return_counts=False,
        return_index=False,
        return_inverse=True,
        equal_nan=False,
    )
    # np.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueInverseResult(Array._new(values, device=x.device),
                               Array._new(inverse_indices, device=x.device))


@requires_data_dependent_shapes
def unique_values(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    res = np.unique(
        x._array,
        return_counts=False,
        return_index=False,
        return_inverse=False,
        equal_nan=False,
    )
    return Array._new(res, device=x.device)


@requires_api_version('2025.12')
def isin(x1: Array | int, x2: Array | int, /, *, invert: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.isin <numpy.isin>`.

    See its docstring for more information.
    """
    # implementation here is from _elementwise_functions.py::_binary_ufunc_proto
    x1, x2 = _maybe_normalize_py_scalars(x1, x2, "integer", "isin")

    if x1.device != x2.device:
        raise ValueError(
            f"Arrays from two different devices ({x1.device} and {x2.device}) can not be combined."
        )
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    # x1, x2 = Array._normalize_two_args(x1, x2)   # no need to change 0D -> 1D here
    return Array._new(np.isin(x1._array, x2._array), device=x1.device)
