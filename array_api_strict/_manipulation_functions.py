from __future__ import annotations

from ._array_object import Array
from ._creation_functions import asarray
from ._data_type_functions import result_type
from ._dtypes import _integer_dtypes
from ._flags import requires_api_version, get_array_api_strict_flags

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Union

import numpy as np

# Note: the function name is different here
def concat(
    arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.

    See its docstring for more information.
    """
    # Note: Casting rules here are different from the np.concatenate default
    # (no for scalars with axis=None, no cross-kind casting)
    dtype = result_type(*arrays)
    arrays = tuple(a._array for a in arrays)
    return Array._new(np.concatenate(arrays, axis=axis, dtype=dtype._np_dtype))


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.

    See its docstring for more information.
    """
    return Array._new(np.expand_dims(x._array, axis))


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.

    See its docstring for more information.
    """
    return Array._new(np.flip(x._array, axis=axis))

@requires_api_version('2023.12')
def moveaxis(
    x: Array,
    source: Union[int, Tuple[int, ...]],
    destination: Union[int, Tuple[int, ...]],
    /,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.moveaxis <numpy.moveaxis>`.

    See its docstring for more information.
    """
    return Array._new(np.moveaxis(x._array, source, destination))

# Note: The function name is different here (see also matrix_transpose).
# Unlike transpose(), the axes argument is required.
def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    return Array._new(np.transpose(x._array, axes))

@requires_api_version('2023.12')
def repeat(
    x: Array,
    repeats: Union[int, Array],
    /,
    *,
    axis: Optional[int] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.repeat <numpy.repeat>`.

    See its docstring for more information.
    """
    if isinstance(repeats, Array):
        data_dependent_shapes = get_array_api_strict_flags()['data_dependent_shapes']
        if not data_dependent_shapes:
            raise RuntimeError("repeat() with repeats as an array requires data-dependent shapes, but the data_dependent_shapes flag has been disabled for array-api-strict")
        if repeats.dtype not in _integer_dtypes:
            raise TypeError("The repeats array must have an integer dtype")
    elif isinstance(repeats, int):
        repeats = asarray(repeats)
    else:
        raise TypeError("repeats must be an int or array")

    return Array._new(np.repeat(x._array, repeats, axis=axis))

# Note: the optional argument is called 'shape', not 'newshape'
def reshape(x: Array,
            /,
            shape: Tuple[int, ...],
            *,
            copy: Optional[bool] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.reshape <numpy.reshape>`.

    See its docstring for more information.
    """

    data = x._array
    if copy:
        data = np.copy(data)

    reshaped = np.reshape(data, shape)

    if copy is False and not np.shares_memory(data, reshaped):
        raise AttributeError("Incompatible shape for in-place modification.")

    return Array._new(reshaped)


def roll(
    x: Array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.roll <numpy.roll>`.

    See its docstring for more information.
    """
    return Array._new(np.roll(x._array, shift, axis=axis))


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.

    See its docstring for more information.
    """
    return Array._new(np.squeeze(x._array, axis=axis))


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    result_type(*arrays)
    arrays = tuple(a._array for a in arrays)
    return Array._new(np.stack(arrays, axis=axis))


@requires_api_version('2023.12')
def tile(x: Array, repetitions: Tuple[int, ...], /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.tile <numpy.tile>`.

    See its docstring for more information.
    """
    # Note: NumPy allows repetitions to be an int or array
    if not isinstance(repetitions, tuple):
        raise TypeError("repetitions must be a tuple")
    return Array._new(np.tile(x._array, repetitions))

# Note: this function is new
@requires_api_version('2023.12')
def unstack(x: Array, /, *, axis: int = 0) -> Tuple[Array, ...]:
    if not (-x.ndim <= axis < x.ndim):
        raise ValueError("axis out of range")

    if axis < 0:
        axis += x.ndim

    slices = (slice(None),) * axis
    return tuple(x[slices + (i, ...)] for i in range(x.shape[axis]))
