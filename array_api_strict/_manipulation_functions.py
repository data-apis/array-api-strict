from __future__ import annotations

import numpy as np

from ._array_object import Array
from ._creation_functions import asarray
from ._data_type_functions import astype, result_type
from ._dtypes import _integer_dtypes, int64, uint64
from ._flags import get_array_api_strict_flags, requires_api_version


# Note: the function name is different here
def concat(
    arrays: tuple[Array, ...] | list[Array], /, *, axis: int | None = 0
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.

    See its docstring for more information.
    """
    # Note: Casting rules here are different from the np.concatenate default
    # (no for scalars with axis=None, no cross-kind casting)
    dtype = result_type(*arrays)
    if len({a.device for a in arrays}) > 1:
        raise ValueError("concat inputs must all be on the same device")
    result_device = arrays[0].device

    np_arrays = tuple(a._array for a in arrays)
    return Array._new(
        np.concatenate(np_arrays, axis=axis, dtype=dtype._np_dtype),
        device=result_device,
    )


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.

    See its docstring for more information.
    """
    return Array._new(np.expand_dims(x._array, axis), device=x.device)


def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.

    See its docstring for more information.
    """
    return Array._new(np.flip(x._array, axis=axis), device=x.device)

@requires_api_version('2023.12')
def moveaxis(
    x: Array,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
    /,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.moveaxis <numpy.moveaxis>`.

    See its docstring for more information.
    """
    return Array._new(np.moveaxis(x._array, source, destination), device=x.device)

# Note: The function name is different here (see also matrix_transpose).
# Unlike transpose(), the axes argument is required.
def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    return Array._new(np.transpose(x._array, axes), device=x.device)

@requires_api_version('2023.12')
def repeat(
    x: Array,
    repeats: int | Array,
    /,
    *,
    axis: int | None = None,
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
        if x.device != repeats.device:
            raise ValueError(f"Arrays from two different devices ({x.device} and {repeats.device}) can not be combined.")
    elif isinstance(repeats, int):
        repeats = asarray(repeats)
    else:
        raise TypeError("repeats must be an int or array")

    if repeats.dtype == uint64:
        # NumPy does not allow uint64 because can't be cast down to x.dtype
        # with 'safe' casting. However, repeats values larger than 2**63 are
        # infeasable, and even if they are present by mistake, this will
        # lead to underflow and an error.
        repeats = astype(repeats, int64)
    return Array._new(np.repeat(x._array, repeats._array, axis=axis), device=x.device)


# Note: the optional argument is called 'shape', not 'newshape'
def reshape(x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Array:
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

    return Array._new(reshaped, device=x.device)


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.roll <numpy.roll>`.

    See its docstring for more information.
    """
    return Array._new(np.roll(x._array, shift, axis=axis), device=x.device)


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.

    See its docstring for more information.
    """
    if axis is None:
        raise ValueError(
            "squeeze(..., axis=None is not supported. See "
            "https://github.com/data-apis/array-api/pull/100 for a discussion."
        )
    return Array._new(np.squeeze(x._array, axis=axis), device=x.device)


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    result_type(*arrays)
    if len({a.device for a in arrays}) > 1:
        raise ValueError("concat inputs must all be on the same device")
    result_device = arrays[0].device
    np_arrays = tuple(a._array for a in arrays)
    return Array._new(np.stack(np_arrays, axis=axis), device=result_device)


@requires_api_version('2023.12')
def tile(x: Array, repetitions: tuple[int, ...], /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.tile <numpy.tile>`.

    See its docstring for more information.
    """
    # Note: NumPy allows repetitions to be an int or array
    if not isinstance(repetitions, tuple):
        raise TypeError("repetitions must be a tuple")
    return Array._new(np.tile(x._array, repetitions), device=x.device)

# Note: this function is new
@requires_api_version('2023.12')
def unstack(x: Array, /, *, axis: int = 0) -> tuple[Array, ...]:
    if not (-x.ndim <= axis < x.ndim):
        raise ValueError("axis out of range")

    if axis < 0:
        axis += x.ndim

    slices = (slice(None),) * axis
    return tuple(x[slices + (i, ...)] for i in range(x.shape[axis]))
