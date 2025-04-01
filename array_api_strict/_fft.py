from collections.abc import Sequence
from typing import Literal

import numpy as np

from ._array_object import ALL_DEVICES, Array, Device
from ._data_type_functions import astype
from ._dtypes import (
    DType,
    _complex_floating_dtypes,
    _floating_dtypes,
    _real_floating_dtypes,
    complex64,
    float32,
)
from ._flags import requires_extension


@requires_extension('fft')
def fft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.fft <numpy.fft.fft>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in fft")
    res = Array._new(np.fft.fft(x._array, n=n, axis=axis, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, complex64)
    return res

@requires_extension('fft')
def ifft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.ifft <numpy.fft.ifft>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in ifft")
    res = Array._new(np.fft.ifft(x._array, n=n, axis=axis, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, complex64)
    return res

@requires_extension('fft')
def fftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.fftn <numpy.fft.fftn>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in fftn")
    res = Array._new(np.fft.fftn(x._array, s=s, axes=axes, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, complex64)
    return res

@requires_extension('fft')
def ifftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.ifftn <numpy.fft.ifftn>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in ifftn")
    res = Array._new(np.fft.ifftn(x._array, s=s, axes=axes, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, complex64)
    return res

@requires_extension('fft')
def rfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.rfft <numpy.fft.rfft>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in rfft")
    res = Array._new(np.fft.rfft(x._array, n=n, axis=axis, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == float32:
        return astype(res, complex64)
    return res

@requires_extension('fft')
def irfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.irfft <numpy.fft.irfft>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in irfft")
    res = Array._new(np.fft.irfft(x._array, n=n, axis=axis, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, float32)
    return res

@requires_extension('fft')
def rfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.rfftn <numpy.fft.rfftn>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in rfftn")
    res = Array._new(np.fft.rfftn(x._array, s=s, axes=axes, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == float32:
        return astype(res, complex64)
    return res

@requires_extension('fft')
def irfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.irfftn <numpy.fft.irfftn>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in irfftn")
    res = Array._new(np.fft.irfftn(x._array, s=s, axes=axes, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, float32)
    return res

@requires_extension('fft')
def hfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.hfft <numpy.fft.hfft>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in hfft")
    res = Array._new(np.fft.hfft(x._array, n=n, axis=axis, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, float32)
    return res

@requires_extension('fft')
def ihfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.ihfft <numpy.fft.ihfft>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in ihfft")
    res = Array._new(np.fft.ihfft(x._array, n=n, axis=axis, norm=norm), device=x.device)
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == float32:
        return astype(res, complex64)
    return res

@requires_extension('fft')
def fftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.fftfreq <numpy.fft.fftfreq>`.

    See its docstring for more information.
    """
    if device is not None and device not in ALL_DEVICES:
        raise ValueError(f"Unsupported device {device!r}")
    if dtype and not dtype in _real_floating_dtypes:
        raise ValueError(f"`dtype` must be a real floating-point type. Got {dtype=}.")

    np_result = np.fft.fftfreq(n, d=d)
    if dtype:
        np_result = np_result.astype(dtype._np_dtype)
    return Array._new(np_result, device=device)

@requires_extension('fft')
def rfftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.rfftfreq <numpy.fft.rfftfreq>`.

    See its docstring for more information.
    """
    if device is not None and device not in ALL_DEVICES:
        raise ValueError(f"Unsupported device {device!r}")
    if dtype and not dtype in _real_floating_dtypes:
        raise ValueError(f"`dtype` must be a real floating-point type. Got {dtype=}.")

    np_result = np.fft.rfftfreq(n, d=d)
    if dtype:
        np_result = np_result.astype(dtype._np_dtype)
    return Array._new(np_result, device=device)

@requires_extension('fft')
def fftshift(x: Array, /, *, axes: int | Sequence[int] | None = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.fftshift <numpy.fft.fftshift>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in fftshift")
    return Array._new(np.fft.fftshift(x._array, axes=axes), device=x.device)

@requires_extension('fft')
def ifftshift(x: Array, /, *, axes: int | Sequence[int] | None = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.ifftshift <numpy.fft.ifftshift>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in ifftshift")
    return Array._new(np.fft.ifftshift(x._array, axes=axes), device=x.device)

__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
