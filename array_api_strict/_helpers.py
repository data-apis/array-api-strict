"""Private helper routines."""

from ._array_object import Array
from ._dtypes import _dtype_categories
from ._flags import get_array_api_strict_flags

_PY_SCALARS = (bool, int, float, complex)


def _maybe_normalize_py_scalars(
    x1: Array | bool | int | float | complex,
    x2: Array | bool | int | float | complex,
    dtype_category: str,
    func_name: str,
) -> tuple[Array, Array]:
    flags = get_array_api_strict_flags()
    if flags["api_version"] < "2024.12":
        # scalars will fail at the call site
        return x1, x2  # type: ignore[return-value]

    _allowed_dtypes = _dtype_categories[dtype_category]

    # Disallow subclasses, e.g. np.float64 and np.complex128
    if type(x1) in _PY_SCALARS:
        if type(x2) in _PY_SCALARS:
            raise TypeError(f"Two scalars not allowed, got {type(x1) = } and {type(x2) =}")
        if not isinstance(x2, Array):
            raise TypeError(f"Argument is neither an Array nor a Python scalar: {type(x2)=} ")
        if x2.dtype not in _allowed_dtypes:
            raise TypeError(f"Only {dtype_category} dtypes are allowed {func_name}. Got {x2.dtype}.")
        x1 = x2._promote_scalar(x1)

    elif type(x2) in _PY_SCALARS:
        if not isinstance(x1, Array):
            raise TypeError(f"Argument is neither an Array nor a Python scalar: {type(x2)=} ")
        if x1.dtype not in _allowed_dtypes:
            raise TypeError(f"Only {dtype_category} dtypes are allowed {func_name}. Got {x1.dtype}.")
        x2 = x1._promote_scalar(x2)
    else:
        if not isinstance(x1, Array) or not isinstance(x2, Array):
            raise TypeError(f"Argument(s) are neither Array nor Python scalars: {type(x1)=} and {type(x2)=}")

        if x1.dtype not in _allowed_dtypes or x2.dtype not in _allowed_dtypes:
            raise TypeError(f"Only {dtype_category} dtypes are allowed in {func_name}(...). "
                            f"Got {x1.dtype} and {x2.dtype}.")
    return x1, x2
