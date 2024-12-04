"""Private helper routines.
"""

from ._flags import get_array_api_strict_flags
from ._dtypes import _dtype_categories

_py_scalars = (bool, int, float, complex)


def _maybe_normalize_py_scalars(x1, x2, dtype_category, func_name):

    flags = get_array_api_strict_flags()
    if flags["api_version"] < "2024.12":
        # scalars will fail at the call site
        return x1, x2

    _allowed_dtypes = _dtype_categories[dtype_category]

    if isinstance(x1, _py_scalars):
        if isinstance(x2, _py_scalars):
            raise TypeError(f"Two scalars not allowed, got {type(x1) = } and {type(x2) =}")
        # x2 must be an array
        if x2.dtype not in _allowed_dtypes:
            raise TypeError(f"Only {dtype_category} dtypes are allowed {func_name}. Got {x2.dtype}.")
        x1 = x2._promote_scalar(x1)

    elif isinstance(x2, _py_scalars):
        # x1 must be an array
        if x1.dtype not in _allowed_dtypes:
            raise TypeError(f"Only {dtype_category} dtypes are allowed {func_name}. Got {x1.dtype}.")
        x2 = x1._promote_scalar(x2)
    else:
        if x1.dtype not in _allowed_dtypes or x2.dtype not in _allowed_dtypes:
            raise TypeError(f"Only {dtype_category} dtypes are allowed {func_name}. "
                            f"Got {x1.dtype} and {x2.dtype}.")
    return x1, x2

