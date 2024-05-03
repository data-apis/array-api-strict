"""
array_api_strict is a strict, minimal implementation of the Python array
API (https://data-apis.org/array-api/latest/)

The purpose of array-api-strict is to provide an implementation of the array
API for consuming libraries to test against so they can be completely sure
their usage of the array API is portable.

It is *not* intended to be used by end-users. End-users of the array API
should just use their favorite array library (NumPy, CuPy, PyTorch, etc.) as
usual. It is also not intended to be used as a dependency by consuming
libraries. Consuming library code should use the
array-api-compat (https://data-apis.org/array-api-compat/) package to
support the array API. Rather, it is intended to be used in the test suites of
consuming libraries to test their array API usage.

"""

__all__ = []

# Warning: __array_api_version__ could change globally with
# set_array_api_strict_flags(). This should always be accessed as an
# attribute, like xp.__array_api_version__, or using
# array_api_strict.get_array_api_strict_flags()['api_version'].
from ._flags import API_VERSION as __array_api_version__

__all__ += ["__array_api_version__"]

from ._constants import e, inf, nan, pi, newaxis

__all__ += ["e", "inf", "nan", "pi", "newaxis"]

from ._creation_functions import (
    asarray,
    arange,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)

__all__ += [
    "asarray",
    "arange",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]

from ._data_type_functions import (
    astype,
    broadcast_arrays,
    broadcast_to,
    can_cast,
    finfo,
    isdtype,
    iinfo,
    result_type,
)

__all__ += [
    "astype",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "finfo",
    "isdtype",
    "iinfo",
    "result_type",
]

from ._dtypes import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
    bool,
)

__all__ += [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
]

from ._elementwise_functions import (
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_left_shift,
    bitwise_invert,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    clip,
    conj,
    copysign,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    hypot,
    imag,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    real,
    remainder,
    round,
    sign,
    signbit,
    sin,
    sinh,
    square,
    sqrt,
    subtract,
    tan,
    tanh,
    trunc,
)

__all__ += [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clip",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "hypot",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]

from ._indexing_functions import take

__all__ += ["take"]

from ._info import __array_namespace_info__

__all__ += [
    "__array_namespace_info__",
]

from ._linear_algebra_functions import matmul, tensordot, matrix_transpose, vecdot

__all__ += ["matmul", "tensordot", "matrix_transpose", "vecdot"]

from ._manipulation_functions import (
    concat,
    expand_dims,
    flip,
    moveaxis,
    permute_dims,
    repeat,
    reshape,
    roll,
    squeeze,
    stack,
    tile,
    unstack,
)

__all__ += ["concat", "expand_dims", "flip", "moveaxis", "permute_dims", "repeat", "reshape", "roll", "squeeze", "stack", "tile", "unstack"]

from ._searching_functions import argmax, argmin, nonzero, searchsorted, where

__all__ += ["argmax", "argmin", "nonzero", "searchsorted", "where"]

from ._set_functions import unique_all, unique_counts, unique_inverse, unique_values

__all__ += ["unique_all", "unique_counts", "unique_inverse", "unique_values"]

from ._sorting_functions import argsort, sort

__all__ += ["argsort", "sort"]

from ._statistical_functions import cumulative_sum, max, mean, min, prod, std, sum, var

__all__ += ["cumulative_sum", "max", "mean", "min", "prod", "std", "sum", "var"]

from ._utility_functions import all, any

__all__ += ["all", "any"]

# Helper functions that are not part of the standard

from ._flags import (
    set_array_api_strict_flags,
    get_array_api_strict_flags,
    reset_array_api_strict_flags,
    ArrayAPIStrictFlags,
)

__all__ += ['set_array_api_strict_flags', 'get_array_api_strict_flags', 'reset_array_api_strict_flags', 'ArrayAPIStrictFlags']

from . import _version
__version__ = _version.get_versions()['version']
del _version


# Extensions can be enabled or disabled dynamically. In order to make
# "array_api_strict.linalg" give an AttributeError when it is disabled, we
# use __getattr__. Note that linalg and fft are dynamically added and removed
# from __all__ in set_array_api_strict_flags.

def __getattr__(name):
    if name in ['linalg', 'fft']:
        if name in get_array_api_strict_flags()['enabled_extensions']:
            if name == 'linalg':
                from . import _linalg
                return _linalg
            elif name == 'fft':
                from . import _fft
                return _fft
        else:
            raise AttributeError(f"The {name!r} extension has been disabled for array_api_strict")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
