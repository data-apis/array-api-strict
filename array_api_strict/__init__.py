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
array-api-compat (https://github.com/data-apis/array-api-compat) package to
support the array API. Rather, it is intended to be used in the test suites of
consuming libraries to test their array API usage.

"""

__array_api_version__ = "2022.12"

__all__ = ["__array_api_version__"]

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
    conj,
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
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    real,
    remainder,
    round,
    sign,
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
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "remainder",
    "round",
    "sign",
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

# linalg is an extension in the array API spec, which is a sub-namespace. Only
# a subset of functions in it are imported into the top-level namespace.
from . import linalg

__all__ += ["linalg"]

from .linalg import matmul, tensordot, matrix_transpose, vecdot

__all__ += ["matmul", "tensordot", "matrix_transpose", "vecdot"]

from . import fft
__all__ += ["fft"]

from ._manipulation_functions import (
    concat,
    expand_dims,
    flip,
    permute_dims,
    reshape,
    roll,
    squeeze,
    stack,
)

__all__ += ["concat", "expand_dims", "flip", "permute_dims", "reshape", "roll", "squeeze", "stack"]

from ._searching_functions import argmax, argmin, nonzero, where

__all__ += ["argmax", "argmin", "nonzero", "where"]

from ._set_functions import unique_all, unique_counts, unique_inverse, unique_values

__all__ += ["unique_all", "unique_counts", "unique_inverse", "unique_values"]

from ._sorting_functions import argsort, sort

__all__ += ["argsort", "sort"]

from ._statistical_functions import max, mean, min, prod, std, sum, var

__all__ += ["max", "mean", "min", "prod", "std", "sum", "var"]

from ._utility_functions import all, any

__all__ += ["all", "any"]

from . import _version
__version__ = _version.get_versions()['version']
del _version
