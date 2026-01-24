from collections.abc import Callable
from functools import wraps
from types import NoneType

import numpy as np

from ._array_object import Array
from ._creation_functions import asarray
from ._data_type_functions import broadcast_to, iinfo
from ._dtypes import (
    _complex_floating_dtypes,
    _dtype_categories,
    _integer_dtypes,
    _numeric_dtypes,
    _real_floating_dtypes,
    _real_numeric_dtypes,
    _result_type,
)
from ._flags import requires_api_version
from ._helpers import _maybe_normalize_py_scalars


def _binary_ufunc_proto(x1, x2, dtype_category, func_name, np_func):
    """Base implementation of a binary function, `func_name`, defined for
    dtypes from `dtype_category`
    """
    x1, x2 = _maybe_normalize_py_scalars(x1, x2, dtype_category, func_name)

    if x1.device != x2.device:
        raise ValueError(
            f"Arrays from two different devices ({x1.device} and {x2.device}) can not be combined."
        )
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np_func(x1._array, x2._array), device=x1.device)


_docstring_template = """
Array API compatible wrapper for :py:func:`np.%s <numpy.%s>`.

See its docstring for more information.
"""


def _create_binary_func(func_name, dtype_category, np_func):
    def inner(x1, x2, /) -> Array:
        return _binary_ufunc_proto(x1, x2, dtype_category, func_name, np_func)
    return inner


# static type annotation for ArrayOrPythonScalar arguments given a category
# NB: keep the keys in sync with the _dtype_categories dict
_annotations = {
    "all": "complex | Array",
    "real numeric": "float | Array",
    "numeric": "complex | Array",
    "integer": "int | Array",
    "integer or boolean": "int | Array",
    "boolean": "bool | Array",
    "real floating-point": "float | Array",
    "complex floating-point": "complex | Array",
    "floating-point": "complex | Array",
}


# func_name: dtype_category (must match that from _dtypes.py)
_binary_funcs = {
    "add": "numeric",
    "atan2": "real floating-point",
    "bitwise_and": "integer or boolean",
    "bitwise_or": "integer or boolean",
    "bitwise_xor": "integer or boolean",
    "_bitwise_left_shift": "integer",  # leading underscore deliberate
    "_bitwise_right_shift": "integer",
    # XXX: copysign: real fp or numeric?
    "copysign": "real floating-point",
    "divide": "floating-point",
    "equal": "all",
    "greater": "real numeric",
    "greater_equal": "real numeric",
    "less": "real numeric",
    "less_equal": "real numeric",
    "not_equal": "all",
    "floor_divide": "real numeric",
    "hypot": "real floating-point",
    "logaddexp": "real floating-point",
    "logical_and": "boolean",
    "logical_or": "boolean",
    "logical_xor": "boolean",
    "maximum": "real numeric",
    "minimum": "real numeric",
    "multiply": "numeric",
    "nextafter": "real floating-point",
    "pow": "numeric",
    "remainder": "real numeric",
    "subtract": "numeric",
}


# map array-api-name : numpy-name
_numpy_renames = {
    "atan2": "arctan2",
    "_bitwise_left_shift": "left_shift",
    "_bitwise_right_shift": "right_shift",
    "pow": "power",
}


# create and attach functions to the module
for func_name, dtype_category in _binary_funcs.items():
    # sanity check
    assert dtype_category in _dtype_categories

    numpy_name = _numpy_renames.get(func_name, func_name)
    np_func = getattr(np, numpy_name)

    func = _create_binary_func(func_name, dtype_category, np_func)
    func.__name__ = func_name

    func.__doc__ = _docstring_template % (numpy_name, numpy_name)
    func.__annotations__['x1'] = _annotations[dtype_category]
    func.__annotations__['x2'] = _annotations[dtype_category]

    vars()[func_name] = func


copysign = requires_api_version('2023.12')(copysign)  # noqa: F821
hypot = requires_api_version('2023.12')(hypot)  # noqa: F821
maximum = requires_api_version('2023.12')(maximum)  # noqa: F821
minimum = requires_api_version('2023.12')(minimum)  # noqa: F821
nextafter = requires_api_version('2024.12')(nextafter)  # noqa: F821


def bitwise_left_shift(x1: int | Array, x2: int | Array, /) -> Array:
    is_negative = np.any(x2._array < 0) if isinstance(x2, Array) else x2 < 0
    if is_negative:
        raise ValueError("bitwise_left_shift(x1, x2) is only defined for x2 >= 0")
    return _bitwise_left_shift(x1, x2)   # noqa: F821
if _bitwise_left_shift.__doc__:  # noqa: F821
    bitwise_left_shift.__doc__ = _bitwise_left_shift.__doc__ # noqa: F821


def bitwise_right_shift(x1: int | Array, x2: int | Array, /) -> Array:
    is_negative = np.any(x2._array < 0) if isinstance(x2, Array) else x2 < 0
    if is_negative:
        raise ValueError("bitwise_left_shift(x1, x2) is only defined for x2 >= 0")
    return _bitwise_right_shift(x1, x2)   # noqa: F821
if _bitwise_right_shift.__doc__: # noqa: F821
    bitwise_right_shift.__doc__ = _bitwise_right_shift.__doc__   # noqa: F821


# clean up to not pollute the namespace
del func, _create_binary_func


def _create_unary_func(
    func_name: str, 
    dtype_category: str,
    np_func_name: str | None = None,
) -> Callable[[Array], Array]:
    allowed_dtypes = _dtype_categories[dtype_category]
    np_func_name = np_func_name or func_name
    np_func = getattr(np, np_func_name)

    def func(x: Array, /) -> Array:
        if not isinstance(x, Array):
            raise TypeError(f"Only Array objects are allowed; got {type(x)}")
        if x.dtype not in allowed_dtypes:
            raise TypeError(
                f"Only {dtype_category} dtypes are allowed in {func_name}; "
                f"got {x.dtype}."
            )
        return Array._new(np_func(x._array), device=x.device)

    func.__name__ = func_name
    func.__doc__ = _docstring_template % (np_func_name, np_func_name)
    return func


def _identity_if_integer(func: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Hack around NumPy 1.x behaviour for ceil, floor, and trunc
    vs. integer inputs
    """

    @wraps(func)
    def wrapper(x: Array, /) -> Array:
        if isinstance(x, Array) and x.dtype in _integer_dtypes:
            return x
        return func(x)
    
    return wrapper


abs = _create_unary_func("abs", "numeric")
acos = _create_unary_func("acos", "floating-point", "arccos")
acosh = _create_unary_func("acosh", "floating-point", "arccosh")
asin = _create_unary_func("asin", "floating-point", "arcsin")
asinh = _create_unary_func("asinh", "floating-point", "arcsinh")
atan = _create_unary_func("atan", "floating-point", "arctan")
atanh = _create_unary_func("atanh", "floating-point", "arctanh")
bitwise_invert = _create_unary_func("bitwise_invert", "integer or boolean", "invert")
ceil = _identity_if_integer(_create_unary_func("ceil", "real numeric"))
conj = _create_unary_func("conj", "numeric")
cos = _create_unary_func("cos", "floating-point", "cos")
cosh = _create_unary_func("cosh", "floating-point", "cosh")
exp = _create_unary_func("exp", "floating-point")
expm1 = _create_unary_func("expm1", "floating-point")
floor = _identity_if_integer(_create_unary_func("floor", "real numeric"))
imag = _create_unary_func("imag", "complex floating-point")
isfinite = _create_unary_func("isfinite", "numeric")
isinf = _create_unary_func("isinf", "numeric")
isnan = _create_unary_func("isnan", "numeric")
log = _create_unary_func("log", "floating-point")
log10 = _create_unary_func("log10", "floating-point")
log1p = _create_unary_func("log1p", "floating-point")
log2 = _create_unary_func("log2", "floating-point")
logical_not = _create_unary_func("logical_not", "boolean")
negative = _create_unary_func("negative", "numeric")
positive = _create_unary_func("positive", "numeric")
real = _create_unary_func("real", "numeric")
reciprocal = requires_api_version("2024.12")(
    _create_unary_func("reciprocal", "floating-point")
)
round = _create_unary_func("round", "numeric")
signbit = requires_api_version("2023.12")(
    _create_unary_func("signbit", "real floating-point")
)
sin = _create_unary_func("sin", "floating-point")
sinh = _create_unary_func("sinh", "floating-point")
sqrt = _create_unary_func("sqrt", "floating-point")
square = _create_unary_func("square", "numeric")
tan = _create_unary_func("tan", "floating-point")
tanh = _create_unary_func("tanh", "floating-point")
trunc = _identity_if_integer(_create_unary_func("trunc", "real numeric"))


# Note: min and max argument names are different and not optional in numpy.
@requires_api_version('2023.12')
def clip(
    x: Array,
    /,
    min: Array | float | None = None,
    max: Array | float | None = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.clip <numpy.clip>`.

    See its docstring for more information.
    """
    if not isinstance(x, Array):
        raise TypeError(f"Only Array objects are allowed; got {type(x)}")

    if (x.dtype not in _real_numeric_dtypes
        or isinstance(min, Array) and min.dtype not in _real_numeric_dtypes
        or isinstance(max, Array) and max.dtype not in _real_numeric_dtypes):
        raise TypeError("Only real numeric dtypes are allowed in clip")

    if min is max is None:
        return Array._new(x._array.copy(), device=x.device)

    for argname, arg in ("min", min), ("max", max):
        if isinstance(arg, Array):
            if x.device != arg.device:
                raise ValueError(
                    f"Arrays from two different devices ({x.device} and {arg.device}) "
                    "can not be combined."
                )
        # Disallow subclasses of Python scalars, e.g. np.float64
        elif type(arg) not in (int, float, NoneType):
            raise TypeError(
                f"{argname} must be None, int, float, or Array; got {type(arg)}"
            )

        # Mixed dtype kinds is implementation defined
        if (x.dtype in _integer_dtypes
            and (isinstance(arg, float) or
                isinstance(arg, Array) and arg.dtype in _real_floating_dtypes)):
            raise TypeError(f"{argname} must be integral when x is integral")
        if (x.dtype in _real_floating_dtypes
            and (isinstance(arg, int) or
                isinstance(arg, Array) and arg.dtype in _integer_dtypes)):
            raise TypeError(f"{arg} must be floating-point when x is floating-point")

    # Normalize to make the below logic simpler
    if min is not None:
        min = asarray(min)._array
    if max is not None:
        max = asarray(max)._array

    # min > max is implementation defined
    if min is not None and max is not None and np.any(min > max):
        raise ValueError("min must be less than or equal to max")

    # np.clip does type promotion but the array API clip requires that the
    # output have the same dtype as x. We do this instead of just downcasting
    # the result of xp.clip() to handle some corner cases better (e.g.,
    # avoiding uint64 -> float64 promotion).

    # Note: cases where min or max overflow (integer) or round (float) in the
    # wrong direction when downcasting to x.dtype are unspecified. This code
    # just does whatever NumPy does when it downcasts in the assignment, but
    # other behavior could be preferred, especially for integers. For example,
    # this code produces:

    # >>> clip(asarray(0, dtype=int8), asarray(128, dtype=int16), None)
    # -128

    # but an answer of 0 might be preferred. See
    # https://github.com/numpy/numpy/issues/24976 for more discussion on this issue.

    # At least handle the case of Python integers correctly (see
    # https://github.com/numpy/numpy/pull/26892).
    if type(min) is int and min <= iinfo(x.dtype).min:
        min = None
    if type(max) is int and max >= iinfo(x.dtype).max:
        max = None

    def _isscalar(a):
        return isinstance(a, (int, float, type(None)))

    min_shape = () if _isscalar(min) else min.shape
    max_shape = () if _isscalar(max) else max.shape

    result_shape = np.broadcast_shapes(x.shape, min_shape, max_shape)

    out = asarray(broadcast_to(x, result_shape), copy=True)._array
    device = x.device
    x = x._array

    if min is not None:
        a = np.broadcast_to(np.asarray(min), result_shape)
        ia = (out < a) | np.isnan(a)

        out[ia] = a[ia]
    if max is not None:
        b = np.broadcast_to(np.asarray(max), result_shape)
        ib = (out > b) | np.isnan(b)
        out[ib] = b[ib]
    return Array._new(out, device=device)


def sign(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sign <numpy.sign>`.

    See its docstring for more information.
    """
    if not isinstance(x, Array):
        raise TypeError(f"Only Array objects are allowed; got {type(x)}")
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    # Special treatment to work around non-compliant NumPy 1.x behaviour
    if x.dtype in _complex_floating_dtypes:
        _x = x._array
        _result = _x / np.abs(np.where(_x != 0, _x, np.asarray(1.0, dtype=_x.dtype)))
        return Array._new(_result, device=x.device)
    return Array._new(np.sign(x._array), device=x.device)
