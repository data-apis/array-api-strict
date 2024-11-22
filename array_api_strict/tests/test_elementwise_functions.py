from inspect import signature, getmodule

from pytest import raises as assert_raises
from numpy.testing import suppress_warnings

import pytest

from .. import asarray, _elementwise_functions
from .._elementwise_functions import bitwise_left_shift, bitwise_right_shift
from .._dtypes import (
    _dtype_categories,
    _boolean_dtypes,
    _floating_dtypes,
    _integer_dtypes,
    int8,
    int16,
    int32,
    int64,
    uint64,
)
from .._flags import set_array_api_strict_flags

from .test_array_object import _check_op_array_scalar, BIG_INT

import array_api_strict


def nargs(func):
    """Count number of 'array' arguments a function takes."""
    positional_only = 0
    for param in signature(func).parameters.values():
        if param.kind == param.POSITIONAL_ONLY:
            positional_only += 1
    return positional_only


elementwise_function_input_types = {
    "abs": "numeric",
    "acos": "floating-point",
    "acosh": "floating-point",
    "add": "numeric",
    "asin": "floating-point",
    "asinh": "floating-point",
    "atan": "floating-point",
    "atan2": "real floating-point",
    "atanh": "floating-point",
    "bitwise_and": "integer or boolean",
    "bitwise_invert": "integer or boolean",
    "bitwise_left_shift": "integer",
    "bitwise_or": "integer or boolean",
    "bitwise_right_shift": "integer",
    "bitwise_xor": "integer or boolean",
    "ceil": "real numeric",
    "clip": "real numeric",
    "conj": "complex floating-point",
    "copysign": "real floating-point",
    "cos": "floating-point",
    "cosh": "floating-point",
    "divide": "floating-point",
    "equal": "all",
    "exp": "floating-point",
    "expm1": "floating-point",
    "floor": "real numeric",
    "floor_divide": "real numeric",
    "greater": "real numeric",
    "greater_equal": "real numeric",
    "hypot": "real floating-point",
    "imag": "complex floating-point",
    "isfinite": "numeric",
    "isinf": "numeric",
    "isnan": "numeric",
    "less": "real numeric",
    "less_equal": "real numeric",
    "log": "floating-point",
    "logaddexp": "real floating-point",
    "log10": "floating-point",
    "log1p": "floating-point",
    "log2": "floating-point",
    "logical_and": "boolean",
    "logical_not": "boolean",
    "logical_or": "boolean",
    "logical_xor": "boolean",
    "maximum": "real numeric",
    "minimum": "real numeric",
    "multiply": "numeric",
    "negative": "numeric",
    "nextafter": "real floating-point",
    "not_equal": "all",
    "positive": "numeric",
    "pow": "numeric",
    "real": "complex floating-point",
    "reciprocal": "floating-point",
    "remainder": "real numeric",
    "round": "numeric",
    "sign": "numeric",
    "signbit": "real floating-point",
    "sin": "floating-point",
    "sinh": "floating-point",
    "sqrt": "floating-point",
    "square": "numeric",
    "subtract": "numeric",
    "tan": "floating-point",
    "tanh": "floating-point",
    "trunc": "real numeric",
}


def test_nargs():
    # Explicitly check number of arguments for a few functions
    assert nargs(array_api_strict.logaddexp) == 2
    assert nargs(array_api_strict.atan2) == 2
    assert nargs(array_api_strict.clip) == 1

    # All elementwise functions take one or two array arguments
    # if not, it is probably a bug in `nargs` or the definition
    # of the function (missing trailing `, /`).
    for func_name in elementwise_function_input_types:
        func = getattr(_elementwise_functions, func_name)
        assert nargs(func) in (1, 2)


def test_missing_functions():
    # Ensure the above dictionary is complete.
    import array_api_strict._elementwise_functions as mod
    mod_funcs = [n for n in dir(mod) if getmodule(getattr(mod, n)) is mod]
    mod_funcs = [n for n in mod_funcs if not n.startswith("_")]
    assert set(mod_funcs) == set(elementwise_function_input_types)


def test_function_device_persists():
    # Test that the device of the input and output array are the same
    def _array_vals(dtypes):
        for d in dtypes:
            yield asarray(1., dtype=d)

    # Use the latest version of the standard so all functions are included
    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version="2024.12")

    for func_name, types in elementwise_function_input_types.items():
        dtypes = _dtype_categories[types]
        func = getattr(_elementwise_functions, func_name)

        for x in _array_vals(dtypes):
            if nargs(func) == 2:
                # This way we don't have to deal with incompatible
                # types of the two arguments.
                r = func(x, x)
                assert r.device == x.device

            else:
                # `atanh` needs a slightly different input value from
                # everyone else
                if func_name == "atanh":
                    x -= 0.1
                r = func(x)
                assert r.device == x.device


def test_function_types():
    # Test that every function accepts only the required input types. We only
    # test the negative cases here (error). The positive cases are tested in
    # the array API test suite.

    def _array_vals():
        for d in _integer_dtypes:
            yield asarray(1, dtype=d)
        for d in _boolean_dtypes:
            yield asarray(False, dtype=d)
        for d in _floating_dtypes:
            yield asarray(1.0, dtype=d)

    # Use the latest version of the standard so all functions are included
    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version="2024.12")

    for x in _array_vals():
        for func_name, types in elementwise_function_input_types.items():
            dtypes = _dtype_categories[types]
            func = getattr(_elementwise_functions, func_name)
            if nargs(func) == 2:
                for y in _array_vals():
                    # Disallow dtypes that aren't type promotable
                    if (x.dtype == uint64 and y.dtype in [int8, int16, int32, int64]
                         or y.dtype == uint64 and x.dtype in [int8, int16, int32, int64]
                         or x.dtype in _integer_dtypes and y.dtype not in _integer_dtypes
                         or y.dtype in _integer_dtypes and x.dtype not in _integer_dtypes
                         or x.dtype in _boolean_dtypes and y.dtype not in _boolean_dtypes
                         or y.dtype in _boolean_dtypes and x.dtype not in _boolean_dtypes
                         or x.dtype in _floating_dtypes and y.dtype not in _floating_dtypes
                         or y.dtype in _floating_dtypes and x.dtype not in _floating_dtypes
                         ):
                        assert_raises(TypeError, func, x, y)
                    if x.dtype not in dtypes or y.dtype not in dtypes:
                        assert_raises(TypeError, func, x, y)
            else:
                if x.dtype not in dtypes:
                    assert_raises(TypeError, func, x)


def test_bitwise_shift_error():
    # bitwise shift functions should raise when the second argument is negative
    assert_raises(
        ValueError, lambda: bitwise_left_shift(asarray([1, 1]), asarray([1, -1]))
    )
    assert_raises(
        ValueError, lambda: bitwise_right_shift(asarray([1, 1]), asarray([1, -1]))
    )



def test_scalars():
    # mirror test_array_object.py::test_operators()
    #
    # Also check that binary functions accept (array, scalar) and (scalar, array)
    # arguments, and reject (scalar, scalar) arguments.

    # Use the latest version of the standard so that scalars are actually allowed
    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version="2024.12")

    def _array_vals():
        for d in _integer_dtypes:
            yield asarray(1, dtype=d)
        for d in _boolean_dtypes:
            yield asarray(False, dtype=d)
        for d in _floating_dtypes:
            yield asarray(1.0, dtype=d)


    for func_name, dtypes in elementwise_function_input_types.items():
        func = getattr(_elementwise_functions, func_name)
        if nargs(func) != 2:
            continue

        for s in [1, 1.0, 1j, BIG_INT, False]:
            for a in _array_vals():
                for func1 in [lambda s: func(a, s), lambda s: func(s, a)]:
                    allowed = _check_op_array_scalar(dtypes, a, s, func1, func_name)

                    # only check `func(array, scalar) == `func(array, array)` if
                    # the former is legal under the promotion rules
                    if allowed:
                        conv_scalar = asarray(s, dtype=a.dtype)

                        with suppress_warnings() as sup:
                            # ignore warnings from pow(BIG_INT)
                            sup.filter(RuntimeWarning,
                                       "invalid value encountered in power")
                            assert func(s, a) == func(conv_scalar, a)
                            assert func(a, s) == func(a, conv_scalar)

                        with assert_raises(TypeError):
                            func(s, s)


