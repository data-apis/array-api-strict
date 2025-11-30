import warnings
from inspect import signature, getmodule

import numpy as np
import pytest


from .. import asarray, _elementwise_functions
from .._array_object import ALL_DEVICES, CPU_DEVICE, Device
from .._elementwise_functions import bitwise_left_shift, bitwise_right_shift
from .._dtypes import (
    _dtype_categories,
    _boolean_dtypes,
    _floating_dtypes,
    _integer_dtypes,
    bool as xp_bool,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint64,
)
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
    "conj": "numeric",
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
    "real": "numeric",
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


elementwise_binary_function_names = [
    func_name
    for func_name in elementwise_function_input_types
    if nargs(getattr(_elementwise_functions, func_name)) == 2
]


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


@pytest.mark.parametrize("device", ALL_DEVICES)
@pytest.mark.parametrize("func_name,types", elementwise_function_input_types.items())
def test_elementwise_function_device_persists(func_name, types, device):
    """Test that the device of the input and output array are the same"""
    def _array_vals(dtypes):
        for dtype in dtypes:
            yield asarray(1., dtype=dtype, device=device)

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


@pytest.mark.parametrize("func_name", elementwise_binary_function_names)
def test_elementwise_function_device_mismatch(func_name):
    func = getattr(_elementwise_functions, func_name)
    dtypes = elementwise_function_input_types[func_name]
    if dtypes in ("floating-point", "real floating-point"):
        dtype = float64
    elif dtypes == "boolean":
        dtype = xp_bool
    else:
        dtype = int64

    a = asarray(1, dtype=dtype, device=CPU_DEVICE)
    b = asarray(1, dtype=dtype, device=Device("device1"))
    _ = func(a, a)
    with pytest.raises(ValueError, match="different devices"):
        func(a, b)


@pytest.mark.parametrize("func_name", elementwise_function_input_types)
def test_elementwise_function_numpy_scalars(func_name):
    """
    Test that NumPy scalars (np.generic) are explicitly disallowed.

    This must notably include np.float64 and np.complex128, which are
    subclasses of float and complex respectively, so they need
    special treatment in order to be rejected.
    """
    func = getattr(_elementwise_functions, func_name)
    dtypes = elementwise_function_input_types[func_name]
    xp_dtypes = _dtype_categories[dtypes]
    np_dtypes = [dtype._np_dtype for dtype in xp_dtypes]

    value = 0.5 if func_name == "atanh" else 1
    for xp_dtype in xp_dtypes:
        for np_dtype in np_dtypes:
            a = asarray(value, dtype=xp_dtype, device=CPU_DEVICE)
            b = np.asarray(value, dtype=np_dtype)[()]

            if nargs(func) == 2:
                _ = func(a, a)
                with pytest.raises(TypeError, match="neither Array nor Python scalars"):
                    func(a, b)
                with pytest.raises(TypeError, match="neither Array nor Python scalars"):
                    func(b, a)
            else:
                _ = func(a)
                with pytest.raises(TypeError, match="allowed"):
                    func(b)


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
                        with pytest.raises(TypeError):
                            func(x, y)
                    if x.dtype not in dtypes or y.dtype not in dtypes:
                        with pytest.raises(TypeError):
                            func(x, y)
            else:
                if x.dtype not in dtypes:
                    with pytest.raises(TypeError):
                        func(x)


def test_bitwise_shift_error():
    # bitwise shift functions should raise when the second argument is negative
    with pytest.raises(ValueError):
        bitwise_left_shift(asarray([1, 1]), asarray([1, -1]))
    with pytest.raises(ValueError):
        bitwise_right_shift(asarray([1, 1]), asarray([1, -1]))


def test_scalars():
    # mirror test_array_object.py::test_operators()
    #
    # Also check that binary functions accept (array, scalar) and (scalar, array)
    # arguments, and reject (scalar, scalar) arguments.

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

        nocomplex = [
            'atan2', 'copysign', 'floor_divide', 'hypot', 'logaddexp',  'nextafter',
            'remainder',
            'greater', 'less', 'greater_equal', 'less_equal', 'maximum', 'minimum',
        ]

        for s in [1, 1.0, 1j, BIG_INT, False]:
            for a in _array_vals():
                for func1 in [lambda s: func(a, s), lambda s: func(s, a)]:

                    if func_name in nocomplex and type(s) == complex:
                        allowed = False
                    else:
                        allowed = _check_op_array_scalar(dtypes, a, s, func1, func_name)

                    # only check `func(array, scalar) == `func(array, array)` if
                    # the former is legal under the promotion rules
                    if allowed:
                        conv_scalar = a._promote_scalar(s)

                        with warnings.catch_warnings():
                            # ignore warnings from pow(BIG_INT)
                            warnings.filterwarnings(
                                "ignore", category=RuntimeWarning,
                                message="invalid value encountered in power"
                            )

                            assert func(s, a) == func(conv_scalar, a)
                            assert func(a, s) == func(a, conv_scalar)

                        with pytest.raises(TypeError):
                            func(s, s)
