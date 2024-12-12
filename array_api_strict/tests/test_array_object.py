import operator
from builtins import all as all_

from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest

from .. import ones, asarray, result_type, all, equal
from .._array_object import Array, CPU_DEVICE, Device
from .._dtypes import (
    _all_dtypes,
    _boolean_dtypes,
    _real_floating_dtypes,
    _floating_dtypes,
    _complex_floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _real_numeric_dtypes,
    _numeric_dtypes,
    int8,
    int16,
    int32,
    int64,
    uint64,
    float64,
    bool as bool_,
)
from .._flags import set_array_api_strict_flags

import array_api_strict

def test_validate_index():
    # The indexing tests in the official array API test suite test that the
    # array object correctly handles the subset of indices that are required
    # by the spec. But the NumPy array API implementation specifically
    # disallows any index not required by the spec, via Array._validate_index.
    # This test focuses on testing that non-valid indices are correctly
    # rejected. See
    # https://data-apis.org/array-api/latest/API_specification/indexing.html
    # and the docstring of Array._validate_index for the exact indexing
    # behavior that should be allowed. This does not test indices that are
    # already invalid in NumPy itself because Array will generally just pass
    # such indices directly to the underlying np.ndarray.

    a = ones((3, 4))

    # Out of bounds slices are not allowed
    assert_raises(IndexError, lambda: a[:4])
    assert_raises(IndexError, lambda: a[:-4])
    assert_raises(IndexError, lambda: a[:3:-1])
    assert_raises(IndexError, lambda: a[:-5:-1])
    assert_raises(IndexError, lambda: a[4:])
    assert_raises(IndexError, lambda: a[-4:])
    assert_raises(IndexError, lambda: a[4::-1])
    assert_raises(IndexError, lambda: a[-4::-1])

    assert_raises(IndexError, lambda: a[...,:5])
    assert_raises(IndexError, lambda: a[...,:-5])
    assert_raises(IndexError, lambda: a[...,:5:-1])
    assert_raises(IndexError, lambda: a[...,:-6:-1])
    assert_raises(IndexError, lambda: a[...,5:])
    assert_raises(IndexError, lambda: a[...,-5:])
    assert_raises(IndexError, lambda: a[...,5::-1])
    assert_raises(IndexError, lambda: a[...,-5::-1])

    # Boolean indices cannot be part of a larger tuple index
    assert_raises(IndexError, lambda: a[a[:,0]==1,0])
    assert_raises(IndexError, lambda: a[a[:,0]==1,...])
    assert_raises(IndexError, lambda: a[..., a[0]==1])
    assert_raises(IndexError, lambda: a[[True, True, True]])
    assert_raises(IndexError, lambda: a[(True, True, True),])

    # Integer array indices are not allowed (except for 0-D)
    idx = asarray([[0, 1]])
    assert_raises(IndexError, lambda: a[idx])
    assert_raises(IndexError, lambda: a[idx,])
    assert_raises(IndexError, lambda: a[[0, 1]])
    assert_raises(IndexError, lambda: a[(0, 1), (0, 1)])
    assert_raises(IndexError, lambda: a[[0, 1]])

    # NumPy arrays are not allowed
    assert_raises(IndexError, lambda: a[np.ones((3, 4), dtype=bool)])
    assert_raises(IndexError, lambda: a[np.array([[0, 1]])])

    # Multiaxis indices must contain exactly as many indices as dimensions
    assert_raises(IndexError, lambda: a[()])
    assert_raises(IndexError, lambda: a[0,])
    assert_raises(IndexError, lambda: a[0])
    assert_raises(IndexError, lambda: a[:])

def test_promoted_scalar_inherits_device():
    device1 = Device("device1")
    x = asarray([1., 2, 3], device=device1)

    y = x ** 2

    assert y.device == device1

def test_operators():
    # For every operator, we test that it works for the required type
    # combinations and raises TypeError otherwise
    binary_op_dtypes = {
        "__add__": "numeric",
        "__and__": "integer_or_boolean",
        "__eq__": "all",
        "__floordiv__": "real numeric",
        "__ge__": "real numeric",
        "__gt__": "real numeric",
        "__le__": "real numeric",
        "__lshift__": "integer",
        "__lt__": "real numeric",
        "__mod__": "real numeric",
        "__mul__": "numeric",
        "__ne__": "all",
        "__or__": "integer_or_boolean",
        "__pow__": "numeric",
        "__rshift__": "integer",
        "__sub__": "numeric",
        "__truediv__": "floating",
        "__xor__": "integer_or_boolean",
    }
    # Recompute each time because of in-place ops
    def _array_vals():
        for d in _integer_dtypes:
            yield asarray(1, dtype=d)
        for d in _boolean_dtypes:
            yield asarray(False, dtype=d)
        for d in _floating_dtypes:
            yield asarray(1.0, dtype=d)


    BIG_INT = int(1e30)
    for op, dtypes in binary_op_dtypes.items():
        ops = [op]
        if op not in ["__eq__", "__ne__", "__le__", "__ge__", "__lt__", "__gt__"]:
            rop = "__r" + op[2:]
            iop = "__i" + op[2:]
            ops += [rop, iop]
        for s in [1, 1.0, 1j, BIG_INT, False]:
            for _op in ops:
                for a in _array_vals():
                    # Test array op scalar. From the spec, the following combinations
                    # are supported:

                    # - Python bool for a bool array dtype,
                    # - a Python int within the bounds of the given dtype for integer array dtypes,
                    # - a Python int or float for real floating-point array dtypes
                    # - a Python int, float, or complex for complex floating-point array dtypes

                    if ((dtypes == "all"
                         or dtypes == "numeric" and a.dtype in _numeric_dtypes
                         or dtypes == "real numeric" and a.dtype in _real_numeric_dtypes
                         or dtypes == "integer" and a.dtype in _integer_dtypes
                         or dtypes == "integer_or_boolean" and a.dtype in _integer_or_boolean_dtypes
                         or dtypes == "boolean" and a.dtype in _boolean_dtypes
                         or dtypes == "floating" and a.dtype in _floating_dtypes
                        )
                        # bool is a subtype of int, which is why we avoid
                        # isinstance here.
                        and (a.dtype in _boolean_dtypes and type(s) == bool
                             or a.dtype in _integer_dtypes and type(s) == int
                             or a.dtype in _real_floating_dtypes and type(s) in [float, int]
                             or a.dtype in _complex_floating_dtypes and type(s) in [complex, float, int]
                        )):
                        if a.dtype in _integer_dtypes and s == BIG_INT:
                            assert_raises(OverflowError, lambda: getattr(a, _op)(s))
                        else:
                            # Only test for no error
                            with suppress_warnings() as sup:
                                # ignore warnings from pow(BIG_INT)
                                sup.filter(RuntimeWarning,
                                           "invalid value encountered in power")
                                getattr(a, _op)(s)
                    else:
                        assert_raises(TypeError, lambda: getattr(a, _op)(s))

                # Test array op array.
                for _op in ops:
                    for x in _array_vals():
                        for y in _array_vals():
                            # See the promotion table in NEP 47 or the array
                            # API spec page on type promotion. Mixed kind
                            # promotion is not defined.
                            if (x.dtype == uint64 and y.dtype in [int8, int16, int32, int64]
                                or y.dtype == uint64 and x.dtype in [int8, int16, int32, int64]
                                or x.dtype in _integer_dtypes and y.dtype not in _integer_dtypes
                                or y.dtype in _integer_dtypes and x.dtype not in _integer_dtypes
                                or x.dtype in _boolean_dtypes and y.dtype not in _boolean_dtypes
                                or y.dtype in _boolean_dtypes and x.dtype not in _boolean_dtypes
                                or x.dtype in _floating_dtypes and y.dtype not in _floating_dtypes
                                or y.dtype in _floating_dtypes and x.dtype not in _floating_dtypes
                                ):
                                assert_raises(TypeError, lambda: getattr(x, _op)(y))
                            # Ensure in-place operators only promote to the same dtype as the left operand.
                            elif (
                                _op.startswith("__i")
                                and result_type(x.dtype, y.dtype) != x.dtype
                            ):
                                assert_raises(TypeError, lambda: getattr(x, _op)(y))
                            # Ensure only those dtypes that are required for every operator are allowed.
                            elif (dtypes == "all" and (x.dtype in _boolean_dtypes and y.dtype in _boolean_dtypes
                                                      or x.dtype in _numeric_dtypes and y.dtype in _numeric_dtypes)
                                or (dtypes == "real numeric" and x.dtype in _real_numeric_dtypes and y.dtype in _real_numeric_dtypes)
                                or (dtypes == "numeric" and x.dtype in _numeric_dtypes and y.dtype in _numeric_dtypes)
                                or dtypes == "integer" and x.dtype in _integer_dtypes and y.dtype in _integer_dtypes
                                or dtypes == "integer_or_boolean" and (x.dtype in _integer_dtypes and y.dtype in _integer_dtypes
                                                                       or x.dtype in _boolean_dtypes and y.dtype in _boolean_dtypes)
                                or dtypes == "boolean" and x.dtype in _boolean_dtypes and y.dtype in _boolean_dtypes
                                or dtypes == "floating" and x.dtype in _floating_dtypes and y.dtype in _floating_dtypes
                            ):
                                getattr(x, _op)(y)
                            else:
                                assert_raises(TypeError, lambda: getattr(x, _op)(y))

                            # finally, test that array op ndarray raises
                            # XXX: as long as there is __array__, __rop__s still
                            # return ndarrays
                            if not _op.startswith("__r"):
                                with assert_raises(TypeError):
                                    getattr(x, _op)(y._array)


    unary_op_dtypes = {
        "__abs__": "numeric",
        "__invert__": "integer_or_boolean",
        "__neg__": "numeric",
        "__pos__": "numeric",
    }
    for op, dtypes in unary_op_dtypes.items():
        for a in _array_vals():
            if (
                dtypes == "numeric"
                and a.dtype in _numeric_dtypes
                or dtypes == "integer_or_boolean"
                and a.dtype in _integer_or_boolean_dtypes
            ):
                # Only test for no error
                getattr(a, op)()
            else:
                assert_raises(TypeError, lambda: getattr(a, op)())

    # Finally, matmul() must be tested separately, because it works a bit
    # different from the other operations.
    def _matmul_array_vals():
        yield from _array_vals()
        for d in _all_dtypes:
            yield ones((3, 4), dtype=d)
            yield ones((4, 2), dtype=d)
            yield ones((4, 4), dtype=d)

    # Scalars always error
    for _op in ["__matmul__", "__rmatmul__", "__imatmul__"]:
        for s in [1, 1.0, False]:
            for a in _matmul_array_vals():
                if (type(s) in [float, int] and a.dtype in _floating_dtypes
                    or type(s) == int and a.dtype in _integer_dtypes):
                    # Type promotion is valid, but @ is not allowed on 0-D
                    # inputs, so the error is a ValueError
                    assert_raises(ValueError, lambda: getattr(a, _op)(s))
                else:
                    assert_raises(TypeError, lambda: getattr(a, _op)(s))

    for x in _matmul_array_vals():
        for y in _matmul_array_vals():
            if (x.dtype == uint64 and y.dtype in [int8, int16, int32, int64]
                or y.dtype == uint64 and x.dtype in [int8, int16, int32, int64]
                or x.dtype in _integer_dtypes and y.dtype not in _integer_dtypes
                or y.dtype in _integer_dtypes and x.dtype not in _integer_dtypes
                or x.dtype in _floating_dtypes and y.dtype not in _floating_dtypes
                or y.dtype in _floating_dtypes and x.dtype not in _floating_dtypes
                or x.dtype in _boolean_dtypes
                or y.dtype in _boolean_dtypes
                ):
                assert_raises(TypeError, lambda: x.__matmul__(y))
                assert_raises(TypeError, lambda: y.__rmatmul__(x))
                assert_raises(TypeError, lambda: x.__imatmul__(y))
            elif x.shape == () or y.shape == () or x.shape[1] != y.shape[0]:
                assert_raises(ValueError, lambda: x.__matmul__(y))
                assert_raises(ValueError, lambda: y.__rmatmul__(x))
                if result_type(x.dtype, y.dtype) != x.dtype:
                    assert_raises(TypeError, lambda: x.__imatmul__(y))
                else:
                    assert_raises(ValueError, lambda: x.__imatmul__(y))
            else:
                x.__matmul__(y)
                y.__rmatmul__(x)
                if result_type(x.dtype, y.dtype) != x.dtype:
                    assert_raises(TypeError, lambda: x.__imatmul__(y))
                elif y.shape[0] != y.shape[1]:
                    # This one fails because x @ y has a different shape from x
                    assert_raises(ValueError, lambda: x.__imatmul__(y))
                else:
                    x.__imatmul__(y)


def test_python_scalar_construtors():
    b = asarray(False)
    i = asarray(0)
    f = asarray(0.0)
    c = asarray(0j)

    assert bool(b) == False
    assert int(i) == 0
    assert float(f) == 0.0
    assert operator.index(i) == 0

    # bool/int/float/complex should only be allowed on 0-D arrays.
    assert_raises(TypeError, lambda: bool(asarray([False])))
    assert_raises(TypeError, lambda: int(asarray([0])))
    assert_raises(TypeError, lambda: float(asarray([0.0])))
    assert_raises(TypeError, lambda: complex(asarray([0j])))
    assert_raises(TypeError, lambda: operator.index(asarray([0])))

    # bool should work on all types of arrays
    assert bool(b) is bool(i) is bool(f) is bool(c) is False

    # int should fail on complex arrays
    assert int(b) == int(i) == int(f) == 0
    assert_raises(TypeError, lambda: int(c))

    # float should fail on complex arrays
    assert float(b) == float(i) == float(f) == 0.0
    assert_raises(TypeError, lambda: float(c))

    # complex should work on all types of arrays
    assert complex(b) == complex(i) == complex(f) == complex(c) == 0j

    # index should only work on integer arrays
    assert operator.index(i) == 0
    assert_raises(TypeError, lambda: operator.index(b))
    assert_raises(TypeError, lambda: operator.index(f))
    assert_raises(TypeError, lambda: operator.index(c))


def test_device_property():
    a = ones((3, 4))
    assert a.device == CPU_DEVICE
    assert not isinstance(a.device, str)

    assert all(equal(a.to_device(CPU_DEVICE), a))
    assert_raises(ValueError, lambda: a.to_device('cpu'))
    assert_raises(ValueError, lambda: a.to_device('gpu'))

    assert all(equal(asarray(a, device=CPU_DEVICE), a))
    assert_raises(ValueError, lambda: asarray(a, device='cpu'))
    assert_raises(ValueError, lambda: asarray(a, device='gpu'))

def test_array_properties():
    a = ones((1, 2, 3))
    b = ones((2, 3))
    assert_raises(ValueError, lambda: a.T)

    assert isinstance(b.T, Array)
    assert b.T.shape == (3, 2)

    assert isinstance(a.mT, Array)
    assert a.mT.shape == (1, 3, 2)
    assert isinstance(b.mT, Array)
    assert b.mT.shape == (3, 2)


def test_array_conversion():
    # Check that arrays on the CPU device can be converted to NumPy
    # but arrays on other devices can't. Note this is testing the logic in
    # __array__, which is only used in asarray when converting lists of
    # arrays.
    a = ones((2, 3))
    asarray([a])

    for device in ("device1", "device2"):
        a = ones((2, 3), device=array_api_strict.Device(device))
        with pytest.raises(RuntimeError, match="Can not convert array"):
            asarray([a])

def test__array__():
    # __array__ should work for now
    a = ones((2, 3))
    np.array(a)

    # Test the _allow_array private global flag for disabling it in the
    # future.
    from .. import _array_object
    original_value = _array_object._allow_array
    try:
        _array_object._allow_array = False
        a = ones((2, 3))
        with pytest.raises(ValueError, match="Conversion from an array_api_strict array to a NumPy ndarray is not supported"):
            np.array(a)
    finally:
        _array_object._allow_array = original_value

def test_allow_newaxis():
    a = ones(5)
    indexed_a = a[None, :]
    assert indexed_a.shape == (1, 5)

def test_disallow_flat_indexing_with_newaxis():
    a = ones((3, 3, 3))
    with pytest.raises(IndexError):
        a[None, 0, 0]

def test_disallow_mask_with_newaxis():
    a = ones((3, 3, 3))
    with pytest.raises(IndexError):
        a[None, asarray(True)]

@pytest.mark.parametrize("shape", [(), (5,), (3, 3, 3)])
@pytest.mark.parametrize("index", ["string", False, True])
def test_error_on_invalid_index(shape, index):
    a = ones(shape)
    with pytest.raises(IndexError):
        a[index]

def test_mask_0d_array_without_errors():
    a = ones(())
    a[asarray(True)]

@pytest.mark.parametrize(
    "i", [slice(5), slice(5, 0), asarray(True), asarray([0, 1])]
)
def test_error_on_invalid_index_with_ellipsis(i):
    a = ones((3, 3, 3))
    with pytest.raises(IndexError):
        a[..., i]
    with pytest.raises(IndexError):
        a[i, ...]

def test_array_keys_use_private_array():
    """
    Indexing operations convert array keys before indexing the internal array

    Fails when array_api array keys are not converted into NumPy-proper arrays
    in __getitem__(). This is achieved by passing array_api arrays with 0-sized
    dimensions, which NumPy-proper treats erroneously - not sure why!

    TODO: Find and use appropriate __setitem__() case.
    """
    a = ones((0, 0), dtype=bool_)
    assert a[a].shape == (0,)

    a = ones((0,), dtype=bool_)
    key = ones((0, 0), dtype=bool_)
    with pytest.raises(IndexError):
        a[key]

def test_array_namespace():
    a = ones((3, 3))
    assert a.__array_namespace__() == array_api_strict
    assert array_api_strict.__array_api_version__ == "2023.12"

    assert a.__array_namespace__(api_version=None) is array_api_strict
    assert array_api_strict.__array_api_version__ == "2023.12"

    assert a.__array_namespace__(api_version="2022.12") is array_api_strict
    assert array_api_strict.__array_api_version__ == "2022.12"

    assert a.__array_namespace__(api_version="2023.12") is array_api_strict
    assert array_api_strict.__array_api_version__ == "2023.12"

    with pytest.warns(UserWarning):
        assert a.__array_namespace__(api_version="2021.12") is array_api_strict
    assert array_api_strict.__array_api_version__ == "2021.12"

    with pytest.warns(UserWarning):
        assert a.__array_namespace__(api_version="2024.12") is array_api_strict
    assert array_api_strict.__array_api_version__ == "2024.12"

    pytest.raises(ValueError, lambda: a.__array_namespace__(api_version="2021.11"))
    pytest.raises(ValueError, lambda: a.__array_namespace__(api_version="2025.12"))

def test_iter():
    pytest.raises(TypeError, lambda: iter(asarray(3)))
    assert list(ones(3)) == [asarray(1.), asarray(1.), asarray(1.)]
    assert all_(isinstance(a, Array) for a in iter(ones(3)))
    assert all_(a.shape == () for a in iter(ones(3)))
    assert all_(a.dtype == float64 for a in iter(ones(3)))
    pytest.raises(TypeError, lambda: iter(ones((3, 3))))

@pytest.mark.parametrize("api_version", ['2021.12', '2022.12', '2023.12'])
def test_dlpack_2023_12(api_version):
    if api_version == '2021.12':
        with pytest.warns(UserWarning):
            set_array_api_strict_flags(api_version=api_version)
    else:
        set_array_api_strict_flags(api_version=api_version)

    a = asarray([1, 2, 3], dtype=int8)
    # Never an error
    a.__dlpack__()

    if api_version < '2023.12':
        pytest.raises(ValueError, lambda:
                      a.__dlpack__(dl_device=a.__dlpack_device__()))
        pytest.raises(ValueError, lambda:
                      a.__dlpack__(dl_device=None))
        pytest.raises(ValueError, lambda:
                      a.__dlpack__(max_version=(1, 0)))
        pytest.raises(ValueError, lambda:
                      a.__dlpack__(max_version=None))
        pytest.raises(ValueError, lambda:
                      a.__dlpack__(copy=False))
        pytest.raises(ValueError, lambda:
                      a.__dlpack__(copy=True))
        pytest.raises(ValueError, lambda:
                      a.__dlpack__(copy=None))
    elif np.lib.NumpyVersion(np.__version__) < '2.1.0':
        pytest.raises(NotImplementedError, lambda:
                      a.__dlpack__(dl_device=CPU_DEVICE))
        a.__dlpack__(dl_device=None)
        pytest.raises(NotImplementedError, lambda:
                      a.__dlpack__(max_version=(1, 0)))
        a.__dlpack__(max_version=None)
        pytest.raises(NotImplementedError, lambda:
                      a.__dlpack__(copy=False))
        pytest.raises(NotImplementedError, lambda:
                      a.__dlpack__(copy=True))
        a.__dlpack__(copy=None)
    else:
        a.__dlpack__(dl_device=a.__dlpack_device__())
        a.__dlpack__(dl_device=None)
        a.__dlpack__(max_version=(1, 0))
        a.__dlpack__(max_version=None)
        a.__dlpack__(copy=False)
        a.__dlpack__(copy=True)
        a.__dlpack__(copy=None)
