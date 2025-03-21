from __future__ import annotations

import builtins
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

# Note: we wrap the NumPy dtype objects in a bare class, so that none of the
# additional methods and behaviors of NumPy dtype objects are exposed.


class DType:
    _np_dtype: np.dtype[Any]
    __slots__ = ("_np_dtype", "__weakref__")

    def __init__(self, np_dtype: npt.DTypeLike):
        self._np_dtype = np.dtype(np_dtype)

    def __repr__(self) -> str:
        return f"array_api_strict.{self._np_dtype.name}"

    def __eq__(self, other: object) -> builtins.bool:
        # See https://github.com/numpy/numpy/pull/25370/files#r1423259515.
        # Avoid the user error of array_api_strict.float32 == numpy.float32,
        # which gives False. Making == error is probably too egregious, so
        # warn instead.
        if isinstance(other, np.dtype) or (
            isinstance(other, type) and issubclass(other, np.generic)
        ):
            warnings.warn(
                """You are comparing a array_api_strict dtype against \
a NumPy native dtype object, but you probably don't want to do this. \
array_api_strict dtype objects compare unequal to their NumPy equivalents. \
Such cross-library comparison is not supported by the standard.""",
                stacklevel=2,
            )
        if not isinstance(other, DType):
            return NotImplemented
        return self._np_dtype == other._np_dtype

    def __hash__(self) -> int:
        # Note: this is not strictly required
        # (https://github.com/data-apis/array-api/issues/582), but makes the
        # dtype objects much easier to work with here and elsewhere if they
        # can be used as dict keys.
        return hash(self._np_dtype)


def _np_dtype(dtype: DType | None) -> np.dtype[Any] | None:
    return dtype._np_dtype if dtype is not None else None


int8 = DType("int8")
int16 = DType("int16")
int32 = DType("int32")
int64 = DType("int64")
uint8 = DType("uint8")
uint16 = DType("uint16")
uint32 = DType("uint32")
uint64 = DType("uint64")
float32 = DType("float32")
float64 = DType("float64")
complex64 = DType("complex64")
complex128 = DType("complex128")
# Note: This name is changed
bool = DType("bool")

_all_dtypes = (
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
_boolean_dtypes = (bool,)
_real_floating_dtypes = (float32, float64)
_floating_dtypes = (float32, float64, complex64, complex128)
_complex_floating_dtypes = (complex64, complex128)
_integer_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
_signed_integer_dtypes = (int8, int16, int32, int64)
_unsigned_integer_dtypes = (uint8, uint16, uint32, uint64)
_integer_or_boolean_dtypes = (
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
_real_numeric_dtypes = (
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
_numeric_dtypes = (
    float32,
    float64,
    complex64,
    complex128,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

_dtype_categories = {
    "all": _all_dtypes,
    "real numeric": _real_numeric_dtypes,
    "numeric": _numeric_dtypes,
    "integer": _integer_dtypes,
    "integer or boolean": _integer_or_boolean_dtypes,
    "boolean": _boolean_dtypes,
    "real floating-point": _real_floating_dtypes,
    "complex floating-point": _complex_floating_dtypes,
    "floating-point": _floating_dtypes,
}

_real_to_complex_map = {float32: complex64, float64: complex128}

# Note: the spec defines a restricted type promotion table compared to NumPy.
# In particular, cross-kind promotions like integer + float or boolean +
# integer are not allowed, even for functions that accept both kinds.
# Additionally, NumPy promotes signed integer + uint64 to float64, but this
# promotion is not allowed here. To be clear, Python scalar int objects are
# allowed to promote to floating-point dtypes, but only in array operators
# (see Array._promote_scalar) method in _array_object.py.
_promotion_table = {
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int16, int8): int16,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int32, int8): int32,
    (int32, int16): int32,
    (int32, int32): int32,
    (int32, int64): int64,
    (int64, int8): int64,
    (int64, int16): int64,
    (int64, int32): int64,
    (int64, int64): int64,
    (uint8, uint8): uint8,
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,
    (uint16, uint8): uint16,
    (uint16, uint16): uint16,
    (uint16, uint32): uint32,
    (uint16, uint64): uint64,
    (uint32, uint8): uint32,
    (uint32, uint16): uint32,
    (uint32, uint32): uint32,
    (uint32, uint64): uint64,
    (uint64, uint8): uint64,
    (uint64, uint16): uint64,
    (uint64, uint32): uint64,
    (uint64, uint64): uint64,
    (int8, uint8): int16,
    (int8, uint16): int32,
    (int8, uint32): int64,
    (int16, uint8): int16,
    (int16, uint16): int32,
    (int16, uint32): int64,
    (int32, uint8): int32,
    (int32, uint16): int32,
    (int32, uint32): int64,
    (int64, uint8): int64,
    (int64, uint16): int64,
    (int64, uint32): int64,
    (uint8, int8): int16,
    (uint16, int8): int32,
    (uint32, int8): int64,
    (uint8, int16): int16,
    (uint16, int16): int32,
    (uint32, int16): int64,
    (uint8, int32): int32,
    (uint16, int32): int32,
    (uint32, int32): int64,
    (uint8, int64): int64,
    (uint16, int64): int64,
    (uint32, int64): int64,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float32): float64,
    (float64, float64): float64,
    (complex64, complex64): complex64,
    (complex64, complex128): complex128,
    (complex128, complex64): complex128,
    (complex128, complex128): complex128,
    (float32, complex64): complex64,
    (float32, complex128): complex128,
    (float64, complex64): complex128,
    (float64, complex128): complex128,
    (complex64, float32): complex64,
    (complex64, float64): complex128,
    (complex128, float32): complex128,
    (complex128, float64): complex128,
    (bool, bool): bool,
}


def _result_type(type1: DType, type2: DType) -> DType:
    if (type1, type2) in _promotion_table:
        return _promotion_table[type1, type2]
    raise TypeError(f"{type1} and {type2} cannot be type promoted together")
