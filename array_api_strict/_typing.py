"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

from __future__ import annotations

import sys
from typing import Any, Protocol, TypedDict, TypeVar

from ._dtypes import DType

_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...


if sys.version_info >= (3, 12):
    from collections.abc import Buffer as SupportsBufferProtocol
else:
    SupportsBufferProtocol = Any

PyCapsule = Any


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule: ...


Capabilities = TypedDict(
    "Capabilities",
    {
        "boolean indexing": bool,
        "data-dependent shapes": bool,
        "max dimensions": int,
    },
)

DefaultDataTypes = TypedDict(
    "DefaultDataTypes",
    {
        "real floating": DType,
        "complex floating": DType,
        "integral": DType,
        "indexing": DType,
    },
)


class DataTypes(TypedDict, total=False):
    bool: DType
    float32: DType
    float64: DType
    complex64: DType
    complex128: DType
    int8: DType
    int16: DType
    int32: DType
    int64: DType
    uint8: DType
    uint16: DType
    uint32: DType
    uint64: DType
