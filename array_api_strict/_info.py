from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Union, Tuple, List
    from ._typing import device, DefaultDataTypes, DataTypes, Capabilities, Info

from ._array_object import CPU_DEVICE
from ._flags import get_array_api_strict_flags, requires_api_version
from ._dtypes import bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, complex64, complex128

@requires_api_version('2023.12')
def __array_namespace_info__() -> Info:
    import array_api_strict._info
    return array_api_strict._info

@requires_api_version('2023.12')
def capabilities() -> Capabilities:
    flags = get_array_api_strict_flags()
    return {"boolean indexing": flags['boolean_indexing'],
            "data-dependent shapes": flags['data_dependent_shapes'],
            }

@requires_api_version('2023.12')
def default_device() -> device:
    return CPU_DEVICE

@requires_api_version('2023.12')
def default_dtypes(
    *,
    device: Optional[device] = None,
) -> DefaultDataTypes:
    return {
        "real floating": float64,
        "complex floating": complex128,
        "integral": int64,
        "indexing": int64,
    }

@requires_api_version('2023.12')
def dtypes(
    *,
    device: Optional[device] = None,
    kind: Optional[Union[str, Tuple[str, ...]]] = None,
) -> DataTypes:
    if kind is None:
        return {
            "bool": bool,
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
    if kind == "bool":
        return {"bool": bool}
    if kind == "signed integer":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
        }
    if kind == "unsigned integer":
        return {
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
        }
    if kind == "integral":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
        }
    if kind == "real floating":
        return {
            "float32": float32,
            "float64": float64,
        }
    if kind == "complex floating":
        return {
            "complex64": complex64,
            "complex128": complex128,
        }
    if kind == "numeric":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
    if isinstance(kind, tuple):
        res = {}
        for k in kind:
            res.update(dtypes(kind=k))
        return res
    raise ValueError(f"unsupported kind: {kind!r}")

@requires_api_version('2023.12')
def devices() -> List[device]:
    return [CPU_DEVICE]

__all__ = [
    "capabilities",
    "default_device",
    "default_dtypes",
    "devices",
    "dtypes",
]
