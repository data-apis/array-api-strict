from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Optional, Union, Tuple, List
    from ._typing import device, DefaultDataTypes, DataTypes, Capabilities

from ._array_object import ALL_DEVICES, CPU_DEVICE
from ._flags import get_array_api_strict_flags, requires_api_version
from ._dtypes import bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, complex64, complex128

@requires_api_version('2023.12')
class __array_namespace_info__:
    @requires_api_version('2023.12')
    def capabilities(self) -> Capabilities:
        flags = get_array_api_strict_flags()
        res = {"boolean indexing": flags['boolean_indexing'],
                "data-dependent shapes": flags['data_dependent_shapes'],
                }
        if flags['api_version'] >= '2024.12':
            # maxdims is 32 for NumPy 1.x and 64 for NumPy 2.0. Eventually we will
            # drop support for NumPy 1 but for now, just compute the number
            # directly
            for i in range(1, 100):
                try:
                    np.zeros((1,)*i)
                except ValueError:
                    maxdims = i - 1
                    break
            else:
                raise RuntimeError("Could not get max dimensions (this is a bug in array-api-strict)")
            res['max dimensions'] = maxdims
        return res

    @requires_api_version('2023.12')
    def default_device(self) -> device:
        return CPU_DEVICE

    @requires_api_version('2023.12')
    def default_dtypes(
        self,
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
        self,
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
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    @requires_api_version('2023.12')
    def devices(self) -> List[device]:
        return list(ALL_DEVICES)
