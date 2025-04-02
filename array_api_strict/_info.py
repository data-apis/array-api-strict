import numpy as np

from . import _dtypes as dt
from ._array_object import ALL_DEVICES, CPU_DEVICE, Device
from ._flags import get_array_api_strict_flags, requires_api_version
from ._typing import Capabilities, DataTypes, DefaultDataTypes


@requires_api_version('2023.12')
class __array_namespace_info__:
    @requires_api_version('2023.12')
    def capabilities(self) -> Capabilities:
        flags = get_array_api_strict_flags()
        res: Capabilities = {  # type: ignore[typeddict-item]
            "boolean indexing": flags['boolean_indexing'],
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
    def default_device(self) -> Device:
        return CPU_DEVICE

    @requires_api_version('2023.12')
    def default_dtypes(
        self,
        *,
        device: Device | None = None,
    ) -> DefaultDataTypes:
        return {
            "real floating": dt.float64,
            "complex floating": dt.complex128,
            "integral": dt.int64,
            "indexing": dt.int64,
        }

    @requires_api_version('2023.12')
    def dtypes(
        self,
        *,
        device: Device | None = None,
        kind: str | tuple[str, ...] | None = None,
    ) -> DataTypes:
        if kind is None:
            return {
                "bool": dt.bool,
                "int8": dt.int8,
                "int16": dt.int16,
                "int32": dt.int32,
                "int64": dt.int64,
                "uint8": dt.uint8,
                "uint16": dt.uint16,
                "uint32": dt.uint32,
                "uint64": dt.uint64,
                "float32": dt.float32,
                "float64": dt.float64,
                "complex64": dt.complex64,
                "complex128": dt.complex128,
            }
        if kind == "bool":
            return {"bool": dt.bool}
        if kind == "signed integer":
            return {
                "int8": dt.int8,
                "int16": dt.int16,
                "int32": dt.int32,
                "int64": dt.int64,
            }
        if kind == "unsigned integer":
            return {
                "uint8": dt.uint8,
                "uint16": dt.uint16,
                "uint32": dt.uint32,
                "uint64": dt.uint64,
            }
        if kind == "integral":
            return {
                "int8": dt.int8,
                "int16": dt.int16,
                "int32": dt.int32,
                "int64": dt.int64,
                "uint8": dt.uint8,
                "uint16": dt.uint16,
                "uint32": dt.uint32,
                "uint64": dt.uint64,
            }
        if kind == "real floating":
            return {
                "float32": dt.float32,
                "float64": dt.float64,
            }
        if kind == "complex floating":
            return {
                "complex64": dt.complex64,
                "complex128": dt.complex128,
            }
        if kind == "numeric":
            return {
                "int8": dt.int8,
                "int16": dt.int16,
                "int32": dt.int32,
                "int64": dt.int64,
                "uint8": dt.uint8,
                "uint16": dt.uint16,
                "uint32": dt.uint32,
                "uint64": dt.uint64,
                "float32": dt.float32,
                "float64": dt.float64,
                "complex64": dt.complex64,
                "complex128": dt.complex128,
            }
        if isinstance(kind, tuple):
            res: DataTypes = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    @requires_api_version('2023.12')
    def devices(self) -> list[Device]:
        return list(ALL_DEVICES)
