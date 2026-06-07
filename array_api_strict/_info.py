import numpy as np

from . import _devices
from ._devices import ALL_DEVICES, CPU_DEVICE, Device
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
        return _devices.get_default_dtypes(device)

    @requires_api_version('2023.12')
    def dtypes(
        self,
        *,
        device: Device | None = None,
        kind: str | tuple[str, ...] | None = None,
    ) -> DataTypes:
        if device is None:
            device = CPU_DEVICE
        if isinstance(kind, type(None) | str):

            try:
                dtypes = _devices._kind_to_dtypes[kind]
            except KeyError:
                raise ValueError(f"unsupported kind: {kind!r}")  
            res = _devices._map_supported(dtypes, device)
            return res

        elif isinstance(kind, tuple):
            res: DataTypes = {}
            for k in kind:
                res.update(self.dtypes(kind=k, device=device))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    @requires_api_version('2023.12')
    def devices(self) -> tuple[Device]:
        if get_array_api_strict_flags()['api_version'] < '2025.12':
            return list(ALL_DEVICES)
        else:
            return tuple(ALL_DEVICES)
