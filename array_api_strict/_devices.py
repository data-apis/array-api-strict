from typing import Final
from enum import IntEnum

from ._dtypes import (
    DType, float32, float64, complex64, complex128, int64,
    _all_dtypes, _boolean_dtypes, _signed_integer_dtypes,
    _unsigned_integer_dtypes, _integer_dtypes, _real_floating_dtypes,
    _complex_floating_dtypes, _numeric_dtypes
)

_ALL_DEVICE_NAMES = ("CPU_DEVICE", "device1", "device2", "F32_device")

class Device:
    _device: Final[str]
    __slots__ = ("_device", "__weakref__")

    def __init__(self, device: str = "CPU_DEVICE"):
        if device not in _ALL_DEVICE_NAMES:
            raise ValueError(f"The device '{device}' is not a valid choice.")
        self._device = device

    def __repr__(self) -> str:
        return f"array_api_strict.Device('{self._device}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Device):
            return False
        return self._device == other._device

    def __hash__(self) -> int:
        return hash(("Device", self._device))

    def _supported_dtypes(self) -> list[DType]:
        # XXX useful? Unused ATM
        return list(dt for dt in _all_dtypes if device_supports_dtype(self, dt))


CPU_DEVICE = Device()
_F32_DEVICE = Device("F32_device")

ALL_DEVICES = (CPU_DEVICE, Device("device1"), Device("device2"), _F32_DEVICE)

class DLDeviceType(IntEnum):
    kDLCPU = 1
    kDLCUDA = 2


_DLPACK_DEVICE_FOR: Final[dict[Device, tuple[DLDeviceType, int]]] = {
    CPU_DEVICE: (DLDeviceType.kDLCPU, 0),
    Device("device1"): (DLDeviceType.kDLCUDA, 0),
    Device("device2"): (DLDeviceType.kDLCUDA, 1),
}

_DLPACK_DEVICE_TO_LOGICAL: Final[dict[tuple[int, int], Device]] = {
    (int(device_type), device_id): logical_device
    for logical_device, (device_type, device_id) in _DLPACK_DEVICE_FOR.items()
}


def _normalize_dl_device(device_type: IntEnum | int, device_id: int) -> tuple[int, int]:
    return (int(device_type), device_id)


def _device_from_dlpack_device(
    device_type: IntEnum | int, device_id: int
) -> Device:
    return _DLPACK_DEVICE_TO_LOGICAL.get(
        _normalize_dl_device(device_type, device_id), CPU_DEVICE
    )


def check_device(device: Device | None) -> None:
    if device is not None and not isinstance(device, Device):
        raise ValueError(f"Unsupported device {device!r}")

    if device is not None and device not in ALL_DEVICES:
        raise ValueError(f"Unsupported device {device!r}")


# Helpers for device-specific dtype support

def get_default_dtypes(device: Device | None = None) -> dict[str, Device]:
    if device == _F32_DEVICE:
        return {
            "real floating": float32,
            "complex floating": complex64,
            "integral": int64,
            "indexing": int64,
        }
    else:
        return {
            "real floating": float64,
            "complex floating": complex128,
            "integral": int64,
            "indexing": int64,
        }


def device_supports_dtype(device: Device | None, dtype: DType |None) -> bool:
    """True if `device` supports `dtype`, False otherwise."""
    # special-case F32_device
    if device == _F32_DEVICE:
        return dtype not in (float64, complex128)

    # All other devices support all dtypes
    return True


def _map_supported(dtypes: list[DType], device: Device) -> dict[str, DType]:
    return {
        dt._canonic_name: dt
        for dt in dtypes
        if device_supports_dtype(device, dt)
    }


# _info.dtypes() maps "kind" -> dict of {name: dtype}
# Note that "kinds" differ from "categories" above, per the spec.

_kind_to_dtypes = {
    None: _all_dtypes,
    "bool": _boolean_dtypes,
    "signed integer": _signed_integer_dtypes,
    "unsigned integer": _unsigned_integer_dtypes,
    "integral": _integer_dtypes,
    "real floating": _real_floating_dtypes,
    "complex floating": _complex_floating_dtypes,
    "numeric": _numeric_dtypes
}

