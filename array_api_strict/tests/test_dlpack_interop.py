"""Check how other libraries see arrays of array_api_strict via DLPack.

Tests that libraries like NumPy and PyTorch can import arrays from
array_api_strict via DLPack.

Only run if torch is available.
"""
import numpy as np
import pytest

import array_api_strict as xp
from .._devices import ALL_DEVICES, CPU_DEVICE

torch = pytest.importorskip("torch")


def test_export_from_cpu_device():
    x = xp.asarray([1, 2, 3], device=CPU_DEVICE)

    assert np.from_dlpack(x).tolist() == [1, 2, 3]
    assert torch.from_dlpack(x).tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "device", [device for device in ALL_DEVICES if device != CPU_DEVICE]
)
def test_export_from_other_devices(device):
    x = xp.asarray([1, 2, 3], device=device)

    with pytest.raises(BufferError):
        np.from_dlpack(x)
    with pytest.raises(BufferError):
        torch.from_dlpack(x)

    assert torch.from_dlpack(x.to_device(CPU_DEVICE)).tolist() == [1, 2, 3]


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_import_from_torch(device):
    # int32 is the widest integer every device supports
    x = xp.from_dlpack(torch.asarray([1, 2, 3], dtype=torch.int32), device=device)

    assert x.device == device
    assert xp.all(x == xp.asarray([1, 2, 3], dtype=xp.int32, device=device))
