import pytest

import array_api_strict


@pytest.mark.parametrize(
    "func_name",
    (
        "fft",
        "ifft",
        "fftn",
        "ifftn",
        "irfft",
        "irfftn",
        "hfft",
        "fftshift",
        "ifftshift",
    ),
)
def test_fft_device_support_complex(func_name):
    func = getattr(array_api_strict.fft, func_name)
    x = array_api_strict.asarray(
        [1, 2.0],
        dtype=array_api_strict.complex64,
        device=array_api_strict.Device("device1"),
    )
    y = func(x)

    assert x.device == y.device


@pytest.mark.parametrize("func_name", ("rfft", "rfftn", "ihfft"))
def test_fft_device_support_real(func_name):
    func = getattr(array_api_strict.fft, func_name)
    x = array_api_strict.asarray([1, 2.0], device=array_api_strict.Device("device1"))
    y = func(x)

    assert x.device == y.device
