import pytest

import array_api_strict as xp


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
    func = getattr(xp.fft, func_name)
    x = xp.asarray(
        [1, 2.0],
        dtype=xp.complex64,
        device=xp.Device("device1"),
    )
    y = func(x)

    assert x.device == y.device


@pytest.mark.parametrize("func_name", ("rfft", "rfftn", "ihfft"))
def test_fft_device_support_real(func_name):
    func = getattr(xp.fft, func_name)
    x = xp.asarray([1, 2.0], device=xp.Device("device1"))
    y = func(x)

    assert x.device == y.device


@pytest.mark.parametrize("func_name", ("fftfreq", "rfftfreq"))
def test_fft_default_dtype(func_name):
    func = getattr(xp.fft, func_name)
    device = xp.Device("F32_device")
    res = func(3, device=device)
    assert res.device == device
    assert res.dtype == xp.__array_namespace_info__().default_dtypes(device=device)["real floating"]

    with pytest.raises(ValueError):
        func(3, device=device, dtype=xp.float64)


class TestF32Device:
    @pytest.mark.parametrize("dtype_str", ["float64", "complex128"])
    def test_f64_raises(self, dtype_str):
        f32_device = xp.Device("F32_device")
        dtype = getattr(xp, dtype_str)
        with pytest.raises(ValueError):
            xp.arange(3, device=f32_device, dtype=dtype)

    def test_info_no_f64(self):
        f32_device = xp.Device("F32_device")

        info = xp.__array_namespace_info__()
        all_dtypes = info.dtypes(device=f32_device)
        assert "float64" not in all_dtypes
        assert "complex128" not in all_dtypes

    def test_info_default_dtypes(self):
        f32_device = xp.Device("F32_device")
        info = xp.__array_namespace_info__()
        defaults = info.default_dtypes(device=f32_device)
        assert defaults["real floating"] == xp.float32
        assert defaults["complex floating"] == xp.complex64

        cpu_device = xp.Device()
        info = xp.__array_namespace_info__()
        defaults = info.default_dtypes(device=cpu_device)
        assert defaults["real floating"] == xp.float64
        assert defaults["complex floating"] == xp.complex128
