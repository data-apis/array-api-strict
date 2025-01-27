import cmath
import pytest

from .._flags import set_array_api_strict_flags

import array_api_strict as xp

# sum, prod, and trace were changed in 2023.12 to not upcast floating-point dtypes
# with dtype=None
@pytest.mark.parametrize('func_name', ['sum', 'prod', 'trace'])
def test_sum_prod_trace_2022_12(func_name):
    set_array_api_strict_flags(api_version='2022.12')

    if func_name == 'trace':
        func = getattr(xp.linalg, func_name)
    else:
        func = getattr(xp, func_name)

    a_real = xp.asarray([[1., 2.], [3., 4.]], dtype=xp.float32)
    a_complex = xp.asarray([[1., 2.], [3., 4.]], dtype=xp.complex64)
    a_int = xp.asarray([[1, 2], [3, 4]], dtype=xp.int32)

    assert func(a_real).dtype == xp.float64
    assert func(a_complex).dtype == xp.complex128
    assert func(a_int).dtype == xp.int64

@pytest.mark.parametrize('func_name', ['sum', 'prod', 'trace'])
def test_sum_prod_trace_2023_12(func_name):
    a_real = xp.asarray([[1., 2.], [3., 4.]], dtype=xp.float32)
    a_complex = xp.asarray([[1., 2.], [3., 4.]], dtype=xp.complex64)
    a_int = xp.asarray([[1, 2], [3, 4]], dtype=xp.int32)

    if func_name == 'trace':
        func = getattr(xp.linalg, func_name)
    else:
        func = getattr(xp, func_name)

    assert func(a_real).dtype == xp.float32
    assert func(a_complex).dtype == xp.complex64
    assert func(a_int).dtype == xp.int64


# mean(complex-valued array) is allowed from 2024.12 onwards
def test_mean_complex():
    a = xp.asarray([1j, 2j, 3j])

    set_array_api_strict_flags(api_version='2023.12')
    with pytest.raises(TypeError):
        xp.mean(a)

    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version='2024.12')
    m = xp.mean(a)
    assert cmath.isclose(complex(m), 2j)

    # mean of integer arrays is still not allowed
    with pytest.raises(TypeError):
        xp.mean(xp.arange(3))

