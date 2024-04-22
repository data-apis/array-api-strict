from .._flags import (set_array_api_strict_flags, get_array_api_strict_flags,
                      reset_array_api_strict_flags)

from .. import (asarray, unique_all, unique_counts, unique_inverse,
                unique_values, nonzero)

import array_api_strict as xp

import pytest

def test_flags():
    # Test defaults
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2022.12',
        'data_dependent_shapes': True,
        'enabled_extensions': ('linalg', 'fft'),
    }

    # Test setting flags
    set_array_api_strict_flags(data_dependent_shapes=False)
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2022.12',
        'data_dependent_shapes': False,
        'enabled_extensions': ('linalg', 'fft'),
    }
    set_array_api_strict_flags(enabled_extensions=('fft',))
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2022.12',
        'data_dependent_shapes': False,
        'enabled_extensions': ('fft',),
    }
    # Make sure setting the version to 2021.12 disables fft and issues a
    # warning.
    with pytest.warns(UserWarning) as record:
        set_array_api_strict_flags(api_version='2021.12')
    assert len(record) == 1
    assert '2021.12' in str(record[0].message)
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2021.12',
        'data_dependent_shapes': False,
        'enabled_extensions': (),
    }
    reset_array_api_strict_flags()

    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version='2021.12')
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2021.12',
        'data_dependent_shapes': True,
        'enabled_extensions': ('linalg',),
    }
    reset_array_api_strict_flags()

    # 2023.12 should issue a warning
    with pytest.warns(UserWarning) as record:
        set_array_api_strict_flags(api_version='2023.12')
    assert len(record) == 1
    assert '2023.12' in str(record[0].message)
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2023.12',
        'data_dependent_shapes': True,
        'enabled_extensions': ('linalg', 'fft'),
    }

    # Test setting flags with invalid values
    pytest.raises(ValueError, lambda:
                  set_array_api_strict_flags(api_version='2020.12'))
    pytest.raises(ValueError, lambda: set_array_api_strict_flags(
                      enabled_extensions=('linalg', 'fft', 'invalid')))
    with pytest.warns(UserWarning):
        pytest.raises(ValueError, lambda: set_array_api_strict_flags(
            api_version='2021.12',
            enabled_extensions=('linalg', 'fft')))

    # Test resetting flags
    with pytest.warns(UserWarning):
        set_array_api_strict_flags(
            api_version='2021.12',
            data_dependent_shapes=False,
            enabled_extensions=())
    reset_array_api_strict_flags()
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2022.12',
        'data_dependent_shapes': True,
        'enabled_extensions': ('linalg', 'fft'),
    }

def test_api_version():
    # Test defaults
    assert xp.__array_api_version__ == '2022.12'

    # Test setting the version
    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version='2021.12')
    assert xp.__array_api_version__ == '2021.12'

def test_data_dependent_shapes():
    a = asarray([0, 0, 1, 2, 2])
    mask = asarray([True, False, True, False, True])

    # Should not error
    unique_all(a)
    unique_counts(a)
    unique_inverse(a)
    unique_values(a)
    nonzero(a)
    a[mask]
    # TODO: add repeat when it is implemented

    set_array_api_strict_flags(data_dependent_shapes=False)

    pytest.raises(RuntimeError, lambda: unique_all(a))
    pytest.raises(RuntimeError, lambda: unique_counts(a))
    pytest.raises(RuntimeError, lambda: unique_inverse(a))
    pytest.raises(RuntimeError, lambda: unique_values(a))
    pytest.raises(RuntimeError, lambda: nonzero(a))
    pytest.raises(RuntimeError, lambda: a[mask])

linalg_examples = {
    'cholesky': lambda: xp.linalg.cholesky(xp.eye(3)),
    'cross': lambda: xp.linalg.cross(xp.asarray([1, 0, 0]), xp.asarray([0, 1, 0])),
    'det': lambda: xp.linalg.det(xp.eye(3)),
    'diagonal': lambda: xp.linalg.diagonal(xp.eye(3)),
    'eigh': lambda: xp.linalg.eigh(xp.eye(3)),
    'eigvalsh': lambda: xp.linalg.eigvalsh(xp.eye(3)),
    'inv': lambda: xp.linalg.inv(xp.eye(3)),
    'matmul': lambda: xp.linalg.matmul(xp.eye(3), xp.eye(3)),
    'matrix_norm': lambda: xp.linalg.matrix_norm(xp.eye(3)),
    'matrix_power': lambda: xp.linalg.matrix_power(xp.eye(3), 2),
    'matrix_rank': lambda: xp.linalg.matrix_rank(xp.eye(3)),
    'matrix_transpose': lambda: xp.linalg.matrix_transpose(xp.eye(3)),
    'outer': lambda: xp.linalg.outer(xp.asarray([1, 2, 3]), xp.asarray([4, 5, 6])),
    'pinv': lambda: xp.linalg.pinv(xp.eye(3)),
    'qr': lambda: xp.linalg.qr(xp.eye(3)),
    'slogdet': lambda: xp.linalg.slogdet(xp.eye(3)),
    'solve': lambda: xp.linalg.solve(xp.eye(3), xp.eye(3)),
    'svd': lambda: xp.linalg.svd(xp.eye(3)),
    'svdvals': lambda: xp.linalg.svdvals(xp.eye(3)),
    'tensordot': lambda: xp.linalg.tensordot(xp.eye(3), xp.eye(3)),
    'trace': lambda: xp.linalg.trace(xp.eye(3)),
    'vecdot': lambda: xp.linalg.vecdot(xp.asarray([1, 2, 3]), xp.asarray([4, 5, 6])),
    'vector_norm': lambda: xp.linalg.vector_norm(xp.asarray([1., 2., 3.])),
}

assert set(linalg_examples) == set(xp.linalg.__all__)

linalg_main_namespace_examples = {
    'matmul': lambda: xp.matmul(xp.eye(3), xp.eye(3)),
    'matrix_transpose': lambda: xp.matrix_transpose(xp.eye(3)),
    'tensordot': lambda: xp.tensordot(xp.eye(3), xp.eye(3)),
    'vecdot': lambda: xp.vecdot(xp.asarray([1, 2, 3]), xp.asarray([4, 5, 6])),
}

assert set(linalg_main_namespace_examples) == set(xp.__all__) & set(xp.linalg.__all__)

@pytest.mark.parametrize('func_name', linalg_examples.keys())
def test_linalg(func_name):
    func = linalg_examples[func_name]
    if func_name in linalg_main_namespace_examples:
        main_namespace_func = linalg_main_namespace_examples[func_name]
    else:
        main_namespace_func = lambda: None

    # First make sure the example actually works
    func()
    main_namespace_func()

    set_array_api_strict_flags(enabled_extensions=())
    pytest.raises(RuntimeError, func)
    main_namespace_func()

    set_array_api_strict_flags(enabled_extensions=('linalg',))
    func()
    main_namespace_func()

fft_examples = {
    'fft': lambda: xp.fft.fft(xp.asarray([0j, 1j, 0j, 0j])),
    'ifft': lambda: xp.fft.ifft(xp.asarray([0j, 1j, 0j, 0j])),
    'fftn': lambda: xp.fft.fftn(xp.asarray([[0j, 1j], [0j, 0j]])),
    'ifftn': lambda: xp.fft.ifftn(xp.asarray([[0j, 1j], [0j, 0j]])),
    'rfft': lambda: xp.fft.rfft(xp.asarray([0., 1., 0., 0.])),
    'irfft': lambda: xp.fft.irfft(xp.asarray([0j, 1j, 0j, 0j])),
    'rfftn': lambda: xp.fft.rfftn(xp.asarray([[0., 1.], [0., 0.]])),
    'irfftn': lambda: xp.fft.irfftn(xp.asarray([[0j, 1j], [0j, 0j]])),
    'hfft': lambda: xp.fft.hfft(xp.asarray([0j, 1j, 0j, 0j])),
    'ihfft': lambda: xp.fft.ihfft(xp.asarray([0., 1., 0., 0.])),
    'fftfreq': lambda: xp.fft.fftfreq(4),
    'rfftfreq': lambda: xp.fft.rfftfreq(4),
    'fftshift': lambda: xp.fft.fftshift(xp.asarray([0j, 1j, 0j, 0j])),
    'ifftshift': lambda: xp.fft.ifftshift(xp.asarray([0j, 1j, 0j, 0j])),
}

assert set(fft_examples) == set(xp.fft.__all__)

@pytest.mark.parametrize('func_name', fft_examples.keys())
def test_fft(func_name):
    func = fft_examples[func_name]

    # First make sure the example actually works
    func()

    set_array_api_strict_flags(enabled_extensions=())
    pytest.raises(RuntimeError, func)

    set_array_api_strict_flags(enabled_extensions=('fft',))
    func()
