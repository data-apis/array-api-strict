import sys
import subprocess

from .._flags import (set_array_api_strict_flags, get_array_api_strict_flags,
                      reset_array_api_strict_flags)
from .._info import (capabilities, default_device, default_dtypes, devices,
                     dtypes)
from .._fft import (fft, ifft, fftn, ifftn, rfft, irfft, rfftn, irfftn, hfft,
                    ihfft, fftfreq, rfftfreq, fftshift, ifftshift)
from .._linalg import (cholesky, cross, det, diagonal, eigh, eigvalsh, inv,
                       matmul, matrix_norm, matrix_power, matrix_rank, matrix_transpose, outer, pinv,
                       qr, slogdet, solve, svd, svdvals, tensordot, trace, vecdot, vector_norm)

from .. import (asarray, unique_all, unique_counts, unique_inverse,
                unique_values, nonzero, repeat)

import array_api_strict as xp

import pytest

def test_flags():
    # Test defaults
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2022.12',
        'boolean_indexing': True,
        'data_dependent_shapes': True,
        'enabled_extensions': ('linalg', 'fft'),
    }

    # Test setting flags
    set_array_api_strict_flags(data_dependent_shapes=False)
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2022.12',
        'boolean_indexing': True,
        'data_dependent_shapes': False,
        'enabled_extensions': ('linalg', 'fft'),
    }
    set_array_api_strict_flags(enabled_extensions=('fft',))
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2022.12',
        'boolean_indexing': True,
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
        'boolean_indexing': True,
        'data_dependent_shapes': False,
        'enabled_extensions': (),
    }
    reset_array_api_strict_flags()

    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version='2021.12')
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2021.12',
        'boolean_indexing': True,
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
        'boolean_indexing': True,
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
            boolean_indexing=False,
            data_dependent_shapes=False,
            enabled_extensions=())
    reset_array_api_strict_flags()
    flags = get_array_api_strict_flags()
    assert flags == {
        'api_version': '2022.12',
        'boolean_indexing': True,
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
    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version='2023.12') # to enable repeat()

    a = asarray([0, 0, 1, 2, 2])
    mask = asarray([True, False, True, False, True])
    repeats = asarray([1, 1, 2, 2, 2])

    # Should not error
    unique_all(a)
    unique_counts(a)
    unique_inverse(a)
    unique_values(a)
    nonzero(a)
    a[mask]
    repeat(a, repeats)
    repeat(a, 2)

    set_array_api_strict_flags(data_dependent_shapes=False)

    pytest.raises(RuntimeError, lambda: unique_all(a))
    pytest.raises(RuntimeError, lambda: unique_counts(a))
    pytest.raises(RuntimeError, lambda: unique_inverse(a))
    pytest.raises(RuntimeError, lambda: unique_values(a))
    pytest.raises(RuntimeError, lambda: nonzero(a))
    pytest.raises(RuntimeError, lambda: repeat(a, repeats))
    repeat(a, 2) # Should never error
    a[mask] # No error (boolean indexing is a separate flag)

def test_boolean_indexing():
    a = asarray([0, 0, 1, 2, 2])
    mask = asarray([True, False, True, False, True])

    # Should not error
    a[mask]

    set_array_api_strict_flags(boolean_indexing=False)

    pytest.raises(RuntimeError, lambda: a[mask])

linalg_examples = {
    'cholesky': lambda: cholesky(xp.eye(3)),
    'cross': lambda: cross(xp.asarray([1, 0, 0]), xp.asarray([0, 1, 0])),
    'det': lambda: det(xp.eye(3)),
    'diagonal': lambda: diagonal(xp.eye(3)),
    'eigh': lambda: eigh(xp.eye(3)),
    'eigvalsh': lambda: eigvalsh(xp.eye(3)),
    'inv': lambda: inv(xp.eye(3)),
    'matmul': lambda: matmul(xp.eye(3), xp.eye(3)),
    'matrix_norm': lambda: matrix_norm(xp.eye(3)),
    'matrix_power': lambda: matrix_power(xp.eye(3), 2),
    'matrix_rank': lambda: matrix_rank(xp.eye(3)),
    'matrix_transpose': lambda: matrix_transpose(xp.eye(3)),
    'outer': lambda: outer(xp.asarray([1, 2, 3]), xp.asarray([4, 5, 6])),
    'pinv': lambda: pinv(xp.eye(3)),
    'qr': lambda: qr(xp.eye(3)),
    'slogdet': lambda: slogdet(xp.eye(3)),
    'solve': lambda: solve(xp.eye(3), xp.eye(3)),
    'svd': lambda: svd(xp.eye(3)),
    'svdvals': lambda: svdvals(xp.eye(3)),
    'tensordot': lambda: tensordot(xp.eye(3), xp.eye(3)),
    'trace': lambda: trace(xp.eye(3)),
    'vecdot': lambda: vecdot(xp.asarray([1, 2, 3]), xp.asarray([4, 5, 6])),
    'vector_norm': lambda: vector_norm(xp.asarray([1., 2., 3.])),
}

assert set(linalg_examples) == set(xp.linalg.__all__)

linalg_main_namespace_examples = {
    'matmul': lambda: xp.matmul(xp.eye(3), xp.eye(3)),
    'matrix_transpose': lambda: xp.matrix_transpose(xp.eye(3)),
    'tensordot': lambda: xp.tensordot(xp.eye(3), xp.eye(3)),
    'vecdot': lambda: xp.vecdot(xp.asarray([1, 2, 3]), xp.asarray([4, 5, 6])),
    'mT': lambda: xp.eye(3).mT,
}

assert set(linalg_main_namespace_examples) == (set(xp.__all__) & set(xp.linalg.__all__)) | {"mT"}

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
    'fft': lambda: fft(xp.asarray([0j, 1j, 0j, 0j])),
    'ifft': lambda: ifft(xp.asarray([0j, 1j, 0j, 0j])),
    'fftn': lambda: fftn(xp.asarray([[0j, 1j], [0j, 0j]])),
    'ifftn': lambda: ifftn(xp.asarray([[0j, 1j], [0j, 0j]])),
    'rfft': lambda: rfft(xp.asarray([0., 1., 0., 0.])),
    'irfft': lambda: irfft(xp.asarray([0j, 1j, 0j, 0j])),
    'rfftn': lambda: rfftn(xp.asarray([[0., 1.], [0., 0.]])),
    'irfftn': lambda: irfftn(xp.asarray([[0j, 1j], [0j, 0j]])),
    'hfft': lambda: hfft(xp.asarray([0j, 1j, 0j, 0j])),
    'ihfft': lambda: ihfft(xp.asarray([0., 1., 0., 0.])),
    'fftfreq': lambda: fftfreq(4),
    'rfftfreq': lambda: rfftfreq(4),
    'fftshift': lambda: fftshift(xp.asarray([0j, 1j, 0j, 0j])),
    'ifftshift': lambda: ifftshift(xp.asarray([0j, 1j, 0j, 0j])),
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

api_version_2023_12_examples = {
    '__array_namespace_info__': lambda: xp.__array_namespace_info__(),
    # Test these functions directly to ensure they are properly decorated
    'capabilities': capabilities,
    'default_device': default_device,
    'default_dtypes': default_dtypes,
    'devices': devices,
    'dtypes': dtypes,
    'clip': lambda: xp.clip(xp.asarray([1, 2, 3]), 1, 2),
    'copysign': lambda: xp.copysign(xp.asarray([1., 2., 3.]), xp.asarray([-1., -1., -1.])),
    'cumulative_sum': lambda: xp.cumulative_sum(xp.asarray([1, 2, 3])),
    'hypot': lambda: xp.hypot(xp.asarray([3., 4.]), xp.asarray([4., 3.])),
    'maximum': lambda: xp.maximum(xp.asarray([1, 2, 3]), xp.asarray([2, 3, 4])),
    'minimum': lambda: xp.minimum(xp.asarray([1, 2, 3]), xp.asarray([2, 3, 4])),
    'moveaxis': lambda: xp.moveaxis(xp.ones((3, 3)), 0, 1),
    'repeat': lambda: xp.repeat(xp.asarray([1, 2, 3]), 3),
    'searchsorted': lambda: xp.searchsorted(xp.asarray([1, 2, 3]), xp.asarray([0, 1, 2, 3, 4])),
    'signbit': lambda: xp.signbit(xp.asarray([-1., 0., 1.])),
    'tile': lambda: xp.tile(xp.ones((3, 3)), (2, 3)),
    'unstack': lambda: xp.unstack(xp.ones((3, 3)), axis=0),
}

@pytest.mark.parametrize('func_name', api_version_2023_12_examples.keys())
def test_api_version_2023_12(func_name):
    func = api_version_2023_12_examples[func_name]

    # By default, these functions should error
    pytest.raises(RuntimeError, func)

    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version='2023.12')
        func()

    set_array_api_strict_flags(api_version='2022.12')
    pytest.raises(RuntimeError, func)

def test_disabled_extensions():
    # Test that xp.extension errors when an extension is disabled, and that
    # xp.__all__ is updated properly.

    # First test that things are correct on the initial import. Since we have
    # already called set_array_api_strict_flags many times throughout running
    # the tests, we have to test this in a subprocess.
    subprocess_tests = [('''\
import array_api_strict

array_api_strict.linalg # No error
array_api_strict.fft # No error
assert "linalg" in array_api_strict.__all__
assert "fft" in array_api_strict.__all__
assert len(array_api_strict.__all__) == len(set(array_api_strict.__all__))
''', {}),
# Test that the initial population of __all__ works correctly
('''\
from array_api_strict import * # No error
linalg # Should have been imported by the previous line
fft
''', {}),
('''\
from array_api_strict import * # No error
linalg # Should have been imported by the previous line
assert 'fft' not in globals()
''', {"ARRAY_API_STRICT_ENABLED_EXTENSIONS": "linalg"}),
('''\
from array_api_strict import * # No error
fft # Should have been imported by the previous line
assert 'linalg' not in globals()
''', {"ARRAY_API_STRICT_ENABLED_EXTENSIONS": "fft"}),
('''\
from array_api_strict import * # No error
assert 'linalg' not in globals()
assert 'fft' not in globals()
''', {"ARRAY_API_STRICT_ENABLED_EXTENSIONS": ""}),
]
    for test, env in subprocess_tests:
        try:
            subprocess.run([sys.executable, '-c', test], check=True,
                           capture_output=True, encoding='utf-8', env=env)
        except subprocess.CalledProcessError as e:
            print(e.stdout, end='')
            # Ensure the exception is shown in the output log
            raise AssertionError(e.stderr)

    assert 'linalg' in xp.__all__
    assert 'fft' in xp.__all__
    xp.linalg # No error
    xp.fft # No error
    ns = {}
    exec('from array_api_strict import *', ns)
    assert 'linalg' in ns
    assert 'fft' in ns

    set_array_api_strict_flags(enabled_extensions=('linalg',))
    assert 'linalg' in xp.__all__
    assert 'fft' not in xp.__all__
    xp.linalg # No error
    pytest.raises(AttributeError, lambda: xp.fft)
    ns = {}
    exec('from array_api_strict import *', ns)
    assert 'linalg' in ns
    assert 'fft' not in ns

    set_array_api_strict_flags(enabled_extensions=('fft',))
    assert 'linalg' not in xp.__all__
    assert 'fft' in xp.__all__
    pytest.raises(AttributeError, lambda: xp.linalg)
    xp.fft # No error
    ns = {}
    exec('from array_api_strict import *', ns)
    assert 'linalg' not in ns
    assert 'fft' in ns

    set_array_api_strict_flags(enabled_extensions=())
    assert 'linalg' not in xp.__all__
    assert 'fft' not in xp.__all__
    pytest.raises(AttributeError, lambda: xp.linalg)
    pytest.raises(AttributeError, lambda: xp.fft)
    ns = {}
    exec('from array_api_strict import *', ns)
    assert 'linalg' not in ns
    assert 'fft' not in ns


def test_environment_variables():
    # Test that the environment variables work as expected
    subprocess_tests = [
        # ARRAY_API_STRICT_API_VERSION
        ('''\
import array_api_strict as xp
assert xp.__array_api_version__ == '2022.12'

assert xp.get_array_api_strict_flags()['api_version'] == '2022.12'

''', {}),
        *[
        (f'''\
import array_api_strict as xp
assert xp.__array_api_version__ == '{version}'

assert xp.get_array_api_strict_flags()['api_version'] == '{version}'

if {version} == '2021.12':
    assert hasattr(xp, 'linalg')
    assert not hasattr(xp, 'fft')

''', {"ARRAY_API_STRICT_API_VERSION": version}) for version in ('2021.12', '2022.12', '2023.12')],

       # ARRAY_API_STRICT_BOOLEAN_INDEXING
        ('''\
import array_api_strict as xp

a = xp.ones(3)
mask = xp.asarray([True, False, True])

assert xp.all(a[mask] == xp.asarray([1., 1.]))
assert xp.get_array_api_strict_flags()['boolean_indexing'] == True
''', {}),
        *[(f'''\
import array_api_strict as xp

a = xp.ones(3)
mask = xp.asarray([True, False, True])

if {boolean_indexing}:
    assert xp.all(a[mask] == xp.asarray([1., 1.]))
else:
    try:
        a[mask]
    except RuntimeError:
        pass
    else:
        assert False

assert xp.get_array_api_strict_flags()['boolean_indexing'] == {boolean_indexing}
''', {"ARRAY_API_STRICT_BOOLEAN_INDEXING": boolean_indexing})
            for boolean_indexing in ('True', 'False')],

        # ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES
        ('''\
import array_api_strict as xp

a = xp.ones(3)
xp.unique_all(a)

assert xp.get_array_api_strict_flags()['data_dependent_shapes'] == True
''', {}),
        *[(f'''\
import array_api_strict as xp

a = xp.ones(3)
if {data_dependent_shapes}:
    xp.unique_all(a)
else:
    try:
        xp.unique_all(a)
    except RuntimeError:
        pass
    else:
        assert False

assert xp.get_array_api_strict_flags()['data_dependent_shapes'] == {data_dependent_shapes}
''', {"ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES": data_dependent_shapes})
            for data_dependent_shapes in ('True', 'False')],

        # ARRAY_API_STRICT_ENABLED_EXTENSIONS
        ('''\
import array_api_strict as xp
assert hasattr(xp, 'linalg')
assert hasattr(xp, 'fft')

assert xp.get_array_api_strict_flags()['enabled_extensions'] == ('linalg', 'fft')
''', {}),
        *[(f'''\
import array_api_strict as xp

assert hasattr(xp, 'linalg') == ('linalg' in {extensions.split(',')})
assert hasattr(xp, 'fft') == ('fft' in {extensions.split(',')})

assert sorted(xp.get_array_api_strict_flags()['enabled_extensions']) == {sorted(set(extensions.split(','))-{''})}
''', {"ARRAY_API_STRICT_ENABLED_EXTENSIONS": extensions})
            for extensions in ('', 'linalg', 'fft', 'linalg,fft')],
    ]

    for test, env in subprocess_tests:
        try:
            subprocess.run([sys.executable, '-c', test], check=True,
                           capture_output=True, encoding='utf-8', env=env)
        except subprocess.CalledProcessError as e:
            print(e.stdout, end='')
            # Ensure the exception is shown in the output log
            raise AssertionError(f"""\
STDOUT:
{e.stderr}

STDERR:
{e.stderr}

TEST:
{test}

ENV:
{env}""")
