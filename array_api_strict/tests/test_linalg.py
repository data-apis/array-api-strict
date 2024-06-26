import pytest

from .._flags import set_array_api_strict_flags

import array_api_strict as xp

# TODO: Maybe all of these exceptions should be IndexError?

# Technically this is linear_algebra, not linalg, but it's simpler to keep
# both of these tests together
def test_vecdot_2023_12():
    # Test the axis < 0 restriction for 2023.12, and also the 2022.12 axis >=
    # 0 behavior (which is primarily kept for backwards compatibility).

    a = xp.ones((2, 3, 4, 5))
    b = xp.ones((   3, 4, 1))

    # 2022.12 behavior, which is to apply axis >= 0 after broadcasting
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=0))
    assert xp.linalg.vecdot(a, b, axis=1).shape == (2, 4, 5)
    assert xp.linalg.vecdot(a, b, axis=2).shape == (2, 3, 5)
    # This is disallowed because the arrays must have the same values before
    # broadcasting
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=-1))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=-4))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=3))

    # Out-of-bounds axes even after broadcasting
    pytest.raises(IndexError, lambda: xp.linalg.vecdot(a, b, axis=4))
    pytest.raises(IndexError, lambda: xp.linalg.vecdot(a, b, axis=-5))

    # negative axis behavior is unambiguous when it's within the bounds of
    # both arrays before broadcasting
    assert xp.linalg.vecdot(a, b, axis=-2).shape == (2, 3, 5)
    assert xp.linalg.vecdot(a, b, axis=-3).shape == (2, 4, 5)

    # 2023.12 behavior, which is to only allow axis < 0 and axis >=
    # min(x1.ndim, x2.ndim), which is unambiguous
    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version='2023.12')

    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=0))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=1))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=2))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=3))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=-1))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=-4))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=4))
    pytest.raises(ValueError, lambda: xp.linalg.vecdot(a, b, axis=-5))

    assert xp.linalg.vecdot(a, b, axis=-2).shape == (2, 3, 5)
    assert xp.linalg.vecdot(a, b, axis=-3).shape == (2, 4, 5)

@pytest.mark.parametrize('api_version', ['2021.12', '2022.12', '2023.12'])
def test_cross(api_version):
    # This test tests everything that should be the same across all supported
    # API versions.

    if api_version != '2022.12':
        with pytest.warns(UserWarning):
            set_array_api_strict_flags(api_version=api_version)
    else:
        set_array_api_strict_flags(api_version=api_version)

    a = xp.ones((2, 4, 5, 3))
    b = xp.ones((   4, 1, 3))
    assert xp.linalg.cross(a, b, axis=-1).shape == (2, 4, 5, 3)

    a = xp.ones((2, 4, 3, 5))
    b = xp.ones((   4, 3, 1))
    assert xp.linalg.cross(a, b, axis=-2).shape == (2, 4, 3, 5)

    # This is disallowed because the axes must equal 3 before broadcasting
    a = xp.ones((3, 2, 3, 5))
    b = xp.ones((   2, 1, 1))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=-1))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=-2))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=-3))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=-4))

    # Out-of-bounds axes even after broadcasting
    pytest.raises(IndexError, lambda: xp.linalg.cross(a, b, axis=4))
    pytest.raises(IndexError, lambda: xp.linalg.cross(a, b, axis=-5))

@pytest.mark.parametrize('api_version', ['2021.12', '2022.12'])
def test_cross_2022_12(api_version):
    # Test the 2022.12 axis >= 0 behavior, which is primarily kept for
    # backwards compatibility. Note that unlike vecdot, array_api_strict
    # cross() never implemented the "after broadcasting" axis behavior, but
    # just reused NumPy cross(), which applies axes before broadcasting.
    if api_version != '2022.12':
        with pytest.warns(UserWarning):
            set_array_api_strict_flags(api_version=api_version)
    else:
        set_array_api_strict_flags(api_version=api_version)

    a = xp.ones((3, 2, 4, 5))
    b = xp.ones((3, 2, 4, 1))
    assert xp.linalg.cross(a, b, axis=0).shape == (3, 2, 4, 5)

    # ambiguous case
    a = xp.ones((   3, 4, 5))
    b = xp.ones((3, 2, 4, 1))
    assert xp.linalg.cross(a, b, axis=0).shape == (3, 2, 4, 5)

def test_cross_2023_12():
    # 2023.12 behavior, which is to only allow axis < 0 and axis >=
    # min(x1.ndim, x2.ndim), which is unambiguous
    with pytest.warns(UserWarning):
        set_array_api_strict_flags(api_version='2023.12')

    a = xp.ones((3, 2, 4, 5))
    b = xp.ones((3, 2, 4, 1))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=0))

    a = xp.ones((   3, 4, 5))
    b = xp.ones((3, 2, 4, 1))
    pytest.raises(ValueError, lambda: xp. linalg.cross(a, b, axis=0))

    a = xp.ones((2, 4, 5, 3))
    b = xp.ones((   4, 1, 3))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=0))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=1))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=2))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=3))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=-2))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=-3))
    pytest.raises(ValueError, lambda: xp.linalg.cross(a, b, axis=-4))

    pytest.raises(IndexError, lambda: xp.linalg.cross(a, b, axis=4))
    pytest.raises(IndexError, lambda: xp.linalg.cross(a, b, axis=-5))

    assert xp.linalg.cross(a, b, axis=-1).shape == (2, 4, 5, 3)
