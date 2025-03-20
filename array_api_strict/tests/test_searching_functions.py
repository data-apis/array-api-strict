import pytest

import array_api_strict as xp

from array_api_strict import ArrayAPIStrictFlags


def test_where_with_scalars():
    x = xp.asarray([1, 2, 3, 1])

    # Versions up to and including 2023.12 don't support scalar arguments
    with ArrayAPIStrictFlags(api_version='2023.12'):
        with pytest.raises(AttributeError, match="object has no attribute 'dtype'"):
            xp.where(x == 1, 42, 44)

    # Versions after 2023.12 support scalar arguments
    x_where = xp.where(x == 1, xp.asarray(42), 44)

    expected = xp.asarray([42, 44, 44, 42])
    assert xp.all(x_where == expected)

    # The spec does not allow both x1 and x2 to be scalars
    with pytest.raises(TypeError, match="Two scalars"):
        xp.where(x == 1, 42, 44)


def test_where_mixed_dtypes():
    # https://github.com/data-apis/array-api-strict/issues/131
    x =  xp.asarray([1., 2.])
    res = xp.where(x > 1.5, x, 0)
    assert res.dtype == x.dtype
    assert all(res == xp.asarray([0., 2.]))

    # retry with boolean x1, x2
    c = x > 1.5
    res = xp.where(c, False, c)
    assert all(res == xp.asarray([False, False]))


def test_where_f32():
    # https://github.com/data-apis/array-api-strict/issues/131#issuecomment-2723016294
    res = xp.where(xp.asarray([True, False]), 1., xp.asarray([2, 2], dtype=xp.float32))
    assert res.dtype == xp.float32

