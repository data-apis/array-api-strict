import pytest

import array_api_strict as xp

from array_api_strict import ArrayAPIStrictFlags
from array_api_strict._flags import draft_version


def test_where_with_scalars():
    x = xp.asarray([1, 2, 3, 1])

    # Versions up to and including 2023.12 don't support scalar arguments
    with pytest.raises(AttributeError, match="object has no attribute 'dtype'"):
        xp.where(x == 1, 42, 44)

    # Versions after 2023.12 support scalar arguments
    with (pytest.warns(
              UserWarning,
              match="The 2024.12 version of the array API specification is in draft status"
          ),
          ArrayAPIStrictFlags(api_version=draft_version),
        ):
        x_where = xp.where(x == 1, xp.asarray(42), 44)

        expected = xp.asarray([42, 44, 44, 42])
        assert xp.all(x_where == expected)

        # The spec does not allow both x1 and x2 to be scalars
        with pytest.raises(ValueError, match="One of"):
            xp.where(x == 1, 42, 44)
