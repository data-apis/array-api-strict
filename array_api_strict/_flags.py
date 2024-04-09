"""
This file defines flags for that allow array-api-strict to be used in
different "modes". These modes include

- Changing to different supported versions of the standard.
- Enabling or disabling different optional behaviors (such as data-dependent
  shapes).
- Enabling or disabling different optional extensions.

Nothing in this file is part of the standard itself. A typical array API
library will only support one particular configuration of these flags.
"""

import functools
import os

supported_versions = (
    "2021.12",
    "2022.12",
)

STANDARD_VERSION = default_version = "2022.12"

DATA_DEPENDENT_SHAPES = True

all_extensions = (
    "linalg",
    "fft",
)

extension_versions = {
    "linalg": "2021.12",
    "fft": "2022.12",
}

ENABLED_EXTENSIONS = default_extensions = (
    "linalg",
    "fft",
)

# Public functions

def set_array_api_strict_flags(
    *,
    standard_version=None,
    data_dependent_shapes=None,
    enabled_extensions=None,
):
    """
    Set the array-api-strict flags to the specified values.

    Flags are global variables that enable or disable array-api-strict
    behaviors.

    .. note::

       This function is **not** part of the array API standard. It only exists
       in array-api-strict.

    - `standard_version`: The version of the standard to use. Supported
      versions are: ``{supported_versions}``. The default version number is
      ``{default_version!r}``.

    - `data_dependent_shapes`: Whether data-dependent shapes are enabled in
      array-api-strict. This flag is enabled by default. Array libraries that
      use computation graphs may not be able to support functions whose output
      shapes depend on the input data.

      This flag is enabled by default. Array libraries that use computation graphs may not be able to support
      functions whose output shapes depend on the input data.

      The functions that make use of data-dependent shapes, and are therefore
      disabled by setting this flag to False are

      - `unique_all`, `unique_counts`, `unique_inverse`, and `unique_values`.
      - `nonzero`
      - Boolean array indexing
      - `repeat` when the `repeats` argument is an array (requires 2023.12
        version of the standard)

      See
      https://data-apis.org/array-api/latest/design_topics/data_dependent_output_shapes.html
      for more details.

    - `enabled_extensions`: A list of extensions that are enabled in
      array-api-strict. The default is ``{default_extensions}``. Note that
      some extensions require a minimum version of the standard.

    The default values of the flags can also be changed by setting environment
    variables:

    - ``ARRAY_API_STRICT_STANDARD_VERSION``: A string representing the version number.
    - ``ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES``: "True" or "False".
    - ``ARRAY_API_STRICT_ENABLED_EXTENSIONS``: A comma separated list of
      extensions to enable.

    Examples
    --------

    >>> from array_api_strict import set_array_api_strict_flags
    >>> # Set the standard version to 2021.12
    >>> set_array_api_strict_flags(standard_version="2021.12")
    >>> # Disable data-dependent shapes
    >>> set_array_api_strict_flags(data_dependent_shapes=False)
    >>> # Enable only the linalg extension (disable the fft extension)
    >>> set_array_api_strict_flags(enabled_extensions=["linalg"])

    See Also
    --------

    get_array_api_strict_flags
    reset_array_api_strict_flags
    ArrayApiStrictFlags: A context manager to temporarily set the flags.

    """
    global STANDARD_VERSION, DATA_DEPENDENT_SHAPES, ENABLED_EXTENSIONS

    if standard_version is not None:
        if standard_version not in supported_versions:
            raise ValueError(f"Unsupported standard version {standard_version}")
        STANDARD_VERSION = standard_version

    if data_dependent_shapes is not None:
        DATA_DEPENDENT_SHAPES = data_dependent_shapes

    if enabled_extensions is not None:
        for extension in enabled_extensions:
            if extension not in all_extensions:
                raise ValueError(f"Unsupported extension {extension}")
            if extension_versions[extension] > STANDARD_VERSION:
                raise ValueError(
                    f"Extension {extension} requires standard version "
                    f"{extension_versions[extension]} or later"
                )
        ENABLED_EXTENSIONS = tuple(enabled_extensions)
    else:
        ENABLED_EXTENSIONS = tuple([ext for ext in all_extensions if extension_versions[ext] <= STANDARD_VERSION])

# We have to do this separately or it won't get added as the docstring
set_array_api_strict_flags.__doc__ = set_array_api_strict_flags.__doc__.format(
    supported_versions=supported_versions,
    default_version=default_version,
    default_extensions=default_extensions,
)

def get_array_api_strict_flags():
    """
    Get the current array-api-strict flags.

    .. note::

       This function is **not** part of the array API standard. It only exists
       in array-api-strict.

    Returns
    -------
    dict
        A dictionary containing the current array-api-strict flags.

    Examples
    --------

    >>> from array_api_strict import get_array_api_strict_flags
    >>> flags = get_array_api_strict_flags()
    >>> flags
    {'standard_version': '2022.12', 'data_dependent_shapes': True, 'enabled_extensions': ('linalg', 'fft')}

    See Also
    --------

    set_array_api_strict_flags
    reset_array_api_strict_flags
    ArrayApiStrictFlags: A context manager to temporarily set the flags.

    """
    return {
        "standard_version": STANDARD_VERSION,
        "data_dependent_shapes": DATA_DEPENDENT_SHAPES,
        "enabled_extensions": ENABLED_EXTENSIONS,
    }


def reset_array_api_strict_flags():
    """
    Reset the array-api-strict flags to their default values.

    This will also reset any flags that were set by environment variables.

    .. note::

       This function is **not** part of the array API standard. It only exists
       in array-api-strict.

    Examples
    --------

    >>> from array_api_strict import reset_array_api_strict_flags
    >>> reset_array_api_strict_flags()

    See Also
    --------

    set_array_api_strict_flags
    get_array_api_strict_flags
    ArrayApiStrictFlags: A context manager to temporarily set the flags.

    """
    global STANDARD_VERSION, DATA_DEPENDENT_SHAPES, ENABLED_EXTENSIONS
    STANDARD_VERSION = default_version
    DATA_DEPENDENT_SHAPES = True
    ENABLED_EXTENSIONS = default_extensions


class ArrayApiStrictFlags:
    """
    A context manager to temporarily set the array-api-strict flags.

    .. note::

       This class is **not** part of the array API standard. It only exists
       in array-api-strict.

    See :func:`~.array_api_strict.set_array_api_strict_flags` for a
    description of the available flags.

    See Also
    --------

    set_array_api_strict_flags
    get_array_api_strict_flags
    reset_array_api_strict_flags

    """
    def __init__(self, *, standard_version=None, data_dependent_shapes=None,
                 enabled_extensions=None):
        self.kwargs = {
            "standard_version": standard_version,
            "data_dependent_shapes": data_dependent_shapes,
            "enabled_extensions": enabled_extensions,
        }
        self.old_flags = get_array_api_strict_flags()

    def __enter__(self):
        set_array_api_strict_flags(**self.kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        set_array_api_strict_flags(**self.old_flags)

# Private functions

def set_flags_from_environment():
    if "ARRAY_API_STRICT_STANDARD_VERSION" in os.environ:
        set_array_api_strict_flags(
            standard_version=os.environ["ARRAY_API_STRICT_STANDARD_VERSION"]
        )

    if "ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES" in os.environ:
        set_array_api_strict_flags(
            data_dependent_shapes=os.environ["ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES"].lower() == "true"
        )

    if "ARRAY_API_STRICT_ENABLED_EXTENSIONS" in os.environ:
        set_array_api_strict_flags(
            enabled_extensions=os.environ["ARRAY_API_STRICT_ENABLED_EXTENSIONS"].split(",")
        )

set_flags_from_environment()

def requires_data_dependent_shapes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DATA_DEPENDENT_SHAPES:
            raise RuntimeError(f"The function {func.__name__} requires data-dependent shapes, but the data_dependent_shapes flag has been disabled for array-api-strict")
        return func(*args, **kwargs)
    return wrapper
