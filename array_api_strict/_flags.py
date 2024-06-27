"""
These functions configure global flags that allow array-api-strict to be
used in different "modes". These modes include

- Changing to different supported versions of the standard.
- Enabling or disabling different optional behaviors (such as data-dependent
  shapes).
- Enabling or disabling different optional extensions.

None of these functions are part of the standard itself. A typical array API
library will only support one particular configuration of these flags.

"""

import functools
import os
import warnings

import array_api_strict

supported_versions = (
    "2021.12",
    "2022.12",
    "2023.12",
)

API_VERSION = default_version = "2022.12"

BOOLEAN_INDEXING = True

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
    api_version=None,
    boolean_indexing=None,
    data_dependent_shapes=None,
    enabled_extensions=None,
):
    """
    Set the array-api-strict flags to the specified values.

    Flags are global variables that enable or disable array-api-strict
    behaviors.

    The flags can also be changed by setting :ref:`environment variables <environment-variables>`.

    .. note::

       This function is **not** part of the array API standard. It only exists
       in array-api-strict.

    Parameters
    ----------
    api_version : str, optional
        The version of the standard to use. Supported versions are:
        ``{supported_versions}``. The default version number is
        ``{default_version!r}``.

        Note that 2021.12 is supported, but currently gives the same thing as
        2022.12 (except that the fft extension will be disabled).

        2023.12 support is experimental. Some features in 2023.12 may still be
        missing, and it hasn't been fully tested. A future version of
        array-api-strict will change the default version to 2023.12.

    boolean_indexing : bool, optional
        Whether indexing by a boolean array is supported. This flag is enabled
        by default. Note that although boolean array indexing does result in
        data-dependent shapes, this flag is independent of the
        `data_dependent_shapes` flag (see below).

    data_dependent_shapes : bool, optional
        Whether data-dependent shapes are enabled in array-api-strict.
        This flag is enabled by default. Array libraries that use computation
        graphs may not be able to support functions whose output shapes depend
        on the input data.

        The functions that make use of data-dependent shapes, and are therefore
        disabled by setting this flag to False are

        - `unique_all()`, `unique_counts()`, `unique_inverse()`, and `unique_values()`.
        - `nonzero()`
        - `repeat()` when the `repeats` argument is an array (requires the
          2023.12 version of the standard)

        Note that while boolean indexing is also data-dependent, it is
        controlled by a separate `boolean_indexing` flag (see above).

        See
        https://data-apis.org/array-api/latest/design_topics/data_dependent_output_shapes.html
        for more details.

    enabled_extensions : list of str, optional
        A list of extensions that are enabled in array-api-strict. The default
        is ``{default_extensions}``. Note that some extensions require a
        minimum version of the standard.


    Examples
    --------

    >>> from array_api_strict import set_array_api_strict_flags

    >>> # Set the standard version to 2021.12
    >>> set_array_api_strict_flags(api_version="2021.12")

    >>> # Disable data-dependent shapes and boolean indexing
    >>> set_array_api_strict_flags(data_dependent_shapes=False, boolean_indexing=False)

    >>> # Enable only the linalg extension (disable the fft extension)
    >>> set_array_api_strict_flags(enabled_extensions=["linalg"])

    See Also
    --------

    get_array_api_strict_flags: Get the current values of flags.
    reset_array_api_strict_flags: Reset the flags to their default values.
    ArrayAPIStrictFlags: A context manager to temporarily set the flags.

    """
    global API_VERSION, BOOLEAN_INDEXING, DATA_DEPENDENT_SHAPES, ENABLED_EXTENSIONS

    if api_version is not None:
        if api_version not in supported_versions:
            raise ValueError(f"Unsupported standard version {api_version!r}")
        if api_version == "2021.12":
            warnings.warn("The 2021.12 version of the array API specification was requested but the returned namespace is actually version 2022.12", stacklevel=2)
        if api_version == "2023.12":
            warnings.warn("The 2023.12 version of the array API specification is still preliminary. Some functions are not yet implemented, and it has not been fully tested.", stacklevel=2)
        API_VERSION = api_version
        array_api_strict.__array_api_version__ = API_VERSION

    if boolean_indexing is not None:
        BOOLEAN_INDEXING = boolean_indexing

    if data_dependent_shapes is not None:
        DATA_DEPENDENT_SHAPES = data_dependent_shapes

    if enabled_extensions is not None:
        for extension in enabled_extensions:
            if extension not in all_extensions:
                raise ValueError(f"Unsupported extension {extension}")
            if extension_versions[extension] > API_VERSION:
                raise ValueError(
                    f"Extension {extension} requires standard version "
                    f"{extension_versions[extension]} or later"
                )
        ENABLED_EXTENSIONS = tuple(enabled_extensions)
    else:
        ENABLED_EXTENSIONS = tuple([ext for ext in ENABLED_EXTENSIONS if extension_versions[ext] <= API_VERSION])

    array_api_strict.__all__[:] = sorted(set(ENABLED_EXTENSIONS) |
                                         set(array_api_strict.__all__) -
                                         set(default_extensions))

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

    .. note::

       The `inspection API
       <https://data-apis.org/array-api/latest/API_specification/inspection.html>`__
       provides a portable way to access most of this information. However, it
       is only present in standard versions starting with 2023.12. The array
       API version can be accessed portably using `xp.__array_api_version__`.

    Returns
    -------
    dict
        A dictionary containing the current array-api-strict flags.

    Examples
    --------

    >>> from array_api_strict import get_array_api_strict_flags
    >>> flags = get_array_api_strict_flags()
    >>> flags
    {'api_version': '2022.12',
     'boolean_indexing': True,
     'data_dependent_shapes': True,
     'enabled_extensions': ('linalg', 'fft')
    }

    See Also
    --------

    set_array_api_strict_flags: Set one or more flags to a given value.
    reset_array_api_strict_flags: Reset the flags to their default values.
    ArrayAPIStrictFlags: A context manager to temporarily set the flags.

    """
    return {
        "api_version": API_VERSION,
        "boolean_indexing": BOOLEAN_INDEXING,
        "data_dependent_shapes": DATA_DEPENDENT_SHAPES,
        "enabled_extensions": ENABLED_EXTENSIONS,
    }


def reset_array_api_strict_flags():
    """
    Reset the array-api-strict flags to their default values.

    This will also reset any flags that were set by :ref:`environment
    variables <environment-variables>` back to their default values.

    .. note::

       This function is **not** part of the array API standard. It only exists
       in array-api-strict.

    See :func:`set_array_api_strict_flags` for a list of flags and their
    default values.

    Examples
    --------

    >>> from array_api_strict import reset_array_api_strict_flags
    >>> reset_array_api_strict_flags()

    See Also
    --------

    get_array_api_strict_flags: Get the current values of flags.
    set_array_api_strict_flags: Set one or more flags to a given value.
    ArrayAPIStrictFlags: A context manager to temporarily set the flags.

    """
    global API_VERSION, BOOLEAN_INDEXING, DATA_DEPENDENT_SHAPES, ENABLED_EXTENSIONS
    API_VERSION = default_version
    array_api_strict.__array_api_version__ = API_VERSION
    BOOLEAN_INDEXING = True
    DATA_DEPENDENT_SHAPES = True
    ENABLED_EXTENSIONS = default_extensions


class ArrayAPIStrictFlags:
    """
    A context manager to temporarily set the array-api-strict flags.

    .. note::

       This class is **not** part of the array API standard. It only exists
       in array-api-strict.

    See :func:`set_array_api_strict_flags` for a
    description of the available flags.

    Examples
    --------

    >>> from array_api_strict import ArrayAPIStrictFlags, get_array_api_strict_flags
    >>> with ArrayAPIStrictFlags(api_version="2022.12", boolean_indexing=False):
    ...     flags = get_array_api_strict_flags()
    >>> flags
    {'api_version': '2022.12',
     'boolean_indexing': False,
     'data_dependent_shapes': True,
     'enabled_extensions': ('linalg', 'fft')
    }

    See Also
    --------

    get_array_api_strict_flags: Get the current values of flags.
    set_array_api_strict_flags: Set one or more flags to a given value.
    reset_array_api_strict_flags: Reset the flags to their default values.

    """
    def __init__(self, *, api_version=None, boolean_indexing=None,
                 data_dependent_shapes=None, enabled_extensions=None):
        self.kwargs = {
            "api_version": api_version,
            "boolean_indexing": boolean_indexing,
            "data_dependent_shapes": data_dependent_shapes,
            "enabled_extensions": enabled_extensions,
        }
        self.old_flags = get_array_api_strict_flags()

    def __enter__(self):
        set_array_api_strict_flags(**self.kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        set_array_api_strict_flags(**self.old_flags)

# Private functions

ENVIRONMENT_VARIABLES = [
    "ARRAY_API_STRICT_API_VERSION",
    "ARRAY_API_STRICT_BOOLEAN_INDEXING",
    "ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES",
    "ARRAY_API_STRICT_ENABLED_EXTENSIONS",
]

def set_flags_from_environment():
    kwargs = {}
    if "ARRAY_API_STRICT_API_VERSION" in os.environ:
        kwargs["api_version"] = os.environ["ARRAY_API_STRICT_API_VERSION"]

    if "ARRAY_API_STRICT_BOOLEAN_INDEXING" in os.environ:
        kwargs["boolean_indexing"] = os.environ["ARRAY_API_STRICT_BOOLEAN_INDEXING"].lower() == "true"

    if "ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES" in os.environ:
        kwargs["data_dependent_shapes"] = os.environ["ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES"].lower() == "true"

    if "ARRAY_API_STRICT_ENABLED_EXTENSIONS" in os.environ:
        enabled_extensions = os.environ["ARRAY_API_STRICT_ENABLED_EXTENSIONS"].split(",")
        if enabled_extensions == [""]:
            enabled_extensions = []
        kwargs["enabled_extensions"] = enabled_extensions

    # Called unconditionally because it is needed at first import to add
    # linalg and fft to __all__
    set_array_api_strict_flags(**kwargs)

set_flags_from_environment()

# Decorators

def requires_api_version(version):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if version > API_VERSION:
                raise RuntimeError(
                    f"The function {func.__name__} requires API version {version} or later, "
                    f"but the current API version for array-api-strict is {API_VERSION}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def requires_data_dependent_shapes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DATA_DEPENDENT_SHAPES:
            raise RuntimeError(f"The function {func.__name__} requires data-dependent shapes, but the data_dependent_shapes flag has been disabled for array-api-strict")
        return func(*args, **kwargs)
    return wrapper

def requires_extension(extension):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if extension not in ENABLED_EXTENSIONS:
                if extension == 'linalg' \
                   and func.__name__ in ['matmul', 'tensordot',
                                         'matrix_transpose', 'vecdot']:
                    raise RuntimeError(f"The linalg extension has been disabled for array-api-strict. However, {func.__name__} is also present in the main array_api_strict namespace and may be used from there.")
                raise RuntimeError(f"The function {func.__name__} requires the {extension} extension, but it has been disabled for array-api-strict")
            return func(*args, **kwargs)
        return wrapper
    return decorator
