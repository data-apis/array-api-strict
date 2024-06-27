# Changelog

## 2.0 (2024-06-27)

### Major Changes

- array-api-strict has a new set of [flags](array-api-strict-flags) that can
  be used to dynamically enable or disable features in array-api-strict. These
  flags allow you to change the supported array API version, enable or disable
  [extensions](https://data-apis.org/array-api/latest/extensions/index.html),
  enable or disable features that rely on data-dependent shapes, and enable or
  disable boolean indexing. Future versions may add additional flags to allow
  changing other optional or varying behavior in the standard.

- Added experimental support for the
  [2023.12](https://data-apis.org/array-api/2023.12/changelog.html#v2023-12)
  version of the array API standard. The default version is still 2022.12, but
  the version can be changed to 2023.12 using the aforementioned flags, either
  by calling
  {func}`array_api_strict.set_array_api_strict_flags(api_version='2023.12')
  <array_api_strict.set_array_api_strict_flags>` or setting the environment
  variable {envvar}`ARRAY_API_STRICT_API_VERSION=2023.12
  <ARRAY_API_STRICT_API_VERSION>`.

### Minor Changes

- Calling `iter()` on an array now correctly raises `TypeError`.

- Add some missing names to `__all__`.

## 1.1.1 (2024-04-29)

- Fix the `api_version` argument to `__array_namespace__` to accept
  `'2021.12'` or `'2022.12'`.

## 1.1 (2024-04-08)

- Fix the `copy` flag in `__array__` for NumPy 2.0.

- Add full `copy=False` support to `asarray()`. This is emulated in NumPy 1.26 by creating
  the array and seeing if it is copied. For NumPy 2.0, the new native
  `copy=False` flag is used.

- Add broadcasting support to `cross`.

## 1.0 (2024-01-24)

This is the first release of `array_api_strict`. It is extracted from
`numpy.array_api`, which was included as an experimental submodule in NumPy
versions prior to 2.0. Note that the commit history in this repository is
extracted from the git history of numpy/array_api/ (see [](numpy.array_api)).

Additionally, the following changes are new to `array_api_strict` from
`numpy.array_api` in NumPy 1.26 (the last NumPy feature release to include
`numpy.array_api`):

- ``array_api_strict`` was made more portable. In particular:

  - ``array_api_strict`` no longer uses ``"cpu"`` as its "device", but rather a
    separate ``CPU_DEVICE`` object (which is not accessible in the namespace).
    This is because "cpu" is not part of the array API standard.

  - ``array_api_strict`` now uses separate wrapped objects for dtypes.
    Previously it reused the ``numpy`` dtype objects. This makes it clear
    which behaviors on dtypes are part of the array API standard (effectively,
    the standard only requires ``==`` on dtype objects).

- ``numpy.array_api.nonzero`` now errors on zero-dimensional arrays, as
    required by the array API standard.

- Support for the optional [fft
  extension](https://data-apis.org/array-api/latest/extensions/fourier_transform_functions.html)
  was added.
