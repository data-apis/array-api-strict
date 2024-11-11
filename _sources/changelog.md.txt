# Changelog

## 2.2 (2024-11-11)

### Major Changes

- Preliminary support for the draft 2024.12 version of the standard is now
  implemented. This is disabled by default, but can be enabled with the [flags
  API](array-api-strict-flags), e.g., by calling
  `set_array_api_strict_flags(api_version='2024.12')` or setting
  `ARRAY_API_STRICT_API_VERSION=2024.12`.

  Note that this support is still preliminary and still relatively untested.
  Please [report any
  issues](https://github.com/data-apis/array-api-strict/issues) you find.

  The following functions are implemented for 2024:

  - `diff`
  - `nextafter`
  - `reciprocal`
  - `take_along_axis`
  - The `'max dimensions'` key of `__array_namespace_info__().capabilities()`.

  Some planned changes to the 2024.12 standard, including scalar support for
  array functions, is not yet implemented but will be in a future version.

### Minor Changes

- `__array_namespace_info__()` now returns a class instead of a module. This
  prevents extraneous names that aren't part of the standard from appearing on
  it.

## 2.1.3 (2024-11-08)

### Major Changes

- Revert the change to require NumPy >= 2.1 and Python >= 3.10 from
  array-api-strict 2.1.2. array-api-strict now requires NumPy >= 1.21 and
  Python >= 3.9, as before. These changes were made to improve the maintenance
  of array-api-strict, but they caused some issues in upstream packages that
  cannot yet support NumPy 2.0, so this will be postponed to a later date.

- Revert the removal of `__array__` from array-api-strict 2.1.1. This caused
  some difficulties for upstream libraries, so it will be postponed to a later
  date. This is still planned because `__array__` is not part of the array API
  standard. See https://github.com/data-apis/array-api-strict/issues/67 for
  more discussion about this.

## 2.1.2 (2024-11-07)

### Major Changes

- array-api-strict now requires NumPy >= 2.1 and Python >= 3.10

## 2.1.1 (2024-11-07)

### Major Changes

- Remove the `__array__` method from array-api-strict arrays. This means they
  will no longer be implicitly converted to NumPy arrays when passed to `np`
  functions. This method was previously implemented as a convenience, but it
  isn't part of the array API standard. To portably convert an array API
  strict array to a NumPy array, use `np.from_dlpack(x)`

### Minor Changes

- Use a more robust implementation of `clip()` that handles corner cases better.

- Fix the definition of `sign()` for complex numbers when using NumPy 1.x.

- Correctly use the array's device when promoting scalars. (Thanks to
  [@betatim](https://github.com/betatim))

- Correctly propagate the input array's device in `asarray()`. (Thanks to
  [@betatim](https://github.com/betatim))

## 2.1 (2024-10-18)

### Major Changes

- The default version of the array API standard is now 2023.12. 2022.12 can
  still be enabled via the [flags API](array-api-strict-flags).

- Added support for multiple fake "devices", so that code testing against
  array-api-strict can check for proper device support. Currently there are
  three "devices", the "CPU" device, which is the default devices, and two
  pseudo "device" objects. This set of devices can be accessed with
  `array_api_strict.__array_namespace_info__().devices()` (requires the array
  API version to be set to 2023.12), and via the other array API APIs that
  return devices (like `x.device`). These devices do not correspond to any
  actual hardware and only exist for testing array API device semantics; for
  instance, implicitly combining arrays on different devices results in an
  exception. (Thanks to [@betatim](https://github.com/betatim)).

### Minor Changes

- Avoid implicitly relying on `__array__` in some places. These changes should
  not be usef visible.

## 2.0.1 (2024-07-01)

### Minor Changes

- Re-allow iteration on 1-D arrays. A change from 2.0 fixed iter() raising on
  n-D arrays but also made 1-D arrays raise. The standard does not explicitly
  disallow iteration on 1-D arrays, and the default Python `__iter__`
  implementation allows it to work, so for now, it is kept intact as working.

## 2.0 (2024-06-27)

### Major Changes

- array-api-strict has a new set of [flags](array-api-strict-flags) that can
  be used to dynamically enable or disable features in array-api-strict. These
  flags allow you to change the supported array API version, enable or disable
  [extensions](https://data-apis.org/array-api/latest/extensions/index.html),
  enable or disable features that rely on data-dependent shapes, and enable or
  disable boolean indexing. Future versions may add additional flags to allow
  changing other optional or varying behaviors in the standard.

- Added experimental support for the
  [2023.12](https://data-apis.org/array-api/2023.12/changelog.html#v2023-12)
  version of the array API standard. The default version is still 2022.12, but
  the version can be changed to 2023.12 using the aforementioned flags, either
  by calling
  {func}`array_api_strict.set_array_api_strict_flags(api_version='2023.12')
  <array_api_strict.set_array_api_strict_flags>` or setting the environment
  variable {envvar}`ARRAY_API_STRICT_API_VERSION=2023.12
  <ARRAY_API_STRICT_API_VERSION>`. A future version of array-api-strict will
  change the default version to 2023.12.

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
