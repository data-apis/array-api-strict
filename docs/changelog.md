# Changelog

## 2.4.1 (unreleased)

### Major Changes

- The array object defines `__array__` again when `__buffer__` is not available.

- Support for Python versions 3.10 and 3.11 has been reinstated.


### Minor Changes

- Arithmetic operations no longer accept NumPy arrays.

- Disallow `__setitem__` for invalid dtype combinations (e.g. setting a float value
into an integer array)


### Contributors

The following users contributed to this release:

Evgeni Burovski,
Guido Imperiale


## 2.4.0 (2025-06-16)

### Major Changes

- The array object no longer defines `__array__`, and conversion to NumPy relies on the
  buffer protocol, `__buffer__` instead. This change should be completely transparent to
  users. Please report any issues this change causes however.

- Support for Python versions 3.9-3.11 was dropped. The minimum supported Python
  versions is now 3.12. 


### Minor Changes

- `asarray` no longer accepts nested sequences of arrays. This is consistent with the
   standard, which only allows _a (possibly nested) sequence of Python scalars_. In most
   cases, this requires changing `asarray(list_of_arrays)` to `stack(list_of_arrays)`.
   Note that this effectively disallows code such as
   `asarray([x[i] for i in range(x.shape[0]])` since indexing 1D arrays produces 0D
   arrays, not python scalars.

- Fix fancy indexing in a mult-device setting. The indexed arrays and all indexer arrays
  must be the same device. Otherwise, an error is raised.

- Make `finfo` and `iinfo` accept arrays or dtypes, as required by the standard.

- Make `roll` only accept integers and tuples for the `shift` argument.

- Make `reshape` only accept tuples for the `shape` argument.

- Make testing of array iteration compatible with Python 3.14.


### Contributors

The following users contributed to this release:

Lumir Balhar,
Evgeni Burovski,
Joren Hammudoglu,
Tim Head,
Guido Imperiale,
Lucy Liu


## 2.3.1 (2025-03-20)

This is a bugfix release with no new features compared to 2.3. This release fixes an
issue with `where` for scalar arguments, found in downstream testing of the 2024.12
support.


## 2.3 (2025-02-27)

### Major Changes

- The default version of the array API standard is now 2024.12. Previous versions can
  still be enabled via the [flags API](array-api-strict-flags).

  Note that this support is still relatively untested. Please [report any
  issues](https://github.com/data-apis/array-api-strict/issues) you find.

- Binary elementwise functions now accept python scalars: the only requirement is that
  at least one of the arguments must be an array; the other argument may be either
  a python scalar or an array. Python scalars are handled in accordance with the
  type promotion rules, as specified by the standard.
  This change unifies the behavior of binary functions and their matching operators,
  (where available), such as `multiply(x1, x2)` and `__mul__(self, other)`.
  
  `where` accepts arrays or scalars as its 2nd and 3rd arguments, `x1` and `x2`.
  The first argument, `condition`, must be an array.

  `result_type` accepts arrays and scalars and computes the result dtype according
  to the promotion rules.

- Ergonomics of working with complex values has been improved:

  - binary operators accept complex scalars and real arrays and preserve the floating point
    precision: `1j*f32_array` returns a `complex64` array
  - `mean` accepts complex floating-point arrays.
  - `real` and `conj` accept numeric arguments, including real floating point data.
    Note that `imag` still requires its input to be a complex array.

- The following functions, new in the 2024.12 standard revision, are implemented:

  - `count_nonzero`
  - `cumulative_prod`

- `fftfreq` and `rfftfreq` functions accept a new `dtype` argument to control the
  the data type of their output.


### Minor Changes

- `vecdot` now conjugates the first argument, in accordance with the standard.

- `astype` now raises a `TypeError` instead of casting a complex floating-point
  array to a real-valued or an integral data type.

- `where` requires that its first argument, `condition` has a boolean data dtype,
  and raises a `TypeError` otherwise.

- `isdtype` raises a `TypeError` is its argument is not a dtype object.

- arrays created with `from_dlpack` now correctly set their `device` attribute.

- the build system now uses `pyproject.toml`, not `setup.py`.

### Contributors

The following users contributed to this release:

Aaron Meurer
Clément Robert
Guido Imperiale
Evgeni Burovski
Lucas Colley
Tim Head


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
