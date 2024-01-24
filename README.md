# array-api-strict

`array_api_strict` is a strict, minimal implementation of the [Python array
API](https://data-apis.org/array-api/latest/)

The purpose of array-api-strict is to provide an implementation of the array
API for consuming libraries to test against so they can be completely sure
their usage of the array API is portable.

It is *not* intended to be used by end-users. End-users of the array API
should just use their favorite array library (NumPy, CuPy, PyTorch, etc.) as
usual. It is also not intended to be used as a dependency by consuming
libraries. Consuming library code should use the
[array-api-compat](https://github.com/data-apis/array-api-compat) package to
support the array API. Rather, it is intended to be used in the test suites of
consuming libraries to test their array API usage.

## Install

`array-api-strict` is available on both
[PyPI](https://pypi.org/project/array-api-strict/)

```
python -m pip install array-api-strict
```

and [Conda-forge](https://anaconda.org/conda-forge/array-api-strict)

```
conda install --channel conda-forge array-api-strict
```

array-api-strict supports NumPy 1.26 and (the upcoming) NumPy 2.0.

## Rationale

The array API has many functions and behaviors that are required to be
implemented by conforming libraries, but it does not, in most cases, disallow
implementing additional functions, keyword arguments, and behaviors that
aren't explicitly required by the standard.

However, this poses a problem for consumers of the array API, as they may
accidentally use a function or rely on a behavior which just happens to be
implemented in every array library they test against (e.g., NumPy and
PyTorch), but isn't required by the standard and may not be included in other
libraries.

array-api-strict solves this problem by providing a strict, minimal
implementation of the array API standard. Only those functions and behaviors
that are explicitly *required* by the standard are implemented. For example,
most NumPy functions accept Python scalars as inputs:

```py
>>> import numpy as np
>>> np.sin(0.0)
0.0
```

However, the standard only specifies function inputs on `Array` objects. And
indeed, some libraries, such as PyTorch, do not allow this:

```py
>>> import torch
>>> torch.sin(0.0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: sin(): argument 'input' (position 1) must be Tensor, not float
```

In array-api-strict, this is also an error:

```py
>>> import array_api_strict as xp
>>> xp.sin(0.0)
Traceback (most recent call last):
...
AttributeError: 'float' object has no attribute 'dtype'
```

Here is an (incomplete) list of the sorts of ways that array-api-strict is
strict/minimal:

- Only those functions and methods that are [defined in the
  standard](https://data-apis.org/array-api/latest/API_specification/index.html)
  are included.

- In those functions, only the keyword-arguments that are defined by the
  standard are included. All signatures in array-api-strict use
  [positional-only
  arguments](https://data-apis.org/array-api/latest/API_specification/function_and_method_signatures.html#function-and-method-signatures).
  As noted above, only `array_api_strict` array objects are accepted by
  functions, except in the places where the standard allows Python scalars
  (i.e., functions do not automatically call `asarray` on their inputs).

- Only those [dtypes that are defined in the
  standard](https://data-apis.org/array-api/latest/API_specification/data_types.html)
  are included.

- All functions and methods reject inputs if the standard does not *require*
  the input dtype(s) to be supported. This is one of the most restrictive
  aspects of the library. For example, in NumPy, most transcendental functions
  like `sin` will accept integer array inputs, but the [standard only requires
  them to accept floating-point
  inputs](https://data-apis.org/array-api/latest/API_specification/generated/array_api.sin.html#array_api.sin),
  so in array-api-strict, `sin(integer_array)` will raise an exception.

- The
  [indexing](https://data-apis.org/array-api/latest/API_specification/indexing.html)
  semantics required by the standard are limited compared to those implemented
  by NumPy (e.g., out-of-bounds slices are not supported, integer array
  indexing is not supported, only a single boolean array index is supported).

- There are no distinct "scalar" objects as in NumPy. There are only 0-D
  arrays.

- Dtype objects are just empty objects that only implement [equality
  comparison](https://data-apis.org/array-api/latest/API_specification/generated/array_api.data_types.__eq__.html).
  The way to access dtype objects in the standard is by name, like
  `xp.float32`.

- The array object type itself is private and should not be accessed.
  Subclassing or otherwise trying to directly initialize this object is not
  supported. Arrays should be created with one of the [array creation
  functions](https://data-apis.org/array-api/latest/API_specification/creation_functions.html)
  such as `asarray`.

## Caveats

array-api-strict is a thin pure Python wrapper around NumPy. NumPy 2.0 fully
supports the array API but NumPy 1.26 does not, so many behaviors are wrapped
in NumPy 1.26 to provide array API compatible behavior. Although it is based
on NumPy, mixing NumPy arrays with array-api-strict arrays is not supported.
This should generally raise an error, as it indicates a potential portability
issue, but this hasn't necessarily been tested thoroughly.

1. array-api-strict is validated against the [array API test
   suite](https://github.com/data-apis/array-api-tests). However, there may be
   a few minor instances where NumPy deviates from the standard in a way that
   is inconvenient to workaround in array-api-strict, since it aims to remain
   pure Python. You can see the full list of tests that are known to fail in
   the [xfails
   file](https://github.com/data-apis/array-api-strict/blob/main/array-api-tests-xfails.txt).

    The most notable of these is that in NumPy 1.26, the `copy=False` flag is
    not implemented for `asarray` and therefore `array_api_strict` raises
    `NotImplementedError` in that case.

2. Since NumPy is a CPU-only library, the [device
   support](https://data-apis.org/array-api/latest/design_topics/device_support.html)
   in array-api-strict is superficial only. `x.device` is always a (private)
   `CPU_DEVICE` object, and `device` keywords to creation functions only
   accept either this object or `None`. A future version of array-api-strict
   [may add support for a CuPy
   backend](https://github.com/data-apis/array-api-strict/issues/5) so that
   more significant device support can be tested.

3. Although only array types are expected in array-api-strict functions,
   currently most functions do not do extensive type checking on their inputs,
   so a sufficiently duck-typed object may pass through silently (or at best,
   you may get `AttributeError` instead of `TypeError`). However, all type
   signatures have type annotations (based on those from the standard), so
   this deviation may be tested with type checking. This [behavior may improve
   in the future](https://github.com/data-apis/array-api-strict/issues/6).

4. There are some behaviors in the standard that are not required to be
   implemented by libraries that cannot support [data dependent
   shapes](https://data-apis.org/array-api/latest/design_topics/data_dependent_output_shapes.html).
   This includes [the `unique_*`
   functions](https://data-apis.org/array-api/latest/API_specification/set_functions.html),
   [boolean array
   indexing](https://data-apis.org/array-api/latest/API_specification/indexing.html#boolean-array-indexing),
   and the
   [`nonzero`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.nonzero.html)
   function. array-api-strict currently implements all of these. In the
   future, [there may be a way to disable them](https://github.com/data-apis/array-api-strict/issues/7).

5. array-api-strict currently only supports the latest version of the array
   API standard. [This may change in the future depending on
   need](https://github.com/data-apis/array-api-strict/issues/8).

## Usage

TODO: Add a sample CI script here.

## Relationship to `numpy.array_api`

Previously this implementation was available as `numpy.array_api`, but it was
moved to a separate package for NumPy 2.0.

Note that the history of this repo prior to commit
fbefd42e4d11e9be20e0a4785f2619fc1aef1e7c was generated automatically
from the numpy git history, using the following
[git-filter-repo](https://github.com/newren/git-filter-repo) command:

```
git_filter_repo.py --path numpy/array_api/ --path-rename numpy/array_api:array_api_strict --replace-text <(echo -e "numpy.array_api==>array_api_strict\nfrom ..core==>from numpy.core\nfrom .._core==>from numpy._core\nfrom ..linalg==>from numpy.linalg\nfrom numpy import array_api==>import array_api_strict") --commit-callback 'commit.message = commit.message.rstrip() + b"\n\nOriginal NumPy Commit: " + commit.original_id'
```
