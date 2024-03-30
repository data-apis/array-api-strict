# array-api-strict

`array_api_strict` is a strict, minimal implementation of the [Python array
API](https://data-apis.org/array-api/latest/).

The purpose of array-api-strict is to provide an implementation of the array
API for consuming libraries to test against so they can be completely sure
their usage of the array API is portable.

It is *not* intended to be used by end-users. End-users of the array API
should just use their favorite array library (NumPy, CuPy, PyTorch, etc.) as
usual. It is also not intended to be used as a dependency by consuming
libraries. Consuming library code should use the
[array-api-compat](https://data-apis.org/array-api-compat/) package to
support the array API. Rather, it is intended to be used in the test suites of
consuming libraries to test their array API usage.

array-api-strict currently supports the 2022.12 version of the standard.
2023.12 support is planned and is tracked by [this
issue](https://github.com/data-apis/array-api-strict/issues/25).

See the documentation for more details https://data-apis.org/array-api-strict/
