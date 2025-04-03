"""
Wrapper class around the ndarray object for the array API standard.

The array API standard defines some behaviors differently than ndarray, in
particular, type promotion rules are different (the standard has no
value-based casting). The standard also specifies a more limited subset of
array methods and functionalities than are implemented on ndarray. Since the
goal of the array_api_strict namespace is to be a minimal implementation of the array
API standard, we need to define a separate wrapper class for the array_api_strict
namespace.

The standard compliant class is only a wrapper class. It is *not* a subclass
of ndarray.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator
from enum import IntEnum
from types import EllipsisType, ModuleType
from typing import Any, Final, Literal, SupportsIndex

import numpy as np
import numpy.typing as npt

from ._creation_functions import Undef, _undef, asarray
from ._dtypes import (
    DType,
    _all_dtypes,
    _boolean_dtypes,
    _complex_floating_dtypes,
    _dtype_categories,
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
    _real_floating_dtypes,
    _real_to_complex_map,
    _result_type,
)
from ._flags import get_array_api_strict_flags, set_array_api_strict_flags
from ._typing import PyCapsule


class Device:
    _device: Final[str]
    __slots__ = ("_device", "__weakref__")

    def __init__(self, device: str = "CPU_DEVICE"):
        if device not in ("CPU_DEVICE", "device1", "device2"):
            raise ValueError(f"The device '{device}' is not a valid choice.")
        self._device = device

    def __repr__(self) -> str:
        return f"array_api_strict.Device('{self._device}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Device):
            return False
        return self._device == other._device

    def __hash__(self) -> int:
        return hash(("Device", self._device))


CPU_DEVICE = Device()
ALL_DEVICES = (CPU_DEVICE, Device("device1"), Device("device2"))

# See https://github.com/data-apis/array-api-strict/issues/67 and the comment
# on __array__ below.
_allow_array = True


class Array:
    """
    n-d array object for the array API namespace.

    See the docstring of :py:obj:`np.ndarray <numpy.ndarray>` for more
    information.

    This is a wrapper around numpy.ndarray that restricts the usage to only
    those things that are required by the array API namespace. Note,
    attributes on this object that start with a single underscore are not part
    of the API specification and should only be used internally. This object
    should not be constructed directly. Rather, use one of the creation
    functions, such as asarray().

    """

    _array: npt.NDArray[Any]
    _dtype: DType
    _device: Device
    __slots__ = ("_array", "_dtype", "_device", "__weakref__")

    # Use a custom constructor instead of __init__, as manually initializing
    # this class is not supported API.
    @classmethod
    def _new(cls, x: npt.NDArray[Any] | np.generic, /, device: Device | None) -> Array:
        """
        This is a private method for initializing the array API Array
        object.

        Functions outside of the array_api_strict module should not use this
        method. Use one of the creation functions instead, such as
        ``asarray``.

        """
        obj = super().__new__(cls)
        # Note: The spec does not have array scalars, only 0-D arrays.
        if isinstance(x, np.generic):
            # Convert the array scalar to a 0-D array
            x = np.asarray(x)
        _dtype = DType(x.dtype)
        if _dtype not in _all_dtypes:
            raise TypeError(
                f"The array_api_strict namespace does not support the dtype '{x.dtype}'"
            )
        obj._array = x
        obj._dtype = _dtype
        if device is None:
            device = CPU_DEVICE
        obj._device = device
        return obj

    # Prevent Array() from working
    def __new__(cls, *args: object, **kwargs: object) -> Array:
        raise TypeError(
            "The array_api_strict Array object should not be instantiated directly. Use an array creation function, such as asarray(), instead."
        )

    # These functions are not required by the spec, but are implemented for
    # the sake of usability.

    def __repr__(self) -> str:
        """
        Performs the operation __repr__.
        """
        suffix = f", dtype={self.dtype}"
        if self.device != CPU_DEVICE:
            suffix += f", device={self.device})"
        else:
            suffix += ")"
        if 0 in self.shape:
            prefix = "empty("
            mid = str(self.shape)
        else:
            prefix = "Array("
            mid = np.array2string(self._array, separator=', ', prefix=prefix, suffix=suffix)
        return prefix + mid + suffix

    __str__ = __repr__

    # In the future, _allow_array will be set to False, which will disallow
    # __array__. This means calling `np.func()` on an array_api_strict array
    # will give an error. If we don't explicitly disallow it, NumPy defaults
    # to creating an object dtype array, which would lead to confusing error
    # messages at best and surprising bugs at worst. The reason for doing this
    # is that __array__ is not actually supported by the standard, so it can
    # lead to code assuming np.asarray(other_array) would always work in the
    # standard.
    #
    # This was implemented historically for compatibility, and removing it has
    # caused issues for some libraries (see
    # https://github.com/data-apis/array-api-strict/issues/67).
    def __array__(
        self, dtype: None | np.dtype[Any] = None, copy: None | bool = None
    ) -> npt.NDArray[Any]:
        # We have to allow this to be internally enabled as there's no other
        # easy way to parse a list of Array objects in asarray().
        if _allow_array:
            if self._device != CPU_DEVICE:
                raise RuntimeError(f"Can not convert array on the '{self._device}' device to a Numpy array.")
            # copy keyword is new in 2.0.0; for older versions don't use it
            # retry without that keyword.
            if np.__version__[0] < '2':
                return np.asarray(self._array, dtype=dtype)
            elif np.__version__.startswith('2.0.0-dev0'):
                # Handle dev version for which we can't know based on version
                # number whether or not the copy keyword is supported.
                try:
                    return np.asarray(self._array, dtype=dtype, copy=copy)
                except TypeError:
                    return np.asarray(self._array, dtype=dtype)
            else:
                return np.asarray(self._array, dtype=dtype, copy=copy)
        raise ValueError("Conversion from an array_api_strict array to a NumPy ndarray is not supported")

    # These are various helper functions to make the array behavior match the
    # spec in places where it either deviates from or is more strict than
    # NumPy behavior

    def _check_allowed_dtypes(
        self, other: Array | bool | int | float | complex, dtype_category: str, op: str
    ) -> Array:
        """
        Helper function for operators to only allow specific input dtypes

        Use like

            other = self._check_allowed_dtypes(other, 'numeric', '__add__')
            if other is NotImplemented:
                return other
        """

        if self.dtype not in _dtype_categories[dtype_category]:
            raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        if isinstance(other, (bool, int, float, complex)):
            other = self._promote_scalar(other)
        elif isinstance(other, Array):
            if other.dtype not in _dtype_categories[dtype_category]:
                raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        else:
            return NotImplemented

        # This will raise TypeError for type combinations that are not allowed
        # to promote in the spec (even if the NumPy array operator would
        # promote them).
        res_dtype = _result_type(self.dtype, other.dtype)
        if op.startswith("__i"):
            # Note: NumPy will allow in-place operators in some cases where
            # the type promoted operator does not match the left-hand side
            # operand. For example,

            # >>> a = np.array(1, dtype=np.int8)
            # >>> a += np.array(1, dtype=np.int16)

            # The spec explicitly disallows this.
            if res_dtype != self.dtype:
                raise TypeError(
                    f"Cannot perform {op} with dtypes {self.dtype} and {other.dtype}"
                )

        return other

    def _check_device(self, other: Array | bool | int | float | complex) -> None:
        """Check that other is on a device compatible with the current array"""
        if isinstance(other, (bool, int, float, complex)):
            return
        elif isinstance(other, Array):
            if self.device != other.device:
                raise ValueError(f"Arrays from two different devices ({self.device} and {other.device}) can not be combined.")
        else:
            raise TypeError(f"Expected Array | python scalar; got {type(other)}")

    # Helper function to match the type promotion rules in the spec
    def _promote_scalar(self, scalar: bool | int | float | complex) -> Array:
        """
        Returns a promoted version of a Python scalar appropriate for use with
        operations on self.

        This may raise an OverflowError in cases where the scalar is an
        integer that is too large to fit in a NumPy integer dtype, or
        TypeError when the scalar type is incompatible with the dtype of self.
        """
        from ._data_type_functions import iinfo

        target_dtype = self.dtype
        # Note: Only Python scalar types that match the array dtype are
        # allowed.
        if isinstance(scalar, bool):
            if self.dtype not in _boolean_dtypes:
                raise TypeError(
                    "Python bool scalars can only be promoted with bool arrays"
                )
        elif isinstance(scalar, int):
            if self.dtype in _boolean_dtypes:
                raise TypeError(
                    "Python int scalars cannot be promoted with bool arrays"
                )
            if self.dtype in _integer_dtypes:
                info = iinfo(self.dtype)
                if not (info.min <= scalar <= info.max):
                    raise OverflowError(
                        "Python int scalars must be within the bounds of the dtype for integer arrays"
                    )
            # int + array(floating) is allowed
        elif isinstance(scalar, float):
            if self.dtype not in _floating_dtypes:
                raise TypeError(
                    "Python float scalars can only be promoted with floating-point arrays."
                )
        elif isinstance(scalar, complex):
            if self.dtype not in _floating_dtypes:
                raise TypeError(
                    "Python complex scalars can only be promoted with floating-point arrays."
                )
            # 1j * array(floating) is allowed
            if self.dtype in _real_floating_dtypes:
                target_dtype = _real_to_complex_map[self.dtype]
        else:
            raise TypeError("'scalar' must be a Python scalar")

        # Note: scalars are unconditionally cast to the same dtype as the
        # array.

        # Note: the spec only specifies integer-dtype/int promotion
        # behavior for integers within the bounds of the integer dtype.
        # Outside of those bounds we use the default NumPy behavior (either
        # cast or raise OverflowError).
        return Array._new(np.array(scalar, dtype=target_dtype._np_dtype), device=self.device)

    @staticmethod
    def _normalize_two_args(x1: Array, x2: Array) -> tuple[Array, Array]:
        """
        Normalize inputs to two arg functions to fix type promotion rules

        NumPy deviates from the spec type promotion rules in cases where one
        argument is 0-dimensional and the other is not. For example:

        >>> import numpy as np
        >>> a = np.array([1.0], dtype=np.float32)
        >>> b = np.array(1.0, dtype=np.float64)
        >>> np.add(a, b) # The spec says this should be float64
        array([2.], dtype=float32)

        To fix this, we add a dimension to the 0-dimension array before passing it
        through. This works because a dimension would be added anyway from
        broadcasting, so the resulting shape is the same, but this prevents NumPy
        from not promoting the dtype.
        """
        # Another option would be to use signature=(x1.dtype, x2.dtype, None),
        # but that only works for ufuncs, so we would have to call the ufuncs
        # directly in the operator methods. One should also note that this
        # sort of trick wouldn't work for functions like searchsorted, which
        # don't do normal broadcasting, but there aren't any functions like
        # that in the array API namespace.
        if x1.ndim == 0 and x2.ndim != 0:
            # The _array[None] workaround was chosen because it is relatively
            # performant. broadcast_to(x1._array, x2.shape) is much slower. We
            # could also manually type promote x2, but that is more complicated
            # and about the same performance as this.
            x1 = Array._new(x1._array[None], device=x1.device)
        elif x2.ndim == 0 and x1.ndim != 0:
            x2 = Array._new(x2._array[None], device=x2.device)
        return (x1, x2)

    # Note: A large fraction of allowed indices are disallowed here (see the
    # docstring below)
    def _validate_index(
        self,
        key: (
            int
            | slice
            | EllipsisType
            | Array
            | tuple[int | slice | EllipsisType | Array | None, ...]
        ),
        op: Literal["getitem", "setitem"] = "getitem",
    ) -> None:
        """
        Validate an index according to the array API.

        The array API specification only requires a subset of indices that are
        supported by NumPy. This function will reject any index that is
        allowed by NumPy but not required by the array API specification. We
        always raise ``IndexError`` on such indices (the spec does not require
        any specific behavior on them, but this makes the NumPy array API
        namespace a minimal implementation of the spec). See
        https://data-apis.org/array-api/latest/API_specification/indexing.html
        for the full list of required indexing behavior

        This function raises IndexError if the index ``key`` is invalid. It
        only raises ``IndexError`` on indices that are not already rejected by
        NumPy, as NumPy will already raise the appropriate error on such
        indices. ``shape`` may be None, in which case, only cases that are
        independent of the array shape are checked.

        The following cases are allowed by NumPy, but not specified by the array
        API specification:

        - Indices to not include an implicit ellipsis at the end. That is,
          every axis of an array must be explicitly indexed or an ellipsis
          included. This behaviour is sometimes referred to as flat indexing.

        - The start and stop of a slice may not be out of bounds. In
          particular, for a slice ``i:j:k`` on an axis of size ``n``, only the
          following are allowed:

          - ``i`` or ``j`` omitted (``None``).
          - ``-n <= i <= max(0, n - 1)``.
          - For ``k > 0`` or ``k`` omitted (``None``), ``-n <= j <= n``.
          - For ``k < 0``, ``-n - 1 <= j <= max(0, n - 1)``.

        - Boolean array indices are not allowed as part of a larger tuple
          index.

        - Integer array indices are not allowed (with the exception of 0-D
          arrays, which are treated the same as scalars).

        Additionally, it should be noted that indices that would return a
        scalar in NumPy will return a 0-D array. Array scalars are not allowed
        in the specification, only 0-D arrays. This is done in the
        ``Array._new`` constructor, not this function.

        """
        _key = key if isinstance(key, tuple) else (key,)
        for i in _key:
            if isinstance(i, bool) or not (
                isinstance(i, SupportsIndex)  # i.e. ints
                or isinstance(i, slice)
                or i == Ellipsis
                or i is None
                or isinstance(i, Array)
                or isinstance(i, np.ndarray)
            ):
                raise IndexError(
                    f"Single-axes index {i} has {type(i)=}, but only "
                    "integers, slices (:), ellipsis (...), newaxis (None), "
                    "zero-dimensional integer arrays and boolean arrays "
                    "are specified in the Array API."
                )
            if op == "setitem":
                if isinstance(i, Array) and i.dtype in _integer_dtypes:
                    raise IndexError("Fancy indexing __setitem__ is not supported.")

        nonexpanding_key = []
        single_axes = []
        n_ellipsis = 0
        key_has_mask = False
        key_has_index_array = False
        key_has_slices = False
        for i in _key:
            if i is not None:
                nonexpanding_key.append(i)
                if isinstance(i, np.ndarray):
                    raise IndexError("Index arrays for array_api_strict must be array_api_strict arrays")
                if isinstance(i, Array):
                    if i.dtype in _boolean_dtypes:
                        key_has_mask = True
                    elif i.dtype in _integer_dtypes:
                        key_has_index_array = True
                    single_axes.append(i)
                else:
                    # i must not be an array here, to avoid elementwise equals
                    if i == Ellipsis:
                        n_ellipsis += 1
                    else:
                        single_axes.append(i)
                        if isinstance(i, slice):
                            key_has_slices = True

        n_single_axes = len(single_axes)
        if n_ellipsis > 1:
            return  # handled by ndarray
        elif n_ellipsis == 0:
            # Note boolean masks must be the sole index, which we check for
            # later on.
            if not key_has_mask and n_single_axes < self.ndim:
                raise IndexError(
                    f"{self.ndim=}, but the multi-axes index only specifies "
                    f"{n_single_axes} dimensions. If this was intentional, "
                    "add a trailing ellipsis (...) which expands into as many "
                    "slices (:) as necessary - this is what np.ndarray arrays "
                    "implicitly do, but such flat indexing behaviour is not "
                    "specified in the Array API."
                )

        if (key_has_index_array and (n_ellipsis > 0 or key_has_slices or key_has_mask)):
            raise IndexError(
                "Integer index arrays are only allowed with integer indices; "
                f"got {key}."
            )

        if n_ellipsis == 0:
            indexed_shape = self.shape
        else:
            ellipsis_start = None
            for pos, i in enumerate(nonexpanding_key):
                if not (isinstance(i, Array) or isinstance(i, np.ndarray)):
                    if i == Ellipsis:
                        ellipsis_start = pos
                        break
            assert ellipsis_start is not None  # sanity check
            ellipsis_end = self.ndim - (n_single_axes - ellipsis_start)
            indexed_shape = (
                self.shape[:ellipsis_start] + self.shape[ellipsis_end:]
            )
        for i, side in zip(single_axes, indexed_shape):
            if isinstance(i, slice):
                if side == 0:
                    f_range = "0 (or None)"
                else:
                    f_range = f"between -{side} and {side - 1} (or None)"
                if i.start is not None:
                    try:
                        start = operator.index(i.start)
                    except TypeError:
                        pass  # handled by ndarray
                    else:
                        if not (-side <= start <= side):
                            raise IndexError(
                                f"Slice {i} contains {start=}, but should be "
                                f"{f_range} for an axis of size {side} "
                                "(out-of-bounds starts are not specified in "
                                "the Array API)"
                            )
                if i.stop is not None:
                    try:
                        stop = operator.index(i.stop)
                    except TypeError:
                        pass  # handled by ndarray
                    else:
                        if not (-side <= stop <= side):
                            raise IndexError(
                                f"Slice {i} contains {stop=}, but should be "
                                f"{f_range} for an axis of size {side} "
                                "(out-of-bounds stops are not specified in "
                                "the Array API)"
                            )
            elif isinstance(i, Array):
                if i.dtype in _boolean_dtypes:
                    if len(_key) != 1:
                        assert isinstance(key, tuple)  # sanity check
                        raise IndexError(
                            f"Single-axes index {i} is a boolean array and "
                            f"{len(key)=}, but masking is only specified in the "
                            "Array API when the array is the sole index."
                        )
                    if not get_array_api_strict_flags()['boolean_indexing']:
                        raise RuntimeError(
                            "The boolean_indexing flag has been disabled for array-api-strict"
                        )
            elif isinstance(i, tuple):
                raise IndexError(
                    f"Single-axes index {i} is a tuple, but nested tuple "
                    "indices are not specified in the Array API."
                )

    # Everything below this line is required by the spec.

    def __abs__(self) -> Array:
        """
        Performs the operation __abs__.
        """
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __abs__")
        res = self._array.__abs__()
        return self.__class__._new(res, device=self.device)

    def __add__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __add__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "numeric", "__add__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__add__(other._array)
        return self.__class__._new(res, device=self.device)

    def __and__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __and__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer or boolean", "__and__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__and__(other._array)
        return self.__class__._new(res, device=self.device)

    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """
        Return the array_api_strict namespace corresponding to api_version.

        The default API version is '2022.12'. Note that '2021.12' is supported,
        but currently identical to '2022.12'.

        For array_api_strict, calling this function with api_version will set
        the API version for the array_api_strict module globally. This can
        also be achieved with the
        {func}`array_api_strict.set_array_api_strict_flags` function. If you
        want to only set the version locally, use the
        {class}`array_api_strict.ArrayApiStrictFlags` context manager.

        """
        set_array_api_strict_flags(api_version=api_version)
        import array_api_strict
        return array_api_strict

    def __bool__(self) -> bool:
        """
        Performs the operation __bool__.
        """
        # Note: This is an error here.
        if self._array.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        res = self._array.__bool__()
        return res

    def __complex__(self) -> complex:
        """
        Performs the operation __complex__.
        """
        # Note: This is an error here.
        if self._array.ndim != 0:
            raise TypeError("complex is only allowed on arrays with 0 dimensions")
        res = self._array.__complex__()
        return res

    def __dlpack__(
        self,
        /,
        *,
        stream: Any = None,
        max_version: tuple[int, int] | None | Undef = _undef,
        dl_device: tuple[IntEnum, int] | None | Undef = _undef,
        copy: bool | None | Undef = _undef,
    ) -> PyCapsule:
        """
        Performs the operation __dlpack__.
        """
        if get_array_api_strict_flags()['api_version'] < '2023.12':
            if max_version is not _undef:
                raise ValueError("The max_version argument to __dlpack__ requires at least version 2023.12 of the array API")
            if dl_device is not _undef:
                raise ValueError("The device argument to __dlpack__ requires at least version 2023.12 of the array API")
            if copy is not _undef:
                raise ValueError("The copy argument to __dlpack__ requires at least version 2023.12 of the array API")

        if np.lib.NumpyVersion(np.__version__) < '2.1.0':
            if max_version not in [_undef, None]:
                raise NotImplementedError("The max_version argument to __dlpack__ is not yet implemented")
            if dl_device not in [_undef, None]:
                raise NotImplementedError("The device argument to __dlpack__ is not yet implemented")
            if copy not in [_undef, None]:
                raise NotImplementedError("The copy argument to __dlpack__ is not yet implemented")

            return self._array.__dlpack__(stream=stream)
        else:
            kwargs = {'stream': stream}
            if max_version is not _undef:
                kwargs['max_version'] = max_version
            if dl_device is not _undef:
                kwargs['dl_device'] = dl_device
            if copy is not _undef:
                kwargs['copy'] = copy
            return self._array.__dlpack__(**kwargs)

    def __dlpack_device__(self) -> tuple[IntEnum, int]:
        """
        Performs the operation __dlpack_device__.
        """
        # Note: device support is required for this
        return self._array.__dlpack_device__()

    def __eq__(self, other: Array | bool | int | float | complex, /) -> Array:  # type: ignore[override]
        """
        Performs the operation __eq__.
        """
        self._check_device(other)
        # Even though "all" dtypes are allowed, we still require them to be
        # promotable with each other.
        other = self._check_allowed_dtypes(other, "all", "__eq__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__eq__(other._array)
        return self.__class__._new(res, device=self.device)

    def __float__(self) -> float:
        """
        Performs the operation __float__.
        """
        # Note: This is an error here.
        if self._array.ndim != 0:
            raise TypeError("float is only allowed on arrays with 0 dimensions")
        if self.dtype in _complex_floating_dtypes:
            raise TypeError("float is not allowed on complex floating-point arrays")
        res = self._array.__float__()
        return res

    def __floordiv__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __floordiv__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "real numeric", "__floordiv__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__floordiv__(other._array)
        return self.__class__._new(res, device=self.device)

    def __ge__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __ge__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "real numeric", "__ge__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__ge__(other._array)
        return self.__class__._new(res, device=self.device)

    def __getitem__(
        self,
        key: (
            int
            | slice
            | EllipsisType
            | Array
            | None
            | tuple[int | slice | EllipsisType | Array | None, ...]
        ),
        /,
    ) -> Array:
        """
        Performs the operation __getitem__.
        """
        # XXX Does key have to be on the same device? Is there an exception for CPU_DEVICE?
        # Note: Only indices required by the spec are allowed. See the
        # docstring of _validate_index
        self._validate_index(key, op="getitem")
        if isinstance(key, Array):
            key = (key,)
        np_key = key
        devices = {self.device}
        if isinstance(key, tuple):
            devices.update(
                [subkey.device for subkey in key if isinstance(subkey, Array)]
            )
            if len(devices) > 1:
                raise ValueError(
                    "Array indexing is only allowed when array to be indexed and all "
                    "indexing arrays are on the same device."
                )
            # Indexing self._array with array_api_strict arrays can be erroneous
            # e.g., when using non-default device
            np_key = tuple(
                subkey._array if isinstance(subkey, Array) else subkey for subkey in key
            )
        res = self._array.__getitem__(np_key)
        return self._new(res, device=self.device)

    def __gt__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __gt__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "real numeric", "__gt__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__gt__(other._array)
        return self.__class__._new(res, device=other.device)

    def __int__(self) -> int:
        """
        Performs the operation __int__.
        """
        # Note: This is an error here.
        if self._array.ndim != 0:
            raise TypeError("int is only allowed on arrays with 0 dimensions")
        if self.dtype in _complex_floating_dtypes:
            raise TypeError("int is not allowed on complex floating-point arrays")
        res = self._array.__int__()
        return res

    def __index__(self) -> int:
        """
        Performs the operation __index__.
        """
        res = self._array.__index__()
        return res

    def __invert__(self) -> Array:
        """
        Performs the operation __invert__.
        """
        if self.dtype not in _integer_or_boolean_dtypes:
            raise TypeError("Only integer or boolean dtypes are allowed in __invert__")
        res = self._array.__invert__()
        return self.__class__._new(res, device=self.device)

    def __iter__(self) -> Iterator[Array]:
        """
        Performs the operation __iter__.
        """
        # Manually disable iteration on higher dimensional arrays, since
        # __getitem__ raises IndexError on things like ones((3, 3))[0], which
        # causes list(ones((3, 3))) to give [].
        if self.ndim > 1:
            raise TypeError("array iteration is not allowed in array-api-strict")
        # Allow iteration for 1-D arrays. The array API doesn't strictly
        # define __iter__, but it doesn't disallow it. The default Python
        # behavior is to implement iter as a[0], a[1], ... when __getitem__ is
        # implemented, which implies iteration on 1-D arrays.
        return (Array._new(i, device=self.device) for i in self._array)

    def __le__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __le__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "real numeric", "__le__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__le__(other._array)
        return self.__class__._new(res, device=self.device)

    def __lshift__(self, other: Array | int, /) -> Array:
        """
        Performs the operation __lshift__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer", "__lshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__lshift__(other._array)
        return self.__class__._new(res, device=self.device)

    def __lt__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __lt__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "real numeric", "__lt__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__lt__(other._array)
        return self.__class__._new(res, device=self.device)

    def __matmul__(self, other: Array, /) -> Array:
        """
        Performs the operation __matmul__.
        """
        self._check_device(other)
        # matmul is not defined for scalars, but without this, we may get
        # the wrong error message from asarray.
        other = self._check_allowed_dtypes(other, "numeric", "__matmul__")
        if other is NotImplemented:
            return other
        res = self._array.__matmul__(other._array)
        return self.__class__._new(res, device=self.device)

    def __mod__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __mod__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "real numeric", "__mod__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__mod__(other._array)
        return self.__class__._new(res, device=self.device)

    def __mul__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __mul__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "numeric", "__mul__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__mul__(other._array)
        return self.__class__._new(res, device=self.device)

    def __ne__(self, other: Array | bool | int | float | complex, /) -> Array:  # type: ignore[override]
        """
        Performs the operation __ne__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "all", "__ne__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__ne__(other._array)
        return self.__class__._new(res, device=self.device)

    def __neg__(self) -> Array:
        """
        Performs the operation __neg__.
        """
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __neg__")
        res = self._array.__neg__()
        return self.__class__._new(res, device=self.device)

    def __or__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __or__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer or boolean", "__or__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__or__(other._array)
        return self.__class__._new(res, device=self.device)

    def __pos__(self) -> Array:
        """
        Performs the operation __pos__.
        """
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __pos__")
        res = self._array.__pos__()
        return self.__class__._new(res, device=self.device)

    def __pow__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __pow__.
        """
        from ._elementwise_functions import pow  # type: ignore[attr-defined]

        self._check_device(other)
        other = self._check_allowed_dtypes(other, "numeric", "__pow__")
        if other is NotImplemented:
            return other
        # Note: NumPy's __pow__ does not follow type promotion rules for 0-d
        # arrays, so we use pow() here instead.
        return pow(self, other)

    def __rshift__(self, other: Array | int, /) -> Array:
        """
        Performs the operation __rshift__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer", "__rshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rshift__(other._array)
        return self.__class__._new(res, device=self.device)

    def __setitem__(
        self,
        # Almost same as __getitem__ key but doesn't accept None
        # or integer arrays
        key: (
            int
            | slice
            | EllipsisType
            | Array
            | tuple[int | slice | EllipsisType, ...]
        ),
        value: Array | bool | int | float | complex,
        /,
    ) -> None:
        """
        Performs the operation __setitem__.
        """
        # Note: Only indices required by the spec are allowed. See the
        # docstring of _validate_index
        self._validate_index(key, op="setitem")
        # Indexing self._array with array_api_strict arrays can be erroneous
        np_key = key._array if isinstance(key, Array) else key
        self._array.__setitem__(np_key, asarray(value)._array)

    def __sub__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __sub__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "numeric", "__sub__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__sub__(other._array)
        return self.__class__._new(res, device=self.device)

    # PEP 484 requires int to be a subtype of float, but __truediv__ should
    # not accept int.
    def __truediv__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __truediv__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "floating-point", "__truediv__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__truediv__(other._array)
        return self.__class__._new(res, device=self.device)

    def __xor__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __xor__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer or boolean", "__xor__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__xor__(other._array)
        return self.__class__._new(res, device=self.device)

    def __iadd__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __iadd__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "numeric", "__iadd__")
        if other is NotImplemented:
            return other
        self._array.__iadd__(other._array)
        return self

    def __radd__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __radd__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "numeric", "__radd__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__radd__(other._array)
        return self.__class__._new(res, device=self.device)

    def __iand__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __iand__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer or boolean", "__iand__")
        if other is NotImplemented:
            return other
        self._array.__iand__(other._array)
        return self

    def __rand__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __rand__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer or boolean", "__rand__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rand__(other._array)
        return self.__class__._new(res, device=self.device)

    def __ifloordiv__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __ifloordiv__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "real numeric", "__ifloordiv__")
        if other is NotImplemented:
            return other
        self._array.__ifloordiv__(other._array)
        return self

    def __rfloordiv__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __rfloordiv__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "real numeric", "__rfloordiv__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rfloordiv__(other._array)
        return self.__class__._new(res, device=self.device)

    def __ilshift__(self, other: Array | int, /) -> Array:
        """
        Performs the operation __ilshift__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer", "__ilshift__")
        if other is NotImplemented:
            return other
        self._array.__ilshift__(other._array)
        return self

    def __rlshift__(self, other: Array | int, /) -> Array:
        """
        Performs the operation __rlshift__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer", "__rlshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rlshift__(other._array)
        return self.__class__._new(res, device=self.device)

    def __imatmul__(self, other: Array, /) -> Array:
        """
        Performs the operation __imatmul__.
        """
        # matmul is not defined for scalars, but without this, we may get
        # the wrong error message from asarray.
        other = self._check_allowed_dtypes(other, "numeric", "__imatmul__")
        if other is NotImplemented:
            return other
        self._check_device(other)
        res = self._array.__imatmul__(other._array)
        return self.__class__._new(res, device=self.device)

    def __rmatmul__(self, other: Array, /) -> Array:
        """
        Performs the operation __rmatmul__.
        """
        # matmul is not defined for scalars, but without this, we may get
        # the wrong error message from asarray.
        other = self._check_allowed_dtypes(other, "numeric", "__rmatmul__")
        if other is NotImplemented:
            return other
        self._check_device(other)
        res = self._array.__rmatmul__(other._array)
        return self.__class__._new(res, device=self.device)

    def __imod__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __imod__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__imod__")
        if other is NotImplemented:
            return other
        self._array.__imod__(other._array)
        return self

    def __rmod__(self, other: Array | int | float, /) -> Array:
        """
        Performs the operation __rmod__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__rmod__")
        if other is NotImplemented:
            return other
        self._check_device(other)
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rmod__(other._array)
        return self.__class__._new(res, device=self.device)

    def __imul__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __imul__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__imul__")
        if other is NotImplemented:
            return other
        self._array.__imul__(other._array)
        return self

    def __rmul__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __rmul__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__rmul__")
        if other is NotImplemented:
            return other
        self._check_device(other)
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rmul__(other._array)
        return self.__class__._new(res, device=self.device)

    def __ior__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __ior__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__ior__")
        if other is NotImplemented:
            return other
        self._array.__ior__(other._array)
        return self

    def __ror__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __ror__.
        """
        self._check_device(other)
        other = self._check_allowed_dtypes(other, "integer or boolean", "__ror__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__ror__(other._array)
        return self.__class__._new(res, device=self.device)

    def __ipow__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __ipow__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__ipow__")
        if other is NotImplemented:
            return other
        self._array.__ipow__(other._array)
        return self

    def __rpow__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __rpow__.
        """
        from ._elementwise_functions import pow  # type: ignore[attr-defined]

        other = self._check_allowed_dtypes(other, "numeric", "__rpow__")
        if other is NotImplemented:
            return other
        # Note: NumPy's __pow__ does not follow the spec type promotion rules
        # for 0-d arrays, so we use pow() here instead.
        return pow(other, self)

    def __irshift__(self, other: Array | int, /) -> Array:
        """
        Performs the operation __irshift__.
        """
        other = self._check_allowed_dtypes(other, "integer", "__irshift__")
        if other is NotImplemented:
            return other
        self._array.__irshift__(other._array)
        return self

    def __rrshift__(self, other: Array | int, /) -> Array:
        """
        Performs the operation __rrshift__.
        """
        other = self._check_allowed_dtypes(other, "integer", "__rrshift__")
        if other is NotImplemented:
            return other
        self._check_device(other)
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rrshift__(other._array)
        return self.__class__._new(res, device=self.device)

    def __isub__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __isub__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__isub__")
        if other is NotImplemented:
            return other
        self._array.__isub__(other._array)
        return self

    def __rsub__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __rsub__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__rsub__")
        if other is NotImplemented:
            return other
        self._check_device(other)
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rsub__(other._array)
        return self.__class__._new(res, device=self.device)

    def __itruediv__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __itruediv__.
        """
        other = self._check_allowed_dtypes(other, "floating-point", "__itruediv__")
        if other is NotImplemented:
            return other
        self._array.__itruediv__(other._array)
        return self

    def __rtruediv__(self, other: Array | int | float | complex, /) -> Array:
        """
        Performs the operation __rtruediv__.
        """
        other = self._check_allowed_dtypes(other, "floating-point", "__rtruediv__")
        if other is NotImplemented:
            return other
        self._check_device(other)
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rtruediv__(other._array)
        return self.__class__._new(res, device=self.device)

    def __ixor__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __ixor__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__ixor__")
        if other is NotImplemented:
            return other
        self._array.__ixor__(other._array)
        return self

    def __rxor__(self, other: Array | bool | int, /) -> Array:
        """
        Performs the operation __rxor__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__rxor__")
        if other is NotImplemented:
            return other
        self._check_device(other)
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rxor__(other._array)
        return self.__class__._new(res, device=self.device)

    def to_device(self, device: Device, /, stream: None = None) -> Array:
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        if device == self._device:
            return self
        elif isinstance(device, Device):
            arr = np.asarray(self._array, copy=True)
            return self.__class__._new(arr, device=device)
        raise ValueError(f"Unsupported device {device!r}")

    @property
    def dtype(self) -> DType:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.dtype <numpy.ndarray.dtype>`.

        See its docstring for more information.
        """
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    # Note: mT is new in array API spec (see matrix_transpose)
    @property
    def mT(self) -> Array:
        from ._linear_algebra_functions import matrix_transpose
        return matrix_transpose(self)

    @property
    def ndim(self) -> int:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.ndim <numpy.ndarray.ndim>`.

        See its docstring for more information.
        """
        return self._array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.shape <numpy.ndarray.shape>`.

        See its docstring for more information.
        """
        return self._array.shape

    @property
    def size(self) -> int:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.size <numpy.ndarray.size>`.

        See its docstring for more information.
        """
        return self._array.size

    @property
    def T(self) -> Array:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.T <numpy.ndarray.T>`.

        See its docstring for more information.
        """
        # Note: T only works on 2-dimensional arrays. See the corresponding
        # note in the specification:
        # https://data-apis.org/array-api/latest/API_specification/array_object.html#t
        if self.ndim != 2:
            raise ValueError("x.T requires x to have 2 dimensions. Use x.mT to transpose stacks of matrices and permute_dims() to permute dimensions.")
        return self.__class__._new(self._array.T, device=self.device)
