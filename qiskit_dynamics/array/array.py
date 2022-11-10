# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Array Class"""

import copy
from functools import wraps
from types import BuiltinMethodType, MethodType
from typing import Any, Optional, Union, Tuple, Set
from numbers import Number

import numpy
from numpy.lib.mixins import NDArrayOperatorsMixin

from qiskit_dynamics.dispatch.dispatch import Dispatch, asarray

__all__ = ["Array"]


class Array(NDArrayOperatorsMixin):
    """Qiskit array class.

    This class provides a Numpy compatible wrapper to supported Python array libraries. Supported
    backends are ``"numpy"`` and ``"jax"``.
    """

    def __init__(
        self,
        data: Any,
        dtype: Optional[Any] = None,
        order: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        """Initialize an :class:`Array` container.

        Args:
            data: An ``array_like`` input. This can be an object of any type supported by the
                  registered :math:`asarray` method for the specified backend.
            dtype: Optional. The dtype of the returned array. This value must be supported by the
                   specified array backend.
            order: Optional. The array order. This value must be supported by the specified array
                   backend.
            backend: A registered array backend name. If ``None`` the default array backend will
                     be used.

        Raises:
            ValueError: If input cannot be converted to an :class:`Array`.
        """

        # Check if we can override setattr and
        # set _data and _backend directly
        if (
            isinstance(data, numpy.ndarray)
            and _is_numpy_backend(backend)
            and _is_equivalent_numpy_array(data, dtype, order)
        ):
            self.__dict__["_data"] = data
            self.__dict__["_backend"] = "numpy"
            return

        if hasattr(data, "__qiskit_array__"):
            array = data.__qiskit_array__(dtype=dtype, backend=backend)
            if not isinstance(array, Array):
                raise ValueError("object __qiskit_array__ method is not producing an Array")
            self._data = array._data
            self._backend = array._backend
            if dtype or order or (backend and backend != self._backend):
                if backend is None:
                    backend = self._backend
                else:
                    self._backend = backend
                self._data = asarray(self._data, dtype=dtype, order=order, backend=backend)
            return

        # Standard init
        self._data = asarray(data, dtype=dtype, order=order, backend=backend)
        self._backend = backend if backend else Dispatch.backend(self._data, subclass=True)

    @property
    def data(self):
        """The wrapped array data object."""
        return self._data

    @data.setter
    def data(self, value):
        self._data[:] = value

    @property
    def backend(self):
        """The backend of the wrapped array class."""
        return self._backend

    @backend.setter
    def backend(self, value: str):
        Dispatch.validate_backend(value)
        self._data = asarray(self._data, backend=value)
        self._backend = value

    @classmethod
    def set_default_backend(cls, backend: Union[str, None]):
        """Set the default array backend."""
        if backend is not None:
            Dispatch.validate_backend(backend)
        Dispatch.DEFAULT_BACKEND = backend

    @classmethod
    def default_backend(cls) -> str:
        """Return the default array backend."""
        return Dispatch.DEFAULT_BACKEND

    @classmethod
    def available_backends(cls) -> Set[str]:
        """Return a tuple of available array backends."""
        return Dispatch.REGISTERED_BACKENDS

    def __repr__(self):
        prefix = "Array("
        if self._backend == Dispatch.DEFAULT_BACKEND:
            suffix = ")"
        else:
            suffix = f"backend='{self._backend}')"
        return Dispatch.repr(self.backend)(self._data, prefix=prefix, suffix=suffix)

    def __copy__(self):
        """Return a shallow copy referencing the same wrapped data array"""
        return Array(self._data, backend=self.backend)

    def __deepcopy__(self, memo=None):
        """Return a deep copy with a copy of the wrapped data array"""
        return Array(copy.deepcopy(self._data), backend=self.backend)

    def __iter__(self):
        return iter(self._data)

    def __getstate__(self):
        return {"_data": self._data, "_backend": self._backend}

    def __setstate__(self, state):
        self._backend = state["_backend"]
        self._data = state["_data"]

    def __getitem__(self, key: str) -> Any:
        """Return value from wrapped array."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        """Return value of wrapped array."""
        self._data[key] = value

    def __setattr__(self, name: str, value: Any):
        """Set attribute of wrapped array."""
        if name in ("_data", "data", "_backend", "backend"):
            super().__setattr__(name, value)
        else:
            setattr(self._data, name, value)

    def __getattr__(self, name: str) -> Any:
        """Get attribute of wrapped array and convert to an :class:`Array`."""
        # Call attribute on inner array object
        attr = getattr(self._data, name)

        # If attribute is a function wrap the return values
        if isinstance(attr, (MethodType, BuiltinMethodType)):

            @wraps(attr)
            def wrapped_method(*args, **kwargs):
                return self._wrap(attr(*args, **kwargs))

            return wrapped_method

        # If return object is a backend array wrap result
        return self._wrap(attr)

    def __qiskit_array__(self, dtype=None, backend=None):
        if (backend and backend != self.backend) or (dtype and dtype != self.data.dtype):
            return Array(self.data, dtype=dtype, backend=backend)
        return self

    def __array__(self, dtype=None) -> numpy.ndarray:
        if isinstance(self._data, numpy.ndarray) and (dtype is None or dtype == self._data.dtype):
            return self._data
        return numpy.asarray(self._data, dtype=dtype)

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def __int__(self):
        """Convert size 1 array to an int."""
        if numpy.size(self) != 1:
            raise TypeError("only size-1 Arrays can be converted to Python scalars.")
        return int(self._data)

    def __float__(self):
        """Convert size 1 array to a float."""
        if numpy.size(self) != 1:
            raise TypeError("only size-1 Arrays can be converted to Python scalars.")
        return float(self._data)

    def __complex__(self):
        """Convert size 1 array to a complex."""
        if numpy.size(self) != 1:
            raise TypeError("only size-1 Arrays can be converted to Python scalars.")
        return complex(self._data)

    @staticmethod
    def _wrap(obj: Union[Any, Tuple[Any]], backend: Optional[str] = None) -> Union[Any, Tuple[Any]]:
        """Wrap return array backend objects as :class:`Array` objects."""
        if isinstance(obj, tuple):
            return tuple(
                Array(x, backend=backend) if isinstance(x, Dispatch.REGISTERED_TYPES) else x
                for x in obj
            )
        if isinstance(obj, Dispatch.REGISTERED_TYPES):
            return Array(obj, backend=backend)
        return obj

    @classmethod
    def _unwrap(cls, obj):
        """Unwrap an Array or list of :class:`Array` objects."""
        if isinstance(obj, Array):
            return obj._data
        if isinstance(obj, tuple):
            return tuple(cls._unwrap(i) for i in obj)
        if isinstance(obj, list):
            return list(cls._unwrap(i) for i in obj)
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Dispatcher for NumPy ufuncs to support the wrapped array backend."""
        out = kwargs.get("out", tuple())

        for i in inputs + out:
            # Only support operations with instances of REGISTERED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(i, Dispatch.REGISTERED_TYPES + (Array, Number)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = self._unwrap(inputs)
        if out:
            kwargs["out"] = self._unwrap(out)

        # Get implementation for backend
        backend = self.backend
        dispatch_func = Dispatch.array_ufunc(backend, ufunc, method)
        if dispatch_func == NotImplemented:
            return NotImplemented
        result = dispatch_func(*inputs, **kwargs)

        # Not sure what this case from NumPy docs is?
        if method == "at":
            return None

        # Wrap array results back into Array objects
        return self._wrap(result, backend=self.backend)

    def __array_function__(self, func, types, args, kwargs):
        """Dispatcher for NumPy array function to support the wrapped :class:`Array` backend."""
        if not all(issubclass(t, (Array,) + Dispatch.REGISTERED_TYPES) for t in types):
            return NotImplemented

        # Unwrap function Array arguments
        args = self._unwrap(args)
        out = kwargs.get("out", tuple())
        if out:
            kwargs["out"] = self._unwrap(out)

        # Get implementation for backend
        backend = self.backend
        dispatch_func = Dispatch.array_function(backend, func)
        if dispatch_func == NotImplemented:
            return NotImplemented
        result = dispatch_func(*args, **kwargs)
        return self._wrap(result, backend=self.backend)


def _is_numpy_backend(backend: Optional[str] = None):
    return backend == "numpy" or (not backend and Dispatch.DEFAULT_BACKEND == "numpy")


def _is_equivalent_numpy_array(data: Any, dtype: Optional[Any] = None, order: Optional[str] = None):
    return (not dtype or dtype == data.dtype) and (
        not order
        or (order == "C" and data.flags["C_CONTIGUOUS"])
        or (order == "F" and data.flags["F_CONTIGUOUS"])
    )
