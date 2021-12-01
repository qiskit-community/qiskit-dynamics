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

"""Deprecated dispatch functions"""

import functools
from types import FunctionType
from typing import Optional, Callable
from qiskit.utils import deprecate_function
from qiskit_dynamics.dispatch.dispatcher import DispatcherError


@deprecate_function(
    "The `set_default_backend` function has been depreacted and will be removed "
    "next release. Use the class method `Array.set_default_backend(backend)` instead."
)
def set_default_backend(backend: Optional[str] = None):
    """Set the default array backend."""
    # pylint: disable = import-outside-toplevel
    from qiskit_dynamics.array.array import Array

    Array.set_default_backend(backend)


@deprecate_function(
    "The `default_backend` function has been depreacted and will be removed next "
    "release. Use the class method `Array.default_backend()` instead."
)
def default_backend():
    """Return the default array backend."""
    # pylint: disable = import-outside-toplevel
    from qiskit_dynamics.array.array import Array

    return Array.default_backend()


@deprecate_function(
    "The `backend_types` function has been depreacted and will be removed next release."
)
def backend_types():
    """Return tuple of array backend types"""
    # pylint: disable = import-outside-toplevel
    from qiskit_dynamics.array.array import DISPATCHER

    return DISPATCHER.registered_types()


@deprecate_function(
    "The `available_backends` function has been depreacted and will be removed next"
    "release. Use the class method `Array.available_backends()` instead."
)
def available_backends():
    """Return a tuple of available array backends"""
    # pylint: disable = import-outside-toplevel
    from qiskit_dynamics.array.array import DISPATCHER

    return DISPATCHER.registered_libraries()


@deprecate_function(
    "The `asarray` function has been depreacted and will be removed next"
    "release. Use the `Dispatcher().asarray` instead."
)
def asarray(
    array: any,
    dtype: Optional[any] = None,
    order: Optional[str] = None,
    backend: Optional[str] = None,
) -> any:
    """Convert input array to an array on the specified backend.

    This functions like `numpy.asarray` but optionally supports
    casting to other registered array backends.

    Args:
        array: An array_like input.
        dtype: Optional. The dtype of the returned array. This value
                must be supported by the specified array backend.
        order: Optional. The array order. This value must be supported
                by the specified array backend.
        backend: A registered array backend name. If None the
                    default array backend will be used.

    Returns:
        array: an array object of the form specified by the backend
                kwarg.
    """
    # pylint: disable = import-outside-toplevel
    from qiskit_dynamics.array.array import _asarray

    return _asarray(array, dtype=dtype, order=order, backend=backend)


def requires_backend(backend: str) -> Callable:
    """Return a function and class decorator for checking a backend is available.

    If the the required backend is not in the list of :func:`available_backends`
    any decorated function or method will raise an exception when called, and
    any decorated class will raise an exeption when its ``__init__`` is called.

    Args:
        backend: the backend name required by class or function.

    Returns:
        Callable: A decorator that may be used to specify that a function, class,
                  or class method requires a specific backend to be installed.
    """
    # pylint: disable = import-outside-toplevel
    from qiskit_dynamics.array.array import DISPATCHER

    def decorator(obj):
        """Specify that the decorated object requires a specifc Array backend."""

        def check_backend(descriptor):
            if backend not in DISPATCHER.registered_libraries():
                raise DispatcherError(
                    f"Array backend '{backend}' required by {descriptor} "
                    "is not installed. Please install the optional "
                    f"library '{backend}'."
                )

        # Decorate a function or method
        if isinstance(obj, FunctionType):

            @functools.wraps(obj)
            def decorated_func(*args, **kwargs):
                check_backend(f"function {obj}")
                return obj(*args, **kwargs)

            return decorated_func

        # Decorate a class
        elif isinstance(obj, type):

            obj_init = obj.__init__

            @functools.wraps(obj_init)
            def decorated_init(self, *args, **kwargs):
                check_backend(f"class {obj}")
                obj_init(self, *args, **kwargs)

            obj.__init__ = decorated_init
            return obj

        else:
            raise Exception(f"Cannot decorate object {obj} that is not a class or function.")

    return decorator
