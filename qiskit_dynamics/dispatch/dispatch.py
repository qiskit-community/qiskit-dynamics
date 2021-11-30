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

from typing import Optional, Callable
from qiskit.utils import deprecate_function
from .functions import requires_library
from .default_dispatcher import DEFAULT_DISPATCHER


@deprecate_function(
    "The `set_default_backend` function has been depreacted and will be removed "
    "next release. Use the class method `Array.set_default_backend` instead."
)
def set_default_backend(backend: Optional[str] = None):
    """Set the default array backend."""
    # pylint: disable = import-outside-toplevel
    from qiskit_dynamics.array.array import Array

    Array.set_default_backend(backend)


@deprecate_function(
    "The `default_backend` function has been depreacted and will be removed next "
    "release. Use the class method `Array.default_backend` instead."
)
def default_backend():
    """Return the default array backend."""
    # pylint: disable = import-outside-toplevel
    from qiskit_dynamics.array.array import Array

    return Array.default_backend()


@deprecate_function(
    "The `backend_types` function has been depreacted and will be "
    "removed next release. Use the `registered_types` method instead."
)
def backend_types():
    """Return tuple of array backend types"""
    return DEFAULT_DISPATCHER.registered_types


@deprecate_function(
    "The `available_backends` function has been depreacted and will be "
    "removed next release. Use the `available_libraries` method instead."
)
def available_backends():
    """Return a tuple of available array backends"""
    return DEFAULT_DISPATCHER.registered_libraries


@deprecate_function(
    "The `requires_backends` decorator has been depreacted and will be "
    "removed next release. Use the `requires_library` decorator instead."
)
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
    return requires_library(backend)
