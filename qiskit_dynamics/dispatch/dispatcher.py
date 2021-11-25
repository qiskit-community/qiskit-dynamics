# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Array library dispatcher functions"""

import functools
from typing import Union, Callable
from .exceptions import DispatchError
from .register import CACHE, register_type

__all__ = ["dispatcher", "dispatch_function", "infer_library"]


def dispatcher(lib: str) -> Callable:
    """Return the dispatcher for the specified library.

    Args:
        lib: An array library name

    Returns:
        The function dispatcher for the specified library.

    Raises:
        DispatchError: if the input library is not registered.
    """
    # Return cached dispatcher
    if lib in CACHE.DISPATCHERS:
        return CACHE.DISPATCHERS[lib]

    if lib not in CACHE.REGISTERED_LIBS:
        raise DispatchError(f"Unregistered array library '{lib}'.")

    @functools.lru_cache()
    def _function_dispatcher(name: Union[str, Callable]) -> Callable:

        # Convert input to string if callable
        if not isinstance(name, str):
            name = name.__name__

        # Check if function is already registered
        if name in CACHE.REGISTERED_FUNCTIONS[lib]:
            return CACHE.REGISTERED_FUNCTIONS[lib][name]

        # If not registered try and import from registed modules
        for mod in CACHE.REGISTERED_MODULES[lib]:
            try:
                func = getattr(mod, name)
                CACHE.REGISTERED_FUNCTIONS[lib][name] = func
                return func
            except AttributeError:
                pass

        # Couldn't find a matching function
        raise DispatchError(
            f"Unable to dispatch function '{name}' for library '{lib}'"
            " using registered functions and modules."
        )

    # Add to cached dispatchers and return
    CACHE.DISPATCHERS[lib] = _function_dispatcher
    return CACHE.DISPATCHERS[lib]


def dispatch_function(func: Union[Callable, str]) -> Callable:
    """Return an automatically dispatching version of input function.

    .. note::

        The first argument of the `func` must be an array. The type
        of this array is used to determine the library for the
        dispatched function.

    Args:
        func: A library function to convert to dispatched function.
              This functions name will be used for the returned
              function.

    Returns:
        The automatically dispatching function.
    """
    func_name = func if isinstance(func, str) else func.__name__

    # Define dispatching wrapper function
    def _function(array, *args, **kwargs):
        lib = infer_library(array)
        if lib is NotImplemented:
            raise DispatchError(
                f"{type(array)} is not a registered type for any registered array libraries."
            )
        lib_func = dispatcher(lib)(func_name)
        return lib_func(array, *args, **kwargs)

    _function.__name__ = func_name
    _function.__qualname__ = func_name

    if isinstance(func, str):
        return _function

    # If input is a funcion, wrap the dispatched function
    # to copy docs and annotations
    return functools.wraps(func, assigned=["__doc__", "__annotations__"])(_function)


def infer_library(array: any) -> Union[str, None]:
    """Return the registered array library name of an array object.

    Args:
        array: an array object.

    Returns:
        The array library name if the array type is registered,
        or None otherwise.
    """
    return _infer_library_cached(type(array))


@functools.lru_cache()
def _infer_library_cached(array_type: type) -> Union[str, None]:
    """Return the registered library name of a array object.

    Args:
        array_type: an array type.

    Returns:
        The array library name if the array type is registered,
        or None otherwise.
    """
    if array_type in CACHE.REGISTERED_TYPES_LIB:
        return CACHE.REGISTERED_TYPES_LIB[array_type]

    for key, lib in CACHE.REGISTERED_TYPES_LIB.items():
        if issubclass(array_type, key):
            register_type(array_type, lib)
            return lib
    return NotImplemented
