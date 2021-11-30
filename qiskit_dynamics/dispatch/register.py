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
"""Functions for registering items in dispatcher"""

import functools
from types import ModuleType, FunctionType, SimpleNamespace
from typing import Optional, Union, Callable, Set, Tuple

from qiskit_dynamics.dispatch.exceptions import DispatchError

__all__ = [
    "registered_libraries",
    "is_registered_library",
    "default_library",
    "set_default_library",
    "registered_types",
    "is_registered_type",
    "register_function",
    "register_module",
    "register_type",
]

# Global variables for caching dispatched functions, libraries, types
CACHE = SimpleNamespace()

# Registered array library names
CACHE.REGISTERED_LIBS = set()

# Set a default array library
CACHE.DEFAULT_LIB = None

# Map from registered type to array library
CACHE.REGISTERED_TYPES_LIB = {}

# Tuple of keys of registered types
CACHE.REGISTERED_TYPES = tuple()

# Map from library name to dictionary of function names to library functions
CACHE.REGISTERED_FUNCTIONS = {}

# Map from library name to module paths to look for functions
CACHE.REGISTERED_MODULES = {}

# Cached dispatchers for specific libaries so they can be shared
CACHE.DISPATCHERS = {}


def registered_libraries() -> Set[str]:
    """Return a set of registered array library names."""
    return CACHE.REGISTERED_LIBS


def is_registered_library(lib: str) -> bool:
    """Return True if input lib is a registered array library."""
    return lib in CACHE.REGISTERED_LIBS


def default_library() -> Union[str, None]:
    """Return the default array library or None if no default is set."""
    return CACHE.DEFAULT_LIB


def set_default_library(lib: str):
    """Return the default array library or None if no default is set."""
    if not is_registered_library(lib):
        raise DispatchError(
            f"Cannot set default library, '{lib}' is not a registered array library"
        )
    CACHE.DEFAULT_LIB = lib


def register_type(array_type: type, lib: Optional[str] = None):
    """Register an array type for dispatching array functions.

    Args:
        array_type: An array type to register for the array library.
        lib: Optional, a name string to identify the array library.
             If None this will be set as the base module name of the
             arrays module.
    """
    if lib is None:
        lib = _lib_from_module(array_type)
    _register_library(lib)

    CACHE.REGISTERED_TYPES_LIB[array_type] = lib
    CACHE.REGISTERED_TYPES = tuple(CACHE.REGISTERED_TYPES_LIB.keys())


def registered_types() -> Tuple[type]:
    """Return a tuple of registered array library types."""
    return CACHE.REGISTERED_TYPES


def is_registered_type(obj: any) -> bool:
    """Return True if object is a registered array library type."""
    return isinstance(obj, CACHE.REGISTERED_TYPES)


def register_module(module: ModuleType, lib: Optional[str] = None):
    """Register a module for looking up dispatched array functions.

    Args:
        module: A module, namespace, or class to look for attributes
                corresponding to the dispatched function name.
        lib: Optional, a name string to identify the array library.
             If None this will be set as the base module name of the
             arrays module.
    """
    if lib is None:
        lib = _lib_from_module(module)
    _register_library(lib)

    if module not in CACHE.REGISTERED_MODULES[lib]:
        CACHE.REGISTERED_MODULES[lib].append(module)
    if lib in CACHE.DISPATCHERS:
        CACHE.DISPATCHERS[lib].cache_clear()


def register_function(
    func: Optional[Callable] = None,
    name: Optional[Union[Callable, str]] = None,
    lib: Optional[str] = None,
) -> Optional[Callable]:
    """Register an array function for dispatching.

    Args:
        func: The function to dispatch to for the specified array library.
              If None this will return a decorator to apply to a function.
        name: Optional, a name for dispatching to this function. If None
              the name of the input function will be used.
        lib: Optional, a name string to identify the array library.
             If None this will be set as the base module name of the
             arrays module.

    Returns:
        If func is None returns a decorator for registering a function.
        Otherwise returns None.
    """
    decorator = _register_function_decorator(name=name, lib=lib)
    if func is None:
        return decorator
    return decorator(func)


def _register_function_decorator(
    name: Optional[Union[str, Callable]] = None, lib: Optional[str] = None
) -> Callable:
    """Return a decorator to register a function.

    Args:
        name: the name for dispatching to this function.
        lib: The name string to identify the array library.

    Returns:
        A function decorator to register a function if dispatched_function
        is None.
    """
    if not isinstance(name, str):
        name = name.__name__

    def decorator(func):
        func_lib = _lib_from_module(func) if lib is None else lib
        _register_library(func_lib)
        CACHE.REGISTERED_FUNCTIONS[lib][name] = func
        if lib in CACHE.DISPATCHERS:
            CACHE.DISPATCHERS[lib].cache_clear()
        return func

    return decorator


def _register_library(lib: str):
    """Register an array library name.

    Args:
        lib: The name string to identify the array library.
    """
    CACHE.REGISTERED_LIBS.add(lib)
    if lib not in CACHE.REGISTERED_FUNCTIONS:
        CACHE.REGISTERED_FUNCTIONS[lib] = {}
    if lib not in CACHE.REGISTERED_MODULES:
        CACHE.REGISTERED_MODULES[lib] = []


@functools.lru_cache()
def _lib_from_module(obj: any) -> str:
    """Infer array library string from base module path of an object."""
    if isinstance(obj, ModuleType):
        modname = obj.__name__
    elif isinstance(obj, (type, FunctionType)):
        modname = obj.__module__
    else:
        modname = type(obj).__module__
    return modname.split(".", maxsplit=1)[0]
