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
"""Dispatcher class"""

import functools
from typing import Optional, Union, Callable, Tuple, Set
from types import ModuleType, FunctionType


class DispatcherError(Exception):
    """Class for Dispatcher errors"""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = " ".join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)


class Dispatcher:
    """Function dispatcher class."""

    def __init__(self):
        """Initialize a function dispatcher"""
        # Set of registered library names for dispatching
        self._libs = set()

        # Map of library types to library names for dispatching
        self._types_lib = {}

        # Tuple of registered types from library types map
        self._types = tuple()

        # Map of library names to function map for dispatching names
        # to specific functions for that library
        self._functions = {}

        # Map of library names to list of modules for checking for functions
        self._modules = {}

        # Optional default library for static dispatching
        self._default_lib = None

        # Optional fallback library for dispatching on unregistered types
        self._fallback_lib = None

        # List of dispatched functions which have been dynamically added to the
        # dispatcher as attributes
        self._cached_attributes = []

    @functools.lru_cache()
    def __call__(self, name: Union[str, Callable], lib: Optional[str] = None) -> Callable:
        lib = lib or self._default_lib
        if lib:
            return self._static_dispatch(lib, name)
        return self._dynamic_dispatch(name)

    def __getattr__(self, name: str) -> Callable:
        self._cached_attributes.append(name)
        setattr(self, name, self.__call__(name))
        return getattr(self, name)

    def registered_libraries(self) -> Set[str]:
        """Return a set of registered array library names."""
        return self._libs

    def registered_types(self) -> Tuple[type]:
        """Return a set of registered array library names."""
        return self._types

    def fallback_library(self) -> Union[str, None]:
        """Return the fallback array library or None if no fallback is set."""
        return self._fallback_lib

    def set_fallback_library(self, lib: Union[str, None]):
        """Set the fallback array library."""
        if lib and not isinstance(lib, str):
            lib = _lib_from_module(lib)
        if lib not in self._libs:
            raise DispatcherError(
                f"Cannot set fallback library, '{lib}' is not a registered array library"
            )
        self._fallback_lib = lib
        self.cache_clear()

    def default_library(self) -> Union[str, None]:
        """Return the default array library or None if no default is set."""
        return self._default_lib

    def set_default_library(self, lib: str):
        """Return the default array library or None if no default is set."""
        if lib not in self._libs:
            raise DispatcherError(
                f"Cannot set default library, '{lib}' is not a registered array library"
            )
        self._default_lib = lib
        self.cache_clear()

    def register_function(
        self,
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
        decorator = self._register_function_decorator(name=name, lib=lib)
        if func is None:
            return decorator
        return decorator(func)

    def register_module(self, module: ModuleType, lib: Optional[str] = None):

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
        self._register_library(lib)
        self._modules[lib].append(module)
        self.cache_clear()

    def register_type(self, array_type: type, lib: Optional[str] = None):
        """Register an array type for dispatching array functions.

        Args:
            array_type: An array type to register for the array library.
            lib: Optional, a name string to identify the array library.
                If None this will be set as the base module name of the
                arrays module.
        """
        if lib is None:
            lib = _lib_from_module(array_type)
        self._register_library(lib)
        self._types_lib[array_type] = lib
        self._types = tuple(self._types_lib.keys())
        self.cache_clear()

    def cache_clear(self):
        """Clear cached dispatched calls."""
        # Clear LRU cached functions
        self.__call__.cache_clear()
        self._dynamic_dispatch.cache_clear()
        self._static_dispatch.cache_clear()
        self.infer_library.cache_clear()

        # Clear cached attributes
        for name in self._cached_attributes:
            delattr(self, name)
        self._cached_attributes = []

    def _register_library(self, lib: str):
        """Register an array library name.

        Args:
            lib: The name string to identify the array library.
        """
        if lib not in self._libs:
            self._libs.add(lib)
            self._functions[lib] = {}
            self._modules[lib] = []

    @functools.lru_cache()
    def _static_dispatch(self, lib: str, name: Union[str, Callable]) -> Callable:
        """Return dispatched function for the specified library"""
        # Convert input to string if callable
        if not isinstance(name, str):
            name = name.__name__

        # Check if function is already registered
        if (lib, name) in self._functions:
            return self._functions[lib][name]

        # If not registered try and import from registed modules
        for mod in self._modules[lib]:
            try:
                func = getattr(mod, name)
                self._functions[lib][name] = func
                return func
            except AttributeError:
                pass

        # Couldn't find a matching function
        raise DispatcherError(
            f"Unable to dispatch function '{name}'  using registered functions and modules."
        )

    @functools.lru_cache()
    def _dynamic_dispatch(self, name: Union[str, Callable]) -> Callable:
        """Return a dynamically dispatching function which infers library from first arg type"""
        # Define dispatching wrapper function
        def _function(array, *args, **kwargs):
            lib = self.infer_library(type(array))
            if lib is NotImplemented:
                if self._fallback_lib is None:
                    raise DispatcherError(
                        f"{type(array)} is not a registered type for any"
                        " registered array libraries and no fallback library is set."
                    )
                lib = self._fallback_lib
            lib_func = self._static_dispatch(lib, name)
            return lib_func(array, *args, **kwargs)

        return _function

    @functools.lru_cache()
    def infer_library(self, array_type: type) -> Union[str, None]:
        """Return the registered library name of a array object.

        Args:
            array_type: an array type.

        Returns:
            The array library name if the array type is registered,
            or None otherwise.
        """
        if array_type in self._types_lib:
            return self._types_lib[array_type]

        for key, lib in self._types_lib.items():
            if issubclass(array_type, key):
                self._types_lib[array_type] = lib
                return lib
        return NotImplemented

    def _register_function_decorator(
        self, name: Optional[Union[str, Callable]] = None, lib: Optional[str] = None
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
            self._register_library(func_lib)
            self._functions[func_lib][name] = func
            self.cache_clear()
            return func

        return decorator


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
