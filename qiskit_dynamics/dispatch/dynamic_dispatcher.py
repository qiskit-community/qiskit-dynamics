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

from qiskit_dynamics.dispatch.exceptions import DispatchError
from qiskit_dynamics.dispatch.static_dispatcher import StaticDispatcher


class DynamicDispatcher:
    """Dynamic function dispatcher class."""

    def __init__(self):
        """Initialize a dynamic function dispatcher"""
        self._default_lib = None
        self._libs = set()
        self._types_lib = {}
        self._types = tuple()
        self._static_dispatchers = {}

    @functools.lru_cache()
    def static_dispatcher(self, lib: Optional[str] = None) -> Callable:
        """Return the dispatcher for the specified library.

        Args:
            lib: An array library name. If None and the default library will be
                used if one has been set.

        Returns:
            The function dispatcher for the specified library.

        Raises:
            DispatchError: if the input library is not registered.
        """
        if lib is None:
            if self._default_lib is None:
                raise DispatchError("No default library has be set.")
            lib = self._default_lib
        return self._static_dispatchers[lib]

    @functools.lru_cache()
    def __call__(self, name: Union[str, Callable]) -> Callable:
        # Define dispatching wrapper function
        def _function(array, *args, **kwargs):
            lib = self._infer_library(type(array))
            if lib is NotImplemented:
                if self._default_lib is None:
                    raise DispatchError(
                        f"{type(array)} is not a registered type for any"
                        " registered array libraries."
                    )
                lib = self._default_lib
            lib_func = self._static_dispatchers[lib](name)
            return lib_func(array, *args, **kwargs)

        return _function

    def __getattr__(self, name: str) -> Callable:
        setattr(self, name, self.__call__(name))
        return getattr(self, name)

    @property
    def registered_libraries(self) -> Set[str]:
        """Return a set of registered array library names."""
        return self._libs

    @property
    def registered_types(self) -> Tuple[type]:
        """Return a set of registered array library names."""
        return self._types

    @property
    def default_library(self) -> Union[str, None]:
        """Return the default array library or None if no default is set."""
        return self._default_lib

    @default_library.setter
    def default_library(self, lib: str):
        """Return the default array library or None if no default is set."""
        if lib not in self._static_dispatchers:
            raise DispatchError(
                f"Cannot set default library, '{lib}' is not a registered array library"
            )
        self._default_lib = lib

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
        self._static_dispatchers[lib].register_module(module)
        self.__call__.cache_clear()

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
        self.__call__.cache_clear()

    def _register_library(self, lib: str):
        """Register an array library name.

        Args:
            lib: The name string to identify the array library.
        """
        if lib not in self._static_dispatchers:
            self._static_dispatchers[lib] = StaticDispatcher()
            self._libs.add(lib)

    def _clear_lru_cache(self):
        """Clear the LRU cache of cached methods."""
        self.__call__.cache_clear()
        self.static_dispatcher.cache_clear()
        self._infer_library.cache_clear()

    @functools.lru_cache()
    def _infer_library(self, array_type: type) -> Union[str, None]:
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
            self._static_dispatchers[lib].register_function(func, name)
            self.__call__.cache_clear()
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
