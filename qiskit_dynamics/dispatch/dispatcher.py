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


class DispatcherCache:
    """Cache class for dispatcher"""

    def __init__(self, dispatcher, lib=None):
        self._dispatcher = dispatcher
        self._lib = lib
        self._cached = []

    def __getattr__(self, name: str) -> Callable:
        func = self._dispatcher(name, self._lib)
        self._cached.append(name)
        setattr(self, name, func)
        return getattr(self, name)

    def cache_clear(self):
        """Clear cached function atttributes."""
        for name in self._cached:
            delattr(self, name)
        self._cached = []


class Dispatcher:
    """Function dispatcher class."""

    __slots__ = [
        "_libs",
        "_types_lib",
        "_types",
        "_functions",
        "_modules",
        "_default_lib",
        "_fallback_lib",
        "_fn_cache",
        "_lib_cache",
    ]

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

        # Cache for attribute based access of dispatched functions to
        # dynamic library
        self._fn_cache = DispatcherCache(self)
        self._lib_cache = {}

    @functools.lru_cache()
    def __call__(
        self, name: Union[str, Callable], lib: Optional[Union[str, type]] = None
    ) -> Callable:
        lib = self._infer_library(lib)
        if lib:
            return self._static_dispatch(lib, name)
        return self._dynamic_dispatch(name)

    @property
    def fn(self) -> DispatcherCache:  # pylint: disable = invalid-name
        """Return function cache for attribute based access to dispatcher functions"""
        return self._fn_cache

    @functools.lru_cache()
    def lib(self, lib: Optional[Union[str, type]] = None) -> DispatcherCache:
        """Return function cache for specific library.

        Args:
            lib: A dispatcher library. If None the default library
                 will be be return if one is set.

        Returns:
            The DispatcherCache for the specified library.

        Raises:
            DispatcherError: If ``lib=None`` and no :meth:`default_library`
                             has been set.
        """
        lib = self._infer_library(lib)
        if lib:
            return self._lib_cache[lib]
        raise DispatcherError(
            f"Unrecognized dispatcher library '{lib}'. Registered libraries are {self._libs}"
        )

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
        self.__call__.cache_clear()
        self._dynamic_dispatch.cache_clear()

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
        self.__call__.cache_clear()

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
        self.lib.cache_clear()
        self._dynamic_dispatch.cache_clear()
        self._static_dispatch.cache_clear()
        self._infer_library.cache_clear()

        # Clear DispatcherCache
        self._fn_cache.cache_clear()
        for cache in self._lib_cache.values():
            cache.cache_clear()

    def _register_library(self, lib: str):
        """Register an array library name.

        Args:
            lib: The name string to identify the array library.
        """
        if lib not in self._libs:
            self._libs.add(lib)
            self._functions[lib] = {}
            self._modules[lib] = []
            self._lib_cache[lib] = DispatcherCache(self, lib)

    @functools.lru_cache()
    def _static_dispatch(self, lib: str, name: Union[str, Callable]) -> Callable:
        """Return dispatched function for the specified library"""
        # Convert input to string if callable
        if not isinstance(name, str):
            name = name.__name__

        # Check if function is already registered
        if name in self._functions[lib]:
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
            lib = self._infer_library(type(array))
            if lib is None:
                if self._fallback_lib is None:
                    raise DispatcherError(
                        f"{type(array)} is not a registered type for any"
                        " registered array libraries and no fallback library is set."
                    )
                lib = self._fallback_lib
            lib_func = self._static_dispatch(lib, name)
            return lib_func(array, *args, **kwargs)

        return _function

    def infer_library(self, obj: any) -> Union[str, None]:
        """Return the registered library name of a array object.

        Args:
            obj: an array type or instance.

        Returns:
            The array library name if the array type is registered,
            or NotImplemented otherwise.
        """
        if obj and not isinstance(obj, (type, str)):
            obj = type(obj)
        return self._infer_library(obj)

    @functools.lru_cache()
    def _infer_library(self, obj: Optional[Union[type, str]] = None) -> Union[str, None]:
        """Return the registered library name of a array object.

        Args:
            obj: an array type or library name.

        Returns:
            The array library name if the array type is registered,
            or NotImplemented otherwise.
        """
        if obj is None:
            if self._default_lib:
                return self._default_lib
            return None

        if obj in self._libs:
            return obj

        if obj in self._types_lib:
            return self._types_lib[obj]

        # Look via subclass
        for key, lib in self._types_lib.items():
            if issubclass(obj, key):
                self._types_lib[obj] = lib
                return lib

        return None

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
