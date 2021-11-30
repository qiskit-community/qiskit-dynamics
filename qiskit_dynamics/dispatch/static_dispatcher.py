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
from typing import Optional, Union, Callable
from types import ModuleType

from qiskit_dynamics.dispatch.exceptions import DispatchError


class StaticDispatcher:
    """Static function Dispatcher class."""

    def __init__(self):
        """Initialize a function dispatcher"""
        self._functions = {}
        self._modules = []

    def register_function(
        self,
        func: Optional[Callable] = None,
        name: Optional[Union[Callable, str]] = None,
    ) -> Optional[Callable]:
        """Register an array function for dispatching.

        Args:
            func: The function to dispatch to for the specified array library.
                If None this will return a decorator to apply to a function.
            name: Optional, a name for dispatching to this function. If None
                the name of the input function will be used.

        Returns:
            If func is None returns a decorator for registering a function.
            Otherwise returns None.
        """
        decorator = self._register_function_decorator(name)
        if func is None:
            return decorator
        return decorator(func)

    def register_module(self, module: ModuleType):
        """Register a module for looking up dispatched array functions.

        Args:
            module: A module, namespace, or class to look for attributes
                    corresponding to the dispatched function name.
        """
        if module not in self._modules:
            self._modules.append(module)
            self.__call__.cache_clear()

    @functools.lru_cache()
    def __call__(self, name: Union[str, Callable]) -> Callable:
        # Convert input to string if callable
        if not isinstance(name, str):
            name = name.__name__

        # Check if function is already registered
        if name in self._functions:
            return self._functions[name]

        # If not registered try and import from registed modules
        for mod in self._modules:
            print("Try module", mod)
            try:
                func = getattr(mod, name)
                self._functions[name] = func
                return func
            except AttributeError:
                pass

        # Couldn't find a matching function
        raise DispatchError(
            f"Unable to dispatch function '{name}'  using registered functions and modules."
        )

    def __getattr__(self, name: str) -> Callable:
        setattr(self, name, self.__call__(name))
        return getattr(self, name)

    def _register_function_decorator(self, name: Optional[Union[str, Callable]] = None) -> Callable:
        """Return a decorator to register a function.

        Args:
            name: the name for dispatching to this function.

        Returns:
            A function decorator to register a function if dispatched_function
            is None.
        """
        if not isinstance(name, str):
            name = name.__name__

        def decorator(func):
            self._functions[name] = func
            self.__call__.cache_clear()
            return func

        return decorator
