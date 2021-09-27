# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Functions for working with Array dispatch."""

import functools
from types import FunctionType
from typing import Callable

from .array import Array


def wrap(
    func: Callable, wrap_return: bool = True, wrap_args: bool = True, decorator: bool = False
) -> Callable:
    """Wrap an array backend function to work with Arrays.

    Args:
        func: a function to wrap.
        wrap_return: If ``True`` convert results that are registered array
                     backend types into Array objects (Default: True).
        wrap_args: If ``True`` also wrap function type args and kwargs of the
                   wrapped function.
        decorator: If ``True`` the wrapped decorator function ``func`` will
                   also wrap the decorated functions (Default: False).

    Returns:
        Callable: The wrapped function.
    """
    if decorator:
        return _wrap_decorator(func, wrap_return=wrap_return, wrap_args=wrap_args)
    else:
        return _wrap_function(func, wrap_return=wrap_return, wrap_args=wrap_args)


def _wrap_array_function(func: Callable) -> Callable:
    """Wrap a function to handle Array-like inputs and returns"""

    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):

        # Unwrap inputs
        args = tuple(
            x.__qiskit_array__().data if hasattr(x, "__qiskit_array__") else x for x in args
        )
        kwargs = dict(
            (key, val.__qiskit_array__().data) if hasattr(val, "__qiskit_array__") else (key, val)
            for key, val in kwargs.items()
        )

        # Evaluate function with unwrapped inputs
        result = func(*args, **kwargs)

        # Unwrap result
        if isinstance(result, tuple):
            result = tuple(
                x.__qiskit_array__().data if hasattr(x, "__qiskit_array__") else x for x in result
            )
        elif hasattr(result, "__qiskit_array__"):
            result = result.__qiskit_array__().data
        return result

    return wrapped_function


def _wrap_args(args):
    """Return wrapped args"""
    return tuple(_wrap_array_function(x) if isinstance(x, FunctionType) else x for x in args)


def _wrap_kwargs(kwargs):
    """Return wrapped kwargs"""
    return dict(
        (key, _wrap_array_function(val)) if isinstance(val, FunctionType) else (key, val)
        for key, val in kwargs.items()
    )


def _wrap_function(func: Callable, wrap_return: bool = True, wrap_args: bool = True) -> Callable:
    """Wrap an array backend function to work with Arrays.

    Args:
        func: a function to wrap.
        wrap_return: If ``True`` convert results that are registered array
                     backend types into Array objects (Default: True).
        wrap_args: If ``True`` also wrap function type args and kwargs of the
                   wrapped function.

    Returns:
        Callable: The wrapped function.
    """
    # pylint: disable = function-redefined
    if wrap_return and wrap_args:

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            args = _wrap_args(args)
            kwargs = _wrap_kwargs(kwargs)
            result = _wrap_array_function(func)(*args, **kwargs)
            return Array._wrap(result)

        return wrapped_func

    elif wrap_return and not wrap_args:

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            result = _wrap_array_function(func)(*args, **kwargs)
            return Array._wrap(result)

        return wrapped_func

    elif not wrap_return and wrap_args:

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            args = _wrap_args(args)
            kwargs = _wrap_kwargs(kwargs)
            return _wrap_array_function(func)(*args, **kwargs)

        return wrapped_func

    else:

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            return _wrap_array_function(func)(*args, **kwargs)

        return wrapped_func


def _wrap_decorator(func: Callable, wrap_return: bool = True, wrap_args: bool = True) -> Callable:
    """Wrap a function decorator to work with Arrays.

    Args:
        func: a function to wrap.
        wrap_return: If ``True`` convert results that are registered array
                     backend types into Array objects (Default: True).
        wrap_args: If ``True`` also wrap function type args and kwargs of the
                   wrapped function.

    Returns:
        Callable: The wrapped function.
    """
    # pylint: disable = function-redefined
    if wrap_return and wrap_args:

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            args = _wrap_args(args)
            kwargs = _wrap_kwargs(kwargs)
            decorated = _wrap_array_function(func)(*args, **kwargs)

            @functools.wraps(args[0])
            def wrapped_decorated(*f_args, **f_kwargs):
                f_args = _wrap_args(f_args)
                f_kwargs = _wrap_kwargs(f_kwargs)
                result = _wrap_function(decorated)(*f_args, **f_kwargs)
                return Array._wrap(result)

            return wrapped_decorated

        return wrapped_func

    if wrap_return and not wrap_args:

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            decorated = _wrap_array_function(func)(*args, **kwargs)

            @functools.wraps(args[0])
            def wrapped_decorated(*f_args, **f_kwargs):
                result = _wrap_function(decorated)(*f_args, **f_kwargs)
                return Array._wrap(result)

            return wrapped_decorated

        return wrapped_func

    if not wrap_return and wrap_args:

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            args = _wrap_args(args)
            kwargs = _wrap_kwargs(kwargs)
            decorated = _wrap_array_function(func)(*args, **kwargs)

            @functools.wraps(args[0])
            def wrapped_decorated(*f_args, **f_kwargs):
                f_args = _wrap_args(f_args)
                f_kwargs = _wrap_kwargs(f_kwargs)
                return _wrap_function(decorated)(*f_args, **f_kwargs)

            return wrapped_decorated

        return wrapped_func

    else:

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            decorated = _wrap_array_function(func)(*args, **kwargs)

            @functools.wraps(args[0])
            def wrapped_decorated(*f_args, **f_kwargs):
                return _wrap_function(decorated)(*f_args, **f_kwargs)

            return wrapped_decorated

        return wrapped_func
