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
"""Dispatch class"""

import functools
from types import FunctionType
from typing import Optional, Callable, Union
from qiskit.utils import deprecate_arguments

from qiskit_dynamics.dispatch.dynamic_dispatcher import DynamicDispatcher
from qiskit_dynamics.dispatch.exceptions import DispatchError
from qiskit_dynamics.dispatch.default_dispatcher import DEFAULT_DISPATCHER


def default_dispatcher() -> DynamicDispatcher:
    """Return the default dispatcher.

    .. note ::

        There is only ever 1 instance of the default dispatcher for each library.
        All calls to this function will return the same :class:`DynamicDispatcher`
        instance.

    Returns:
        The default dynamic dispatcher for the module if ``lib`` is None.
        The default static dispatcher for the specified library other.
    """
    return DEFAULT_DISPATCHER


def array(
    obj: any,
    dtype: Optional[any] = None,
    order: Optional[str] = None,
    lib: Optional[str] = None,
    dispatcher: Optional[DynamicDispatcher] = None,
) -> any:
    """Construct an array of the specified array library.

    .. note::

        This functions like `numpy.array` but supports returning array
        types of other registered array libraries using the supplied
        :class:`DynamicDispatcher`.

    Args:
        obj: An array like input.
        dtype: Optional. The dtype of the returned array. This value
               must be supported by the specified array backend.
        order: Optional. The array order. This value must be supported
               by the specified array backend.
        lib: A library name for the default_dispatcher. If None the default
             library of the default dispatcher will be used if set.
        dispatcher: Optional, a specific dispatcher to use. If None the
                    :func:`default_dispatcher` will be used.

    Returns:
        array: an array object from the specified array library.

    Raises:
        DispatchError: If `lib` is None and the input `array` is not an
                       array type of a registered array library.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if order:
        kwargs["order"] = order
    if dispatcher is None:
        dispatcher = DEFAULT_DISPATCHER
    if lib is None:
        if dispatcher.default_lib is None:
            raise DispatchError(
                "No default array library has been set for the specified dispatcher. "
                "Either specify a library using the `lib` kwarg or set a default "
                "library using `dispatcher.default_library = lib`"
            )
        lib = dispatcher.default_library

    lib_func = dispatcher.static_dispatcher(lib).array
    return lib_func(obj, **kwargs)


@deprecate_arguments({"backend": "lib"})
def asarray(
    obj: any,
    dtype: Optional[any] = None,
    order: Optional[str] = None,
    lib: Optional[str] = None,
    dispatcher: Optional[DynamicDispatcher] = None,
    backend: Optional[str] = None,  # pylint: disable=unused-argument
) -> any:
    """Convert input array to an array of the specified library.

    .. note::

        This functions like `numpy.asarray` but optionally supports
        casting to other registered array types using the supplied
        :class:`DynamicDispatcher`.

    Args:
        obj: An array like input.
        dtype: Optional. The dtype of the returned array. This value
               must be supported by the specified array backend.
        order: Optional. The array order. This value must be supported
               by the specified array backend.
        lib: Optional, A library name for the default_dispatcher. If None
             the default library of the default dispatcher will be used if set.
        dispatcher: Optional, a specific dispatcher to use. If None the
                    :func:`default_dispatcher` will be used.
        backend: DEPREACTED, use lib kwarg instead.

    Returns:
        array: an array object from the specified array library.

    Raises:
        DispatchError: If `lib` is None, the default library is None, and the
                       input `array` is not an array type of a registered array
                       library.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if order:
        kwargs["order"] = order
    if dispatcher is None:
        dispatcher = DEFAULT_DISPATCHER
    if lib is None:
        lib = dispatcher.default_library
    if lib:
        lib_func = dispatcher.static_dispatcher(lib).asarray
    else:
        lib_func = dispatcher.asarray
    return lib_func(obj, **kwargs)


def infer_library(obj: any, dispatcher: Optional[DynamicDispatcher] = None) -> Union[str, None]:
    """Return the registered array library name of an array object.

    Args:
        obj: an array object.
        dispatcher: Optional, a specific dispatcher to use. If None the
                    :func:`default_dispatcher` will be used.

    Returns:
        The array library name if the array type is registered,
        or None otherwise.
    """
    if dispatcher is None:
        dispatcher = DEFAULT_DISPATCHER
    return dispatcher._infer_library(type(obj))


def requires_library(lib: str, dispatcher: Optional[DynamicDispatcher] = None) -> Callable:
    """Return a function and class decorator for checking a required library is available.

    If the the required library is not in the list of :func:`registered_libs`
    any decorated function or method will raise an exception when called, and
    any decorated class will raise an exeption when its ``__init__`` is called.

    Args:
        lib: the array library name required by class or function.
        dispatcher: Optional, a specific dispatcher to use. If None the
                    :func:`default_dispatcher` will be used.

    Returns:
        Callable: A decorator that may be used to specify that a function, class,
                  or class method requires a specific library to be installed.
    """
    if dispatcher is None:
        dispatcher = DEFAULT_DISPATCHER

    def decorator(obj):
        """Specify that the decorated object requires a specifc array library."""

        def check_lib(descriptor):
            if lib not in dispatcher.registered_libraries:
                raise DispatchError(
                    f"Array library '{lib}' required by {descriptor} "
                    "is not installed. Please install the optional "
                    f"library '{lib}'."
                )

        # Decorate a function or method
        if isinstance(obj, FunctionType):

            @functools.wraps(obj)
            def decorated_func(*args, **kwargs):
                check_lib(f"function {obj}")
                return obj(*args, **kwargs)

            return decorated_func

        # Decorate a class
        elif isinstance(obj, type):

            obj_init = obj.__init__

            @functools.wraps(obj_init)
            def decorated_init(self, *args, **kwargs):
                check_lib(f"class {obj}")
                obj_init(self, *args, **kwargs)

            obj.__init__ = decorated_init
            return obj

        else:
            raise Exception(f"Cannot decorate object {obj} that is not a class or function.")

    return decorator
