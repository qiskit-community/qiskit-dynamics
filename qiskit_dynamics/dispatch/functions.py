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
from typing import Optional, Callable
from qiskit.utils import deprecate_arguments
from .exceptions import DispatchError

from .register import is_registered_library, CACHE
from .dispatcher import function, dispatcher

__all__ = ["requires_library", "asarray", "array"]


def requires_library(lib: str) -> Callable:
    """Return a function and class decorator for checking a required library is available.

    If the the required library is not in the list of :func:`registered_libs`
    any decorated function or method will raise an exception when called, and
    any decorated class will raise an exeption when its ``__init__`` is called.

    Args:
        lib: the array library name required by class or function.

    Returns:
        Callable: A decorator that may be used to specify that a function, class,
                  or class method requires a specific library to be installed.
    """

    def decorator(obj):
        """Specify that the decorated object requires a specifc array library."""

        def check_lib(descriptor):
            if not is_registered_library(lib):
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


@deprecate_arguments({"backend": "lib"})
def asarray(
    obj: any,
    dtype: Optional[any] = None,
    order: Optional[str] = None,
    lib: Optional[str] = None,
    backend: Optional[str] = None,  # pylint: disable=unused-argument
) -> any:
    """Convert input array to an array of the specified library.

    This functions like `numpy.asarray` but optionally supports
    casting to other registered array libraries.

    Args:
        obj: An array like input.
        dtype: Optional. The dtype of the returned array. This value
               must be supported by the specified array backend.
        order: Optional. The array order. This value must be supported
               by the specified array backend.
        lib: A registered array library name. If None the default
             array library will be used.
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
    if lib is None:
        lib = CACHE.DEFAULT_LIB
    if lib:
        lib_func = dispatcher(lib)("asarray")
    else:
        lib_func = function("asarray")
    return lib_func(obj, **kwargs)


def array(
    obj: any,
    dtype: Optional[any] = None,
    order: Optional[str] = None,
    lib: Optional[str] = None,
) -> any:
    """Construct an array of the specified array library.

    This functions like `numpy.array` but supports returning array
    types of other registered array libraries.

    Args:
        obj: An array like input.
        dtype: Optional. The dtype of the returned array. This value
               must be supported by the specified array backend.
        order: Optional. The array order. This value must be supported
               by the specified array backend.
        lib: A registered array library name. If None the default
             array library will be used.

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
    if lib is None:
        if CACHE.DEFAULT_LIB is None:
            raise DispatchError(
                "No default array library has been set. Either specify a library using "
                "the `lib` kwarg or set a default library using `set_default_library`"
            )
        lib = CACHE.DEFAULT_LIB

    lib_func = dispatcher(lib)("array")
    return lib_func(obj, **kwargs)
