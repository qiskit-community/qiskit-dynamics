# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Global alias instances.
"""
import functools

from types import FunctionType

from typing import Union, Callable

from scipy.sparse import spmatrix

from arraylias import numpy_alias, scipy_alias

from arraylias.exceptions import LibraryError

from qiskit import QiskitError

from qiskit_dynamics.array import Array

from .register_functions import (
    register_asarray,
    register_matmul,
    register_multiply,
    register_rmatmul,
    register_linear_combo,
    register_transpose,
    register_conjugate,
)

# global NumPy and SciPy aliases
DYNAMICS_NUMPY_ALIAS = numpy_alias()
DYNAMICS_SCIPY_ALIAS = scipy_alias()

DYNAMICS_NUMPY_ALIAS.register_type(Array, "numpy")
DYNAMICS_SCIPY_ALIAS.register_type(Array, "numpy")


DYNAMICS_NUMPY = DYNAMICS_NUMPY_ALIAS()
DYNAMICS_SCIPY = DYNAMICS_SCIPY_ALIAS()

# register required custom versions of functions for sparse type here
DYNAMICS_NUMPY_ALIAS.register_type(spmatrix, lib="scipy_sparse")

try:
    from jax.experimental.sparse import BCOO

    # register required custom versions of functions for BCOO type here
    DYNAMICS_NUMPY_ALIAS.register_type(BCOO, lib="jax_sparse")
except ImportError:
    pass

# register custom functions for numpy_alias
register_asarray(alias=DYNAMICS_NUMPY_ALIAS)
register_matmul(alias=DYNAMICS_NUMPY_ALIAS)
register_multiply(alias=DYNAMICS_NUMPY_ALIAS)
register_rmatmul(alias=DYNAMICS_NUMPY_ALIAS)
register_linear_combo(alias=DYNAMICS_NUMPY_ALIAS)
register_conjugate(alias=DYNAMICS_NUMPY_ALIAS)
register_transpose(alias=DYNAMICS_NUMPY_ALIAS)


ArrayLike = Union[Union[DYNAMICS_NUMPY_ALIAS.registered_types()], list]


def _isArrayLike(x: any) -> bool:
    """Return true if x is an ArrayLike object. Equivalent to isinstance(x, ArrayLike),
    however this does not work in Python 3.9.
    """
    return isinstance(x, (DYNAMICS_NUMPY_ALIAS.registered_types(), list))


def _preferred_lib(*args, **kwargs):
    """Given a list of args and kwargs with potentially mixed array types, determine the appropriate
    library to dispatch to.

    For each argument, DYNAMICS_NUMPY_ALIAS.infer_libs is called to infer the library. If all are
    "numpy", then it returns "numpy", and if any are "jax", it returns "jax".

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    Returns:
        str
    Raises:
        QiskitError: if none of the rules apply.
    """
    args = list(args) + list(kwargs.values())
    if len(args) == 1:
        libs = DYNAMICS_NUMPY_ALIAS.infer_libs(args[0])
        return libs[0] if len(libs) > 0 else "numpy"

    lib0 = _preferred_lib(args[0])
    lib1 = _preferred_lib(args[1:])

    if lib0 == "numpy" and lib1 == "numpy":
        return "numpy"
    elif lib0 == "jax_sparse" or lib1 == "jax_sparse":
        return "jax_sparse"
    elif lib0 == "jax" or lib1 == "jax":
        return "jax"
    elif lib0 == "scipy_sparse" or lib1 == "scipy_sparse":
        return "scipy_sparse"

    raise QiskitError("_preferred_lib could not resolve preferred library.")


def _numpy_multi_dispatch(*args, path, **kwargs):
    """Multiple dispatching for NumPy.

    Given *args and **kwargs, dispatch the function specified by path, to the array library
    specified by _preferred_lib.

    Args:
        *args: Positional arguments to pass to function specified by path.
        path: Path in numpy module structure.
        **kwargs: Keyword arguments to pass to function specified by path.
    Returns:
        Result of evaluating the function at path on the arguments using the preferred library.
    """
    lib = _preferred_lib(*args, **kwargs)
    return DYNAMICS_NUMPY_ALIAS(like=lib, path=path)(*args, **kwargs)


def _to_dense(x):
    """Convert an array to dense."""
    libs = DYNAMICS_NUMPY_ALIAS.infer_libs(x)
    if "scipy_sparse" in libs or "jax_sparse" in libs:
        return DYNAMICS_NUMPY.array(x.todense())
    return x


def _to_dense_list(x):
    """Convert a list of arrays to their corresponding dense version. Assumes input is either a list
    of scipy sparse, a BCOO array, or numpy/jax array.
    """
    libs = DYNAMICS_NUMPY_ALIAS.infer_libs(x)
    if "scipy_sparse" in libs:
        return DYNAMICS_NUMPY.array([op.todense() for op in x])
    elif "jax_sparse" in libs:
        return x.todense()
    return x


def requires_array_library(lib: str) -> Callable:
    """Return a function and class decorator for checking a library is available.

    If the the required library is not in the list of the registered library
    for global alias instances, any decorated function or method will raise
    an exception when called, and any decorated class will raise an exeption
    when its ``__init__`` is called.

    Args:
        lib: the library name required by class or function.

    Returns:
        Callable: A decorator that may be used to specify that a function, class,
                  or class method requires a specific library to be installed.
    """

    def decorator(obj):
        """Specify that the decorated object requires a specifc Array library."""

        def check_library(descriptor):
            if lib not in DYNAMICS_NUMPY_ALIAS.registered_libs():
                raise LibraryError(
                    f"Array library '{lib}' required by {descriptor} "
                    "is not installed. Please install the optional "
                    f"library '{lib}'."
                )

        # Decorate a function or method
        if isinstance(obj, FunctionType):

            @functools.wraps(obj)
            def decorated_func(*args, **kwargs):
                check_library(f"function {obj}")
                return obj(*args, **kwargs)

            return decorated_func

        # Decorate a class
        elif isinstance(obj, type):
            obj_init = obj.__init__

            @functools.wraps(obj_init)
            def decorated_init(self, *args, **kwargs):
                check_library(f"class {obj}")
                obj_init(self, *args, **kwargs)

            obj.__init__ = decorated_init
            return obj

        else:
            raise ValueError(f"Cannot decorate object {obj} that is not a class or function.")

    return decorator
