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

"""
Global alias instances.
"""

from typing import Union

from scipy.sparse import spmatrix

from arraylias import numpy_alias, scipy_alias

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
