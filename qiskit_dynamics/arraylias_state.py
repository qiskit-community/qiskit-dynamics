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

"""Configure custom instance of numpy alias for Dynamics."""

from typing import Union
from arraylias import numpy_alias
from collections.abc import Iterable
import numpy as np
from scipy.sparse import spmatrix, csr_matrix
from .array import Array
from qiskit.quantum_info.operators import Operator


try:
    from jax.experimental.sparse import BCOO
    import jax.numpy as jnp
except ImportError:
    pass


DYNAMICS_ALIAS = numpy_alias()

# Set qiskit_dynamics.array.Array to be dispatched to numpy
DYNAMICS_ALIAS.register_type(Array, "numpy")

# register required custom versions of functions for sparse type here
DYNAMICS_ALIAS.register_type(spmatrix, lib="scipy_sparse")

# register required custom versions of functions for BCOO type here
DYNAMICS_ALIAS.register_type(BCOO, lib="jax_sparse")

# register required custom versions of functions for Operator type here
DYNAMICS_ALIAS.register_type(Operator, lib="operator")

# register required custom versions of functions for Iterable type here
# need to discuss registering Iterable type because the coverage of Iterable is too broad.
DYNAMICS_ALIAS.register_type(Iterable,lib="iterable")

# asarray
@DYNAMICS_ALIAS.register_function(lib="iterable", path="asarray")
def _(arr):
    return DYNAMICS_ALIAS(like=arr[0]).asarray(arr)
@DYNAMICS_ALIAS.register_fallback(lib="scipy_sparse", path="asarray")
def _(arr):
    return np.asarray(arr)
@DYNAMICS_ALIAS.register_fallback(lib="jax_sparse", path="asarray")
def _(arr):
    return jnp.asarray(arr)


# to_dense
@DYNAMICS_ALIAS.register_function(lib="numpy", path="to_dense")
def _(op):
    return op
@DYNAMICS_ALIAS.register_function(lib="jax", path="to_dense")
def _(op):
    return op
@DYNAMICS_ALIAS.register_function(lib="scipy_sparse", path="to_dense")
def _(op):
    return op.toarray()
@DYNAMICS_ALIAS.register_function(lib="jax_sparse", path="to_dense")
def _(op):
    return op.todense()
@DYNAMICS_ALIAS.register_fallback(path="to_dense")
def _(op):
    return np.asarray(op)
@DYNAMICS_ALIAS.register_function(lib="iterable", path="to_dense")
def _(op):
    return DYNAMICS_ALIAS().asarray([DYNAMICS_ALIAS().to_dense(sub_op) for sub_op in op])


# to_sparse
@DYNAMICS_ALIAS.register_function(lib="numpy", path="to_sparse")
def _(op):
    return csr_matrix(op)
@DYNAMICS_ALIAS.register_function(lib="jax", path="to_sparse")
def _(op):
    return BCOO.fromdense(op)
@DYNAMICS_ALIAS.register_function(lib="scipy_sparse", path="to_sparse")
def _(op):
    return op
@DYNAMICS_ALIAS.register_function(lib="jax_sparse", path="to_sparse")
def _(op):
    return op
@DYNAMICS_ALIAS.register_fallback(path="to_sparse")
def _(op):
    return csr_matrix(op)
@DYNAMICS_ALIAS.register_function(lib="iterable", path="to_sparse")
def _(op):
    return DYNAMICS_ALIAS().asarray([DYNAMICS_ALIAS().to_sparse(sub_op) for sub_op in op])


# to_numeric_matrix_type
@DYNAMICS_ALIAS.register_function(lib="iterable", path="to_numeric_matrix_type")
def _(op):
    return DYNAMICS_ALIAS().asarray([DYNAMICS_ALIAS().to_sparse(sub_op) for sub_op in op])
@DYNAMICS_ALIAS.register_fallback(path="to_numeric_matrix_type")
def _(op):
    return DYNAMICS_ALIAS().asarray(op)


# cond
@DYNAMICS_ALIAS.register_function(lib="numpy", path="cond")
def _(pred, true_fun, false_fun, *operands):
  if pred:
    return true_fun(*operands)
  else:
    return false_fun(*operands)


DYNAMICS_NUMPY = DYNAMICS_ALIAS()

ArrayLike = Union[DYNAMICS_ALIAS.registered_types()]
