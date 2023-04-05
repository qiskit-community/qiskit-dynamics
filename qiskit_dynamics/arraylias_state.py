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
from scipy.sparse import csr_matrix
from .array import Array
try:
    from jax.experimental.sparse import BCOO
except ImportError:
    pass


DYNAMICS_ALIAS = numpy_alias()

# Set qiskit_dynamics.array.Array to be dispatched to numpy
DYNAMICS_ALIAS.register_type(Array, "numpy")

# register required custom versions of functions for csr type here
DYNAMICS_ALIAS.register_type(csr_matrix, lib="scipy_sparse")

# register required custom versions of functions for BCOO type here
DYNAMICS_ALIAS.register_type(BCOO, lib="jax_sparse")


@DYNAMICS_ALIAS.register_function(lib="scipy_sparse", path="asarray")
def _(csr: csr_matrix):
    return csr.toarray()


@DYNAMICS_ALIAS.register_function(lib="jax_sparse", path="asarray")
def _(bcoo: BCOO):
    return bcoo.todense()


DYNAMICS_NUMPY = DYNAMICS_ALIAS()

ArrayLike = Union[DYNAMICS_ALIAS.registered_types()]
