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
Registering to_sparse functions to alias.
"""

import numpy as np
from scipy.sparse import csr_matrix
from qiskit_dynamics.type_utils import isinstance_qutip_qobj


def register_tosparse(alias):
    """Register to_sparse functions to each array library."""

    @alias.register_default(path="to_sparse")
    def _(arr):
        if arr is None:
            return None
        if isinstance_qutip_qobj(arr):
            return arr.data
        return arr

    @alias.register_fallback(path="to_sparse")
    def _(arr):
        return csr_matrix(arr)

    @alias.register_function(lib="numpy", path="to_sparse")
    def _(arr):
        if arr.ndim < 3:
            return csr_matrix(arr)
        return np.array([csr_matrix(sub_arr) for sub_arr in arr])

    try:
        from jax.experimental.sparse import BCOO

        @alias.register_function(lib="jax", path="to_sparse")
        def _(arr):
            return BCOO.fromdense(arr)

        @alias.register_function(lib="jax_sparse", path="to_sparse")
        def _(arr):
            return arr

    except ImportError:
        pass

    @alias.register_function(lib="scipy_sparse", path="to_sparse")
    def _(arr):
        return arr
