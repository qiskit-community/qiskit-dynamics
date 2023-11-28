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
Registering linear_combo functions to alias. This computes a linear combination of matrices (given
by a 3d array).
"""

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    pass

def register_linear_combo(alias):
    """Register linear functions for each array library."""

    @alias.register_default(path="linear_combo")
    def _(coeffs, mats):
        return np.tensordot(coeffs, mats, axes=1)

    @alias.register_function(lib="numpy", path="linear_combo")
    def _(coeffs, mats):
        return np.tensordot(coeffs, mats, axes=1)

    try:
        import jax.numpy as jnp
        
        @alias.register_function(lib="jax", path="linear_combo")
        def _(coeffs, mats):
            return jnp.tensordot(coeffs, mats, axes=1)
        
        from jax.experimental.sparse import sparsify
        jsparse_sum = sparsify(jnp.sum)

        @alias.register_function(lib="jax_sparse", path="linear_combo")
        def _(coeffs, mats):
            return jsparse_sum(jnp.broadcast_to(coeffs[:, None, None], mats.shape) * mats, axis=0)

    except ImportError:
        pass
