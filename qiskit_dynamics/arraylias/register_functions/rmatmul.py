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
Registering rmatmul functions to alias
"""

import numpy as np


def register_rmatmul(alias):
    """Register rmatmul functions to each array library."""

    @alias.register_function(lib="numpy", path="rmatmul")
    def _(x, y):
        return np.matmul(y, x)

    @alias.register_function(lib="scipy_sparse", path="rmatmul")
    def _(x, y):
        return y * x

    try:
        from jax.experimental import sparse as jsparse
        import jax.numpy as jnp

        jsparse_matmul = jsparse.sparsify(jnp.matmul)

        @alias.register_function(lib="jax", path="rmatmul")
        def _(x, y):
            return jnp.matmul(y, x)

        @alias.register_function(lib="jax_sparse", path="rmatmul")
        def _(x, y):
            return jsparse_matmul(y, x)

    except ImportError:
        pass
