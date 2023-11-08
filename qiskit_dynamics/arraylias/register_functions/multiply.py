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
Registering multiply functions to alias
"""


def register_multiply(alias):
    """Register multiply functions to each array library."""

    @alias.register_function(lib="scipy_sparse", path="multiply")
    def _(x, y):
        return x.multiply(y)

    try:
        from jax.experimental import sparse as jsparse
        import jax.numpy as jnp

        jsparse_multiply = jsparse.sparsify(jnp.multiply)

        @alias.register_function(lib="jax_sparse", path="multiply")
        def _(x, y):
            return jsparse_multiply(x, y)

    except ImportError:
        pass

    @alias.register_fallback(path="multiply")
    def _(x, y):
        return x * y
