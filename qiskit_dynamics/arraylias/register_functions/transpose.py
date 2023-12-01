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
Registering transpose.
"""


def register_transpose(alias):
    """Register linear functions for each array library."""

    try:
        import jax.numpy as jnp
        from jax.experimental.sparse import bcoo_transpose

        @alias.register_function(lib="jax_sparse", path="transpose")
        def _(a, axes=None):
            return bcoo_transpose(a, permutation=axes)

    except ImportError:
        pass
