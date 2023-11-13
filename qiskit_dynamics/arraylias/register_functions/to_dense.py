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
Register to_dense functions to alias.
"""

import numpy as np
from arraylias.exceptions import LibraryError


def register_todense(alias):
    """Register to_dense functions to each array library."""

    @alias.register_default(path="to_dense")
    def _(arr):
        if arr is None:
            return None
        return np.asarray(arr)

    @alias.register_function(lib="numpy", path="to_dense")
    def _(arr):
        return arr

    try:

        @alias.register_function(lib="jax", path="to_dense")
        def _(arr):
            return arr

        @alias.register_function(lib="jax_sparse", path="to_dense")
        def _(arr):
            return arr.todense()

    except LibraryError:
        pass

    @alias.register_function(lib="scipy_sparse", path="to_dense")
    def _(arr):
        return arr.toarray()

    @alias.register_fallback(path="to_dense")
    def _(arr):
        return np.asarray(arr)
