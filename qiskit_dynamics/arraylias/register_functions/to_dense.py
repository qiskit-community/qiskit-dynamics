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
Registering to_dense functions to alias
"""

import numpy as np


def register_todense(alias):
    """register to_dense functions to each array libraries"""

    @alias.register_default(path="to_dense")
    def _(arr):
        if arr is None:
            return None
        return arr

    @alias.register_function(lib="numpy", path="to_dense")
    def _(arr):
        return arr

    try:
        # check if jax libraries are registered by import jax
        # pylint: disable=unused-import
        import jax

        @alias.register_function(lib="jax", path="to_dense")
        def _(arr):
            return arr

        @alias.register_function(lib="jax_sparse", path="to_dense")
        def _(arr):
            return arr.todense()

    except ImportError:
        pass

    @alias.register_function(lib="scipy_sparse", path="to_dense")
    def _(arr):
        return arr.toarray()

    @alias.register_function(lib="list", path="to_dense")
    def _(arr):
        return alias().asarray([alias().to_dense(sub_arr) for sub_arr in arr])

    @alias.register_fallback(path="to_dense")
    def _(arr):
        return np.asarray(arr)
