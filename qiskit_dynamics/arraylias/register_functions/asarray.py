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
Registering asarray functions to alias
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse


def register_asarray(alias):
    """register asarray functions to each array libraries"""

    @alias.register_default(path="asarray")
    def _(arr):
        return np.asarray(arr)

    @alias.register_function(lib="scipy_sparse", path="asarray")
    def _(arr):
        if issparse(arr):
            return arr
        return csr_matrix(arr)

    try:
        from jax.experimental.sparse import BCOO

        @alias.register_function(lib="jax_sparse", path="asarray")
        def _(arr):
            if type(arr).__name__ == "BCOO":
                return arr
            return BCOO.fromdense(arr)

    except ImportError:
        pass
