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
Registering to_numeric_matrix_type functions to alias
"""

from scipy.sparse import spmatrix
from arraylias.exceptions import LibraryError
from qiskit_dynamics.type_utils import isinstance_qutip_qobj


def register_to_numeric_matrix_type(alias):
    """register to_numeric_matrix_type functions to each array libraries"""

    @alias.register_default(path="to_numeric_matrix_type")
    def _(arr):
        if arr is None:
            return None
        if isinstance_qutip_qobj(arr):
            return alias().to_sparse(arr.data)
        return arr

    @alias.register_function(lib="numpy", path="to_numeric_matrix_type")
    def _(arr):
        return arr

    @alias.register_function(lib="operator", path="to_numeric_matrix_type")
    def _(arr):
        return alias().to_dense(arr)

    try:

        @alias.register_function(lib="jax", path="to_numeric_matrix_type")
        def _(arr):
            return arr

        @alias.register_function(lib="jax_sparse", path="to_numeric_matrix_type")
        def _(arr):
            return arr

    except LibraryError:
        pass

    @alias.register_function(lib="scipy_sparse", path="to_numeric_matrix_type")
    def _(arr):
        return arr

    @alias.register_function(lib="list", path="to_numeric_matrix_type")
    def _(arr):
        if isinstance(arr[0], spmatrix) or isinstance_qutip_qobj(arr[0]):
            return [alias().to_sparse(sub_arr) for sub_arr in arr]
        return alias().asarray([alias().to_dense(sub_arr) for sub_arr in arr])

    @alias.register_fallback(path="to_numeric_matrix_type")
    def _(arr):
        return arr
