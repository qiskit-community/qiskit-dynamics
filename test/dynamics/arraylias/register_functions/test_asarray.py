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
Test asarray functions
"""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from qiskit.quantum_info.operators import Operator


from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS
from qiskit_dynamics import DYNAMICS_NUMPY as unp


class TestAsarrayFunction(unittest.TestCase):
    """Test cases for asarray functions registered in dynamics_numpy_alias."""

    def test_register_default(self):
        """Test register_default."""
        arr = Operator.from_label("X")
        self.assertTrue(isinstance(unp.asarray(arr), np.ndarray))

    def test_scipy_sparse(self):
        """Test asarray for scipy_sparse."""
        arr = np.array([[1, 0], [0, 1]])
        sparse_arr = csr_matrix([[1, 0], [0, 1]])
        self.assertTrue(isinstance(unp.asarray(sparse_arr), csr_matrix))
        self.assertTrue(isinstance(DYNAMICS_NUMPY_ALIAS(like=sparse_arr).asarray(arr), csr_matrix))

    def test_jax_sparse(self):
        """Test asarray for jax_sparse."""
        try:
            from jax.experimental.sparse import BCOO

            arr = np.array([[1, 0], [0, 1]])
            sparse_arr = BCOO.fromdense([[1, 0], [0, 1]])
            self.assertTrue(isinstance(unp.asarray(sparse_arr), BCOO))
            self.assertTrue(isinstance(DYNAMICS_NUMPY_ALIAS(like=sparse_arr).asarray(arr), BCOO))
        except ImportError as err:
            raise unittest.SkipTest("Skipping jax tests.") from err
