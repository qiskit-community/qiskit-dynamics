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
Test matmul functions
"""
import unittest
import numpy as np
from scipy.sparse import csr_matrix

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from ...common import QiskitDynamicsTestCase


class TestMatmulFunction(QiskitDynamicsTestCase):
    """Test cases for matmul functions registered in dynamics_numpy_alias."""

    def test_scipy_sparse(self):
        """Test matmul for scipy_sparse."""
        x = csr_matrix([[1, 0], [0, 1]])
        y = csr_matrix([[2, 2], [2, 2]])
        self.assertTrue(isinstance(unp.matmul(x, y), csr_matrix))
        self.assertAllClose(csr_matrix.toarray(unp.matmul(x, y)), [[2, 2], [2, 2]])

    def test_jax_sparse(self):
        """Test matmul for jax_sparse."""
        try:
            from jax.experimental.sparse import BCOO

            x = BCOO.fromdense([[1, 0], [0, 1]])
            y = BCOO.fromdense([[2, 2], [2, 2]])
            self.assertTrue(isinstance(unp.matmul(x, y), BCOO))
            self.assertAllClose(BCOO.todense(unp.matmul(x, y)), [[2, 2], [2, 2]])
        except ImportError as err:
            raise unittest.SkipTest("Skipping jax tests.") from err
