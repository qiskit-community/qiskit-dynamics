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
Test rmatmul functions
"""
import unittest
import numpy as np
from scipy.sparse import csr_matrix

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from ...common import QiskitDynamicsTestCase


class TestRmatmulFunction(QiskitDynamicsTestCase):
    """Test cases for rmatmul functions registered in dynamics_numpy_alias."""

    def test_numpy(self):
        """Test rmatmul for numpy."""
        x = np.array([[1, 1], [1, 1]])
        y = np.array([[1, 2], [3, 4]])
        self.assertTrue(isinstance(unp.rmatmul(x, y), np.ndarray))
        self.assertAllClose(unp.rmatmul(x, y), [[3, 3], [7, 7]])

    def test_scipy_sparse(self):
        """Test rmatmul for scipy_sparse."""
        x = csr_matrix([[1, 1], [1, 1]])
        y = csr_matrix([[1, 2], [3, 4]])
        self.assertTrue(isinstance(unp.rmatmul(x, y), csr_matrix))
        self.assertAllClose(csr_matrix.toarray(unp.rmatmul(x, y)), [[3, 3], [7, 7]])

    def test_jax(self):
        """Test rmatmul for jax."""
        try:
            import jax.numpy as jnp

            x = jnp.array([[1, 1], [1, 1]])
            y = jnp.array([[1, 2], [3, 4]])
            self.assertTrue(isinstance(unp.rmatmul(x, y), jnp.ndarray))
            self.assertAllClose(unp.rmatmul(x, y), [[3, 3], [7, 7]])
        except ImportError as err:
            raise unittest.SkipTest("Skipping jax tests.") from err

    def test_jax_sparse(self):
        """Test rmatmul for jax_sparse."""
        try:
            from jax.experimental.sparse import BCOO

            x = BCOO.fromdense([[1, 1], [1, 1]])
            y = BCOO.fromdense([[1, 2], [3, 4]])
            self.assertTrue(isinstance(unp.rmatmul(x, y), BCOO))
            self.assertAllClose(BCOO.todense(unp.rmatmul(x, y)), [[3, 3], [7, 7]])
        except ImportError as err:
            raise unittest.SkipTest("Skipping jax tests.") from err
