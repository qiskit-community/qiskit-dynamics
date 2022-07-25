# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Direct tests of helper functions in lanczos.py
"""

import numpy as np
from scipy.linalg import expm
from qiskit_dynamics.solvers.lanczos import (
    lanczos_basis,
    lanczos_eigh,
    lanczos_expm,
    jax_lanczos_basis,
    jax_lanczos_eigh,
    jax_lanczos_expm,
)
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestLanczos(QiskitDynamicsTestCase):
    """Tests for lanczos.py"""

    def lanczos_functions(self):
        """NumPy functions in lanczos.py"""
        self.basis = lanczos_basis
        self.eigh = lanczos_eigh
        self.expm = lanczos_expm

    def setUp(self):
        self.lanczos_functions()

        self.dim = 8
        rng = np.random.default_rng(5213)
        rand_op = rng.uniform(-0.5, 0.5, (self.dim, self.dim))
        # make hermitian
        rand_op = rand_op.conj().T + rand_op
        rand_y0 = rng.uniform(-0.5, 0.5, (self.dim,))

        self.rand_op = rand_op
        self.rand_y0 = rand_y0

    def test_decomposition(self):
        """Test lanczos_basis function for correct projection."""

        tridiagonal, q_basis = self.basis(
            self.rand_op, self.rand_y0 / np.linalg.norm(self.rand_y0), self.dim
        )
        op = q_basis @ tridiagonal @ q_basis.T
        self.assertAllClose(self.rand_op, op)

    def test_ground_state(self):
        """Test lanczos_eigh function for ground state calculation."""

        q_basis, eigen_values_l, eigen_vectors_t = self.eigh(
            self.rand_op, self.rand_y0 / np.linalg.norm(self.rand_y0), self.dim
        )
        eigen_vectors_l = q_basis @ eigen_vectors_t
        eigen_values_np, eigen_vectors_np = np.linalg.eigh(self.rand_op)

        self.assertAllClose(eigen_vectors_np[:, 0], eigen_vectors_l[:, 0])
        self.assertAllClose(eigen_values_np[0], eigen_values_l[0])

    def test_expm(self):
        """Test lanczos_expm function."""

        expAy_l = self.expm(-1j * self.rand_op, self.rand_y0, self.dim)
        expAy_s = expm(-1j * self.rand_op) @ self.rand_y0

        self.assertAllClose(expAy_s, expAy_l)


class TestJaxLanczos(TestLanczos, TestJaxBase):
    """Tests for jax functions in lanczos.py"""

    def lanczos_functions(self):
        """NumPy functions in lanczos.py"""
        self.basis = jax_lanczos_basis
        self.eigh = jax_lanczos_eigh
        self.expm = jax_lanczos_expm
