# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,broad-except

"""Tests for solve_lmde and related functions."""

import numpy as np
from scipy.linalg import expm

from qiskit_dynamics.models import GeneratorModel
from qiskit_dynamics.signals import Signal
from qiskit_dynamics import solve_lmde
from qiskit_dynamics.dispatch import Array

from .common import QiskitDynamicsTestCase, TestJaxBase


# pylint: disable=too-many-instance-attributes
class Testsolve_lmde_Base(QiskitDynamicsTestCase):
    """Some reusable routines for high level solve_lmde tests."""

    def setUp(self):
        self.t_span = [0.0, 1.0]
        self.y0 = Array(np.eye(2, dtype=complex))

        self.X = Array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        self.Y = Array([[0.0, -1j], [1j, 0.0]], dtype=complex)
        self.Z = Array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        # simple generator and rhs
        # pylint: disable=unused-argument
        def generator(t):
            return -1j * 2 * np.pi * self.X / 2

        self.basic_generator = generator

    def _fixed_step_LMDE_method_tests(self, method):
        results = solve_lmde(
            self.basic_generator, t_span=self.t_span, y0=self.y0, method=method, max_dt=0.1
        )

        expected = expm(-1j * np.pi * self.X.data)

        self.assertAllClose(results.y[-1], expected)


class Testsolve_lmde_scipy_expm(Testsolve_lmde_Base):
    """Basic tests for solve_lmde with method=='expm'."""

    def test_scipy_expm_solver(self):
        """Test scipy_expm_solver."""
        self._fixed_step_LMDE_method_tests("scipy_expm")


class Testsolve_lmde_jax_expm(Testsolve_lmde_Base, TestJaxBase):
    """Basic tests for solve_lmde with method=='jax_expm'."""

    def test_jax_expm_solver(self):
        """Test jax_expm_solver."""
        self._fixed_step_LMDE_method_tests("jax_expm")
