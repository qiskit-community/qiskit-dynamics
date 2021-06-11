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
from qiskit_dynamics.solve import setup_lmde_frames_and_generator, lmde_y0_reshape
from qiskit_dynamics.dispatch import Array

from .common import QiskitDynamicsTestCase, TestJaxBase


class TestLMDESetup(QiskitDynamicsTestCase):
    """Test solve_lmde helper functions."""

    def setUp(self):
        self.X = Array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        self.Y = Array([[0.0, -1j], [1j, 0.0]], dtype=complex)
        self.Z = Array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        # define a basic model
        w = 2.0
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2, -1j * 2 * np.pi * r * self.X / 2]
        signals = [w, Signal(1.0, w)]

        self.w = 2
        self.r = r
        self.basic_model = GeneratorModel(operators=operators, signals=signals)

        self.y0 = Array([1.0, 0.0], dtype=complex)

    def test_auto_frame_handling(self):
        """Test automatic setting of frames."""

        self.basic_model.frame = self.X

        input_frame, output_frame, generator = setup_lmde_frames_and_generator(self.basic_model)

        self.assertTrue(
            np.allclose(input_frame.frame_operator, Array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
        )
        self.assertTrue(
            np.allclose(output_frame.frame_operator, Array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
        )
        self.assertTrue(
            np.allclose(generator.frame.frame_operator, -1j * 2 * np.pi * self.w * self.Z / 2)
        )

    def test_y0_reshape(self):
        """Test automatic detection of vectorized LMDE."""

        y0 = Array(np.eye(2))

        output = lmde_y0_reshape(4, y0)
        expected = y0.flatten(order="F")

        self.assertAllClose(output, expected)

    def test_solver_cutoff_freq(self):
        """Test correct setting of solver cutoff freq."""
        _, _, generator = setup_lmde_frames_and_generator(
            self.basic_model, solver_cutoff_freq=2 * self.w
        )

        self.assertTrue(generator.cutoff_freq == 2 * self.w)
        self.assertTrue(self.basic_model.cutoff_freq is None)

    def test_generator(self):
        """Test correct evaluation of generator.
        The generator is evaluated in the solver frame in a basis in which the
        frame operator is diagonal.
        """

        _, _, generator = setup_lmde_frames_and_generator(self.basic_model, solver_frame=self.X)

        t = 13.1231

        output = generator(t, in_frame_basis=True).data

        X = np.array(self.X.data)
        X_diag, U = np.linalg.eigh(X)
        Uadj = U.conj().transpose()
        gen = (
            -1j
            * 2
            * np.pi
            * (self.w * np.array(self.Z.data) / 2 + self.r * np.cos(2 * np.pi * self.w * t) * X / 2)
        )
        expected = Uadj @ expm(1j * t * X) @ gen @ expm(-1j * t * X) @ U + 1j * np.diag(X_diag)

        self.assertAllClose(expected, output)

    def test_rhs(self):
        """Test correct evaluation of rhs.
        The generator is evaluated in the solver frame in a basis in which the
        frame operator is diagonal.
        """

        _, _, generator = setup_lmde_frames_and_generator(self.basic_model, solver_frame=self.X)

        t = 13.1231
        y = np.eye(2, dtype=complex)

        output = generator(t, y, in_frame_basis=True).data

        X = np.array(self.X.data)
        X_diag, U = np.linalg.eigh(X)
        Uadj = U.conj().transpose()
        gen = (
            -1j
            * 2
            * np.pi
            * (self.w * np.array(self.Z.data) / 2 + self.r * np.cos(2 * np.pi * self.w * t) * X / 2)
        )
        expected = (
            Uadj @ expm(1j * t * X) @ gen @ expm(-1j * t * X) @ U + 1j * np.diag(X_diag)
        ) @ y

        self.assertTrue(np.allclose(expected, output))


class TestLMDESetupJax(TestLMDESetup, TestJaxBase):
    """Jax version of TestLMDESetup tests.

    Note: This class has no body but contains tests due to inheritance.
    """


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
