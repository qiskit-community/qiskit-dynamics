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

"""tests for solve"""

import unittest
import numpy as np
from scipy.linalg import expm

from qiskit import QiskitError
from qiskit_ode.de.de_problems import LMDEProblem, OperatorModelProblem
from qiskit_ode.models.operator_models import OperatorModel
from qiskit_ode.models.signals import Constant, Signal
from qiskit_ode.de.solve import solve
from qiskit_ode.dispatch import Array

from ..test_jax_base import TestJaxBase


class Testsolve_exceptions(unittest.TestCase):
    """Test exceptions of solve function."""

    def test_method_does_not_exist(self):
        """Test method does not exist exception."""

        try:
            solve(lambda t, y: y,
                  t_span=[0., 1.],
                  y0=np.array([1.]),
                  method='notamethod')
        except QiskitError as qe:
            self.assertTrue('method does not exist' in str(qe))


# pylint: disable=too-many-instance-attributes
class TestsolveBase(unittest.TestCase):
    """Some reusable routines for testing basic solving functionality."""

    def setUp(self):
        self.t_span = [0., 1.]
        self.y0 = Array(np.eye(2, dtype=complex))

        self.X = Array([[0., 1.], [1., 0.]], dtype=complex)
        self.Y = Array([[0., -1j], [1j, 0.]], dtype=complex)
        self.Z = Array([[1., 0.], [0., -1.]], dtype=complex)

        # simple generator and rhs
        # pylint: disable=unused-argument
        def generator(t):
            return -1j * 2 * np.pi * self.X / 2

        def rhs(t, y):
            return generator(t) @ y

        self.basic_generator = generator
        self.basic_rhs = rhs

        # define simple model
        self.w = 2.
        self.r = 0.1
        signals = [Constant(self.w), Signal(lambda t: 1., self.w)]
        operators = [-1j * 2 * np.pi * self.Z / 2,
                     -1j * 2 * np.pi * self.r * self.X / 2]
        self.basic_model = OperatorModel(operators=operators, signals=signals)

    def test_solve_w_OperatorModelProblem(self):
        """Test solve on an OperatorModelProblem."""

        problem = OperatorModelProblem(self.basic_model)

        results = solve(problem,
                        y0=Array([0., 1.], dtype=complex),
                        t_span=[0, 1 / self.r],
                        rtol=1e-9,
                        atol=1e-9)
        yf = results.y[-1]

        self.assertTrue(np.abs(yf[0])**2 > 0.999)

    def _fixed_step_LMDE_method_tests(self, method):
        results = solve(LMDEProblem(self.basic_generator),
                        t_span=self.t_span,
                        y0=self.y0,
                        method=method,
                        max_dt=0.1)

        expected = expm(-1j * np.pi * self.X.data)

        self.assertAllClose(results.y[-1], expected)

    def _variable_step_method_standard_tests(self, method):
        """tests to run on a variable step solver."""

        results = solve(self.basic_rhs,
                        t_span=self.t_span,
                        y0=self.y0,
                        method=method,
                        atol=1e-10,
                        rtol=1e-10)

        expected = expm(-1j * np.pi * self.X.data)

        self.assertAllClose(results.y[-1], expected)

        # pylint: disable=unused-argument
        def quad_rhs(t, y):
            return Array([t**2], dtype=float)

        results = solve(quad_rhs,
                        t_span=[0., 1.],
                        y0=Array([0.]),
                        method=method,
                        atol=1e-10,
                        rtol=1e-10)
        expected = Array([1./3])
        self.assertAllClose(results.y[-1], expected)

    def assertAllClose(self, A, B, rtol=1e-8, atol=1e-8):
        """Call np.allclose and assert true."""
        self.assertTrue(np.allclose(A, B, rtol=rtol, atol=atol))


class Testsolve_numpy(TestsolveBase):
    """Basic tests for `numpy`-based solver methods."""

    def test_standard_problems_solve_ivp(self):
        """Run standard tests for variable step methods in `solve_ivp`."""

        self._variable_step_method_standard_tests('RK45')
        self._variable_step_method_standard_tests('RK23')
        self._variable_step_method_standard_tests('BDF')
        self._variable_step_method_standard_tests('DOP853')


class Testsolve_jax(TestsolveBase, TestJaxBase):
    """Basic tests for jax solvers."""

    def test_jax_expm(self):
        """Test jax_expm solver."""
        self._fixed_step_LMDE_method_tests('jax_expm')

    def test_standard_problems_jax(self):
        """Run standard tests for variable step `jax` methods."""
        self._variable_step_method_standard_tests('jax_odeint')
