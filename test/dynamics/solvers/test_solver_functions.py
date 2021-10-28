# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Standardized test cases for results of calls to solve_lmde and solve_ode,
for both variable and fixed-step methods.

These tests set up common test cases through inheritance of test classes.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import expm

from qiskit import QiskitError
from qiskit.quantum_info import Operator

from qiskit_dynamics.models import GeneratorModel, HamiltonianModel, LindbladModel
from qiskit_dynamics.signals import Signal, DiscreteSignal
from qiskit_dynamics import solve_ode, solve_lmde
from qiskit_dynamics.solvers.solver_functions import (
    setup_generator_model_rhs_y0_in_frame_basis,
    results_y_out_of_frame_basis,
)
from qiskit_dynamics.dispatch import Array

from ..common import QiskitDynamicsTestCase, TestJaxBase


class ModelSetup(ABC):
    """Abstract base class for setting up models and RHS to be used in tests"""

    def setUp(self):
        """Construct standardized RHS functions and models."""

        self.t_span = [0.0, 1.0]
        self.y0 = Array(np.eye(2, dtype=complex))

        self.X = Array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

        op = -1j * 2 * np.pi * self.X / 2

        # simple generator/RHS
        # pylint: disable=unused-argument
        def basic_rhs(t, y=None):
            if y is None:
                return op
            else:
                return op @ y

        self.basic_rhs = basic_rhs

        # construct randomized RHS
        dim = 7
        b = 0.5
        rng = np.random.default_rng(3093)
        static_operator = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        operators = rng.uniform(low=-b, high=b, size=(1, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(1, dim, dim)
        )
        frame_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = frame_op.conj().transpose() - frame_op
        y0 = rng.uniform(low=-b, high=b, size=(dim,)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim,)
        )

        self.pseudo_random_y0 = y0

        self.pseudo_random_signal = DiscreteSignal(
            samples=rng.uniform(low=-b, high=b, size=(5,)), dt=0.1, carrier_freq=1.0
        )
        self.pseudo_random_model = GeneratorModel(
            operators=operators,
            signals=[self.pseudo_random_signal],
            static_operator=static_operator,
            rotating_frame=frame_op,
        )

        # simulate directly out of frame
        def pseudo_random_rhs(t, y=None):
            op = static_operator + self.pseudo_random_signal(t) * operators[0]
            if y is None:
                return op
            else:
                return op @ y

        self.pseudo_random_rhs = pseudo_random_rhs


class TestFixedStepMethod(ModelSetup, QiskitDynamicsTestCase):
    """Abstract base class for testing fixed step methods."""

    @abstractmethod
    def solve(self, rhs, t_span, y0, t_eval=None, **kwargs):
        """Call the solver to test."""

    @property
    def tol(self):
        """Tolerance to use when checking results."""
        return 1e-5

    def test_basic_model(self):
        """Test case for basic model."""

        results = self.solve(self.basic_rhs, t_span=self.t_span, y0=self.y0)

        expected = expm(-1j * np.pi * self.X.data)

        self.assertAllClose(results.y[-1], expected, atol=self.tol, rtol=self.tol)

    def test_pseudo_random_model(self):
        """Test case for pseudorandom model, compare to directly-defined RHS."""

        # test solving in a frame with generator model and solving with a function
        # and no frame
        results = self.solve(self.pseudo_random_model, t_span=[0, 0.5], y0=self.pseudo_random_y0)
        # verify that model is not in the frame basis
        self.assertFalse(self.pseudo_random_model.in_frame_basis)

        yf = self.pseudo_random_model.rotating_frame.state_out_of_frame(0.5, results.y[-1])
        results2 = self.solve(self.pseudo_random_rhs, t_span=[0, 0.5], y0=self.pseudo_random_y0)
        self.assertAllClose(yf, results2.y[-1], atol=self.tol, rtol=self.tol)

        # test solving in frame basis and compare to previous result
        self.pseudo_random_model.in_frame_basis = True
        rotating_frame = self.pseudo_random_model.rotating_frame
        y0_in_frame_basis = rotating_frame.state_into_frame_basis(self.pseudo_random_y0)
        results3 = self.solve(self.pseudo_random_model, t_span=[0, 0.5], y0=y0_in_frame_basis)
        yf_in_frame_basis = results3.y[-1]
        self.assertAllClose(
            yf,
            rotating_frame.state_out_of_frame(
                0.5, y=yf_in_frame_basis, y_in_frame_basis=True, return_in_frame_basis=False
            ),
            atol=self.tol,
            rtol=self.tol,
        )
        self.assertTrue(self.pseudo_random_model.in_frame_basis)


class TestFixedStepMethodJax(TestFixedStepMethod, TestJaxBase):
    """JAX version of TestFixedStepMethod. Adds additional jit/grad test."""

    def test_pseudo_random_jit_grad(self):
        """Validate jitting and gradding through the method at the level of
        solve_ode/solve_lmde.
        """

        def func(a):
            model_copy = self.pseudo_random_model.copy()
            model_copy.signals = [Signal(Array(a), carrier_freq=1.0)]
            results = self.solve(model_copy, t_span=[0.0, 0.1], y0=self.pseudo_random_y0)
            return results.y[-1]

        jit_func = self.jit_wrap(func)
        self.assertAllClose(jit_func(1.0), func(1.0))

        # just verify that this runs without error
        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)


class TestRK4(TestFixedStepMethod):
    """Test class for RK4_solver."""

    def solve(self, rhs, t_span, y0, t_eval=None, **kwargs):
        return solve_ode(
            rhs=rhs, t_span=t_span, y0=y0, method="RK4", t_eval=t_eval, max_dt=0.001, **kwargs
        )


class Testjax_RK4(TestFixedStepMethodJax):
    """Test class for jax_RK4_solver."""

    def solve(self, rhs, t_span, y0, t_eval=None, **kwargs):
        return solve_ode(
            rhs=rhs, t_span=t_span, y0=y0, method="jax_RK4", t_eval=t_eval, max_dt=0.001, **kwargs
        )


class Testjax_RK4_parallel(TestFixedStepMethodJax):
    """Test class for jax_RK4_parallel_solver."""

    def solve(self, rhs, t_span, y0, t_eval=None, **kwargs):
        return solve_lmde(
            generator=rhs,
            t_span=t_span,
            y0=y0,
            method="jax_RK4_parallel",
            t_eval=t_eval,
            max_dt=0.001,
            **kwargs,
        )


class Testscipy_expm(TestFixedStepMethod):
    """Test class for scipy_expm_solver."""

    def solve(self, rhs, t_span, y0, t_eval=None, **kwargs):
        return solve_lmde(
            generator=rhs,
            t_span=t_span,
            y0=y0,
            method="scipy_expm",
            t_eval=t_eval,
            max_dt=0.01,
            **kwargs,
        )


class Testjax_expm(TestFixedStepMethodJax):
    """Test class for jax_expm_solver."""

    def solve(self, rhs, t_span, y0, t_eval=None, **kwargs):
        return solve_lmde(
            generator=rhs,
            t_span=t_span,
            y0=y0,
            method="jax_expm",
            t_eval=t_eval,
            max_dt=0.01,
            **kwargs,
        )


class Testjax_expm_parallel(TestFixedStepMethodJax):
    """Test class for jax_expm_parallel_solver."""

    def solve(self, rhs, t_span, y0, t_eval=None, **kwargs):
        return solve_lmde(
            generator=rhs,
            t_span=t_span,
            y0=y0,
            method="jax_expm_parallel",
            t_eval=t_eval,
            max_dt=0.01,
            **kwargs,
        )


# delete abstract classes so unittest doesn't attempt to run them
del TestFixedStepMethod, TestFixedStepMethodJax
