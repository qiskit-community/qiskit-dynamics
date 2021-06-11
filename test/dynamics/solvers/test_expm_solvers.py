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
# pylint: disable=invalid-name

"""
Direct tests of jax_expm_solver
"""

import numpy as np

from scipy.linalg import expm

from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.solvers import scipy_expm_solver, jax_expm_solver

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    import jax.numpy as jnp
    from jax.scipy.linalg import expm as jexpm
# pylint: disable=broad-except
except Exception:
    pass


class TestExpmSolver(QiskitDynamicsTestCase):
    """Test cases for scipy_expm_solver."""

    def setUp(self):
        # some generators for testing
        self.constant_generator = lambda t: -1j * np.array([[0.0, 1.0], [1.0, 0.0]])
        self.linear_generator = lambda t: -1j * np.array([[0.0, 1.0 - 1j * t], [1.0 + 1j * t, 0.0]])

        self.expm_solver = scipy_expm_solver

    def test_t_eval_arg_no_overlap(self):
        """Test handling of t_eval when no overlap with t_span."""

        t_span = np.array([0.0, 1.0])
        t_eval = np.array([0.3, 0.5, 0.78])
        y0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

        results = self.expm_solver(self.constant_generator, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = self.constant_generator(0.0)

        expected_y = np.array([expm(0.3 * gen), expm(0.5 * gen), expm(0.78 * gen)])

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_no_overlap_backwards(self):
        """Test handling of t_eval when no overlap with t_span for backwards integration."""

        gen = self.constant_generator(0.0)
        t_span = np.array([1.0, 0.0])
        t_eval = np.array([0.78, 0.5, 0.3])
        y0 = expm(1.0 * gen)

        results = self.expm_solver(self.constant_generator, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y = np.array([expm(0.78 * gen), expm(0.5 * gen), expm(0.3 * gen)])

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap(self):
        """Test handling of t_eval with overlap with t_span."""

        t_span = np.array([0.0, 1.0])
        t_eval = np.array([0.0, 0.5, 0.78])
        y0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

        results = self.expm_solver(self.constant_generator, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = np.array(self.constant_generator(0.0))

        expected_y = np.array([expm(0.0 * gen), expm(0.5 * gen), expm(0.78 * gen)])

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap_backwards(self):
        """Test handling of t_eval with overlap with t_span for backwards integration."""

        gen = np.array(self.constant_generator(0.0))

        t_span = np.array([1.0, 0.0])
        t_eval = np.array([0.78, 0.5, 0.0])
        y0 = expm(1.0 * gen)

        results = self.expm_solver(self.constant_generator, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y = np.array([expm(0.78 * gen), expm(0.5 * gen), expm(0.0 * gen)])

        self.assertAllClose(expected_y, results.y)

    def test_solve_linear_generator(self):
        """Test solving a problem with non-trivial behaviour."""

        t_span = np.array([0.0, 1.0])
        t_eval = np.array([0.3, 0.5, 0.78])
        y0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

        results = self.expm_solver(self.linear_generator, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = self.linear_generator

        expected_y0 = expm(0.1 * gen(0.25)) @ expm(0.1 * gen(0.15)) @ expm(0.1 * gen(0.05))
        expected_y1 = expm(0.1 * gen(0.45)) @ expm(0.1 * gen(0.35)) @ expected_y0

        dt2 = (0.78 - 0.5) / 3
        expected_y2 = (
            expm(dt2 * gen(0.5 + 2.5 * dt2))
            @ expm(dt2 * gen(0.5 + 1.5 * dt2))
            @ expm(dt2 * gen(0.5 + 0.5 * dt2))
            @ expected_y1
        )

        expected_y = np.array([expected_y0, expected_y1, expected_y2])

        self.assertAllClose(expected_y, results.y)

    def test_solve_linear_generator_backwards(self):
        """Test solving a problem with non-trivial behaviour backwards."""

        t_span = np.array([1.0, 0.0])
        t_eval = np.array([0.78, 0.5, 0.3])
        y0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

        results = self.expm_solver(self.linear_generator, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = self.linear_generator

        dt0 = (1.0 - 0.78) / 3
        expected_y0 = (
            expm(-dt0 * gen(1.0 - 2.5 * dt0))
            @ expm(-dt0 * gen(1.0 - 1.5 * dt0))
            @ expm(-dt0 * gen(1.0 - 0.5 * dt0))
        )
        dt1 = (0.78 - 0.5) / 3
        expected_y1 = (
            expm(-dt1 * gen(0.78 - 2.5 * dt1))
            @ expm(-dt1 * gen(0.78 - 1.5 * dt1))
            @ expm(-dt1 * gen(0.78 - 0.5 * dt1))
            @ expected_y0
        )
        expected_y2 = (
            expm(-0.1 * gen(0.5 - 1.5 * 0.1)) @ expm(-0.1 * gen(0.5 - 0.5 * 0.1)) @ expected_y1
        )

        expected_y = np.array([expected_y0, expected_y1, expected_y2])

        self.assertAllClose(expected_y, results.y)


class TestJaxExpmSolver(TestExpmSolver, TestJaxBase):
    """Test cases for jax_expm_solver."""

    def setUp(self):
        # some generators for testing
        self.constant_generator = lambda t: -1j * jnp.array([[0.0, 1.0], [1.0, 0.0]])
        self.linear_generator = lambda t: -1j * jnp.array(
            [[0.0, 1.0 - 1j * t], [1.0 + 1j * t, 0.0]]
        )

        self.expm_solver = jax_expm_solver

    def test_t_span_with_jit(self):
        """Test handling of t_span as a list with jit."""

        from jax import jit

        t_span = [0.0, 1.0]
        y0 = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

        def func(amp):
            results = jax_expm_solver(
                lambda t: amp * self.constant_generator(t), t_span, y0, max_dt=0.1
            )
            return Array(results.y[-1]).data

        jit_func = jit(func)

        output = jit_func(1.0)

        gen = self.constant_generator(0.0)

        expected_y = jexpm(1.0 * gen)

        self.assertAllClose(expected_y, output)
