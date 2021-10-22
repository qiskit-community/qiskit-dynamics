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
Tests for correctness of exponentiation-based fixed step solvers.
"""

import numpy as np

from scipy.linalg import expm

from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.solvers.fixed_step_solvers import scipy_expm_solver, jax_expm_solver, jax_expm_parallel_solver

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

        rng = np.random.default_rng(5213)
        dim = 5
        b = 1.0  # bound on size of random terms
        rand_ops = rng.uniform(low=-b, high=b, size=(3, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(3, dim, dim)
        )
        # make anti-hermitian
        rand_ops = rand_ops - rand_ops.conj().transpose((0, 2, 1))

        self.random_y0 = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )

        def random_generator(t):
            return np.sin(t) * rand_ops[0] + t**5 * rand_ops[1] + np.exp(t) * rand_ops[2]

        self.random_generator = random_generator

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

    def test_random_generator(self):
        """Test generator with pseudo random structure."""

        t_span = np.array([2.1, 3.2])
        t_eval = np.array([2.3, 2.5, 2.78])
        y0 = self.random_y0
        gen = self.random_generator

        results = self.expm_solver(gen, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = gen

        expected_y0 = expm(0.1 * gen(2.25)) @ expm(0.1 * gen(2.15)) @ y0
        expected_y1 = expm(0.1 * gen(2.45)) @ expm(0.1 * gen(2.35)) @ expected_y0

        dt2 = (2.78 - 2.5) / 3
        expected_y2 = (
            expm(dt2 * gen(2.5 + 2.5 * dt2))
            @ expm(dt2 * gen(2.5 + 1.5 * dt2))
            @ expm(dt2 * gen(2.5 + 0.5 * dt2))
            @ expected_y1
        )

        expected_y = np.array([expected_y0, expected_y1, expected_y2])
        self.assertAllClose(expected_y, results.y)

    def test_random_generator_nonsquare_y0(self):
        """Test generator with pseudo random structure for non-square y0."""

        t_span = np.array([2.1, 3.2])
        t_eval = np.array([2.3, 2.5, 2.78])
        y0 = self.random_y0[0]
        gen = self.random_generator

        results = self.expm_solver(gen, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = gen

        expected_y0 = expm(0.1 * gen(2.25)) @ expm(0.1 * gen(2.15)) @ y0
        expected_y1 = expm(0.1 * gen(2.45)) @ expm(0.1 * gen(2.35)) @ expected_y0

        dt2 = (2.78 - 2.5) / 3
        expected_y2 = (
            expm(dt2 * gen(2.5 + 2.5 * dt2))
            @ expm(dt2 * gen(2.5 + 1.5 * dt2))
            @ expm(dt2 * gen(2.5 + 0.5 * dt2))
            @ expected_y1
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

        rng = np.random.default_rng(5213)
        dim = 5
        b = 1.0  # bound on size of random terms
        rand_ops = rng.uniform(low=-b, high=b, size=(3, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(3, dim, dim)
        )

        # make anti-hermitian
        rand_ops = rand_ops - rand_ops.conj().transpose((0, 2, 1))

        self.random_y0 = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )

        def random_generator(t):
            return jnp.sin(t) * rand_ops[0] + t**5 * rand_ops[1] + jnp.exp(1j * t) * rand_ops[2]

        self.random_generator = random_generator

        self.expm_solver = jax_expm_solver

    def test_t_span_with_jax_transformations(self):
        """Test handling of t_span as a list with jax transformations."""
        from jax import jit

        t_span = [0.0, 1.0]
        y0 = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

        def func(amp):
            results = jax_expm_solver(
                lambda t: amp * self.constant_generator(t), t_span, y0, max_dt=0.1
            )
            return Array(results.y[-1]).data

        jit_func = self.jit_wrap(func)
        output = jit_func(1.0)
        gen = self.constant_generator(0.0)
        expected_y = jexpm(1.0 * gen)
        self.assertAllClose(expected_y, output)

        grad_func = self.jit_grad_wrap(func)
        grad_func(1.0)

class TestJaxExpmParallelSolver(TestJaxExpmSolver):
    """Test cases for jax_expm_parallel_solver. Runs the same tests as
    TestJaxExpmSolver but for parallel version.
    """

    def setUp(self):
        # use super to build jax-based generators
        super().setUp()

        # set solver to jax expm parallel solver
        self.expm_solver = jax_expm_parallel_solver
