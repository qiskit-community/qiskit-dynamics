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

from qiskit_ode.solvers import jax_expm_solver

from ..common import QiskitOdeTestCase, TestJaxBase

try:
    import jax.numpy as jnp
    from jax.scipy.linalg import expm as jexpm
# pylint: disable=broad-except
except Exception:
    pass


class TestJaxExpmSolver(QiskitOdeTestCase, TestJaxBase):
    """Test cases for jax_expm_solver."""

    def setUp(self):
        # some generators for testing
        self.constant_generator = lambda t: -1j * jnp.array([[0., 1.], [1., 0.]])
        self.linear_generator = lambda t: -1j * jnp.array([[0., 1. - 1j * t], [1. + 1j * t, 0.]])

    def test_t_eval_arg_no_overlap(self):
        """Test handling of t_eval when no overlap with t_span."""

        t_span = np.array([0., 1.])
        t_eval = np.array([0.3, 0.5, 0.78])
        y0 = jnp.array([[1., 0.], [0., 1.]], dtype=complex)

        results = jax_expm_solver(self.constant_generator,
                                  t_span,
                                  y0,
                                  max_dt=0.1,
                                  t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = self.constant_generator(0.)

        expected_y = jnp.array([jexpm(0.3 * gen), jexpm(0.5 * gen), jexpm(0.78 * gen)])

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_no_overlap_backwards(self):
        """Test handling of t_eval when no overlap with t_span for backwards integration."""

        gen = self.constant_generator(0.)
        t_span = np.array([1., 0.])
        t_eval = np.array([0.78, 0.5, 0.3])
        y0 = jexpm(1. * gen)

        results = jax_expm_solver(self.constant_generator,
                                  t_span,
                                  y0,
                                  max_dt=0.1,
                                  t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y = jnp.array([jexpm(0.78 * gen), jexpm(0.5 * gen), jexpm(0.3 * gen)])

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap(self):
        """Test handling of t_eval with overlap with t_span."""

        t_span = np.array([0., 1.])
        t_eval = np.array([0., 0.5, 0.78])
        y0 = jnp.array([[1., 0.], [0., 1.]], dtype=complex)

        results = jax_expm_solver(self.constant_generator,
                                  t_span,
                                  y0,
                                  max_dt=0.1,
                                  t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = self.constant_generator(0.)

        expected_y = jnp.array([jexpm(0. * gen), jexpm(0.5 * gen), jexpm(0.78 * gen)])

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap_backwards(self):
        """Test handling of t_eval with overlap with t_span for backwards integration."""

        gen = self.constant_generator(0.)

        t_span = np.array([1., 0.])
        t_eval = np.array([0.78, 0.5, 0.])
        y0 = jexpm(1. * gen)

        results = jax_expm_solver(self.constant_generator,
                                  t_span,
                                  y0,
                                  max_dt=0.1,
                                  t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y = jnp.array([jexpm(0.78 * gen), jexpm(0.5 * gen), jexpm(0. * gen)])

        self.assertAllClose(expected_y, results.y)

    def test_solve_linear_generator(self):
        """Test solving a problem with non-trivial behaviour."""

        t_span = np.array([0., 1.])
        t_eval = np.array([0.3, 0.5, 0.78])
        y0 = jnp.array([[1., 0.], [0., 1.]], dtype=complex)

        results = jax_expm_solver(self.linear_generator,
                                  t_span,
                                  y0,
                                  max_dt=0.1,
                                  t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = self.linear_generator

        expected_y0 = jexpm(0.1 * gen(0.25)) @ jexpm(0.1 * gen(0.15)) @ jexpm(0.1 * gen(0.05))
        expected_y1 = jexpm(0.1 * gen(0.45)) @ jexpm(0.1 * gen(0.35)) @ expected_y0

        dt2 = (0.78 - 0.5) / 3
        expected_y2 = (jexpm(dt2 * gen(0.5 + 2.5 * dt2)) @
                       jexpm(dt2 * gen(0.5 + 1.5 * dt2)) @
                       jexpm(dt2 * gen(0.5 + 0.5 * dt2)) @
                       expected_y1)

        expected_y = jnp.array([expected_y0, expected_y1, expected_y2])

        self.assertAllClose(expected_y, results.y)

    def test_solve_linear_generator_backwards(self):
        """Test solving a problem with non-trivial behaviour backwards."""

        t_span = np.array([1., 0.])
        t_eval = np.array([0.78, 0.5, 0.3])
        y0 = jnp.array([[1., 0.], [0., 1.]], dtype=complex)

        results = jax_expm_solver(self.linear_generator,
                                  t_span,
                                  y0,
                                  max_dt=0.1,
                                  t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        gen = self.linear_generator

        dt0 = (1. - 0.78) / 3
        expected_y0 = (jexpm(- dt0 * gen(1. - 2.5 * dt0)) @
                       jexpm(- dt0 * gen(1. - 1.5 * dt0)) @
                       jexpm(- dt0 * gen(1. - 0.5 * dt0)))
        dt1 = (0.78 - 0.5) / 3
        expected_y1 = (jexpm(- dt1 * gen(0.78 - 2.5 * dt1)) @
                       jexpm(- dt1 * gen(0.78 - 1.5 * dt1)) @
                       jexpm(- dt1 * gen(0.78 - 0.5 * dt1)) @
                       expected_y0)
        expected_y2 = (jexpm(- 0.1 * gen(0.5 - 1.5 * 0.1)) @
                       jexpm(- 0.1 * gen(0.5 - 0.5 * 0.1)) @
                       expected_y1)

        expected_y = jnp.array([expected_y0, expected_y1, expected_y2])

        self.assertAllClose(expected_y, results.y)
