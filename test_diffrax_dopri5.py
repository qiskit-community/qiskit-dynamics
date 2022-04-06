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
Direct tests of diffrax_dopri5
"""

import numpy as np

from qiskit_dynamics.solvers.jax_diffrax import diffrax_dopri5

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    import jax.numpy as jnp
    from jax.lax import cond
# pylint: disable=broad-except
except Exception:
    pass


class TestJaxOdeint(QiskitDynamicsTestCase, TestJaxBase):
    """Test cases for diffrax_dopri5."""

    def setUp(self):

        # pylint: disable=unused-argument
        def simple_rhs(t, y):
            return cond(t < 1.0, lambda s: s, lambda s: s**2, jnp.array([t]))

        self.simple_rhs = simple_rhs

    def test_t_eval_arg_no_overlap(self):
        """Test handling of t_eval when no overlap with t_span."""

        t_span = np.array([0.0, 2.0])
        t_eval = np.array([1.0, 1.5, 1.7])
        y0 = jnp.array([1.0])

        results = diffrax_dopri5(self.simple_rhs, t_span, y0, t_eval=t_eval, atol=1e-10, rtol=1e-10)

        self.assertAllClose(t_eval, results.t)

        expected_y = jnp.array(
            [
                [1 + 0.5],
                [1 + 0.5 + (1.5**3 - 1.0**3) / 3],
                [1 + 0.5 + (1.7**3 - 1.0**3) / 3],
            ]
        )

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_no_overlap_backwards(self):
        """Test handling of t_eval when no overlap with t_span with backwards integration."""

        t_span = np.array([2.0, 0.0])
        t_eval = np.array([1.7, 1.5, 1.0])
        y0 = jnp.array([1 + 0.5 + (2.0**3 - 1.0**3) / 3])

        results = diffrax_dopri5(self.simple_rhs, t_span, y0, t_eval=t_eval, atol=1e-10, rtol=1e-10)

        self.assertAllClose(t_eval, results.t)

        expected_y = jnp.array(
            [
                [1 + 0.5 + (1.7**3 - 1.0**3) / 3],
                [1 + 0.5 + (1.5**3 - 1.0**3) / 3],
                [1 + 0.5],
            ]
        )

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap(self):
        """Test handling of t_eval with overlap with t_span."""

        t_span = np.array([0.0, 2.0])
        t_eval = np.array([1.0, 1.5, 1.7, 2.0])
        y0 = jnp.array([1.0])

        results = diffrax_dopri5(self.simple_rhs, t_span, y0, t_eval=t_eval, atol=1e-10, rtol=1e-10)

        self.assertAllClose(t_eval, results.t)

        expected_y = jnp.array(
            [
                [1 + 0.5],
                [1 + 0.5 + (1.5**3 - 1.0**3) / 3],
                [1 + 0.5 + (1.7**3 - 1.0**3) / 3],
                [1 + 0.5 + (2**3 - 1.0**3) / 3],
            ]
        )

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap_backwards(self):
        """Test handling of t_eval with overlap with t_span with backwards integration."""

        t_span = np.array([2.0, 0.0])
        t_eval = np.array([2.0, 1.7, 1.5, 1.0])
        y0 = jnp.array([1 + 0.5 + (2.0**3 - 1.0**3) / 3])

        results = diffrax_dopri5(self.simple_rhs, t_span, y0, t_eval=t_eval, atol=1e-10, rtol=1e-10)

        self.assertAllClose(t_eval, results.t)

        expected_y = jnp.array(
            [
                [1 + 0.5 + (2**3 - 1.0**3) / 3],
                [1 + 0.5 + (1.7**3 - 1.0**3) / 3],
                [1 + 0.5 + (1.5**3 - 1.0**3) / 3],
                [1 + 0.5],
            ]
        )

        self.assertAllClose(expected_y, results.y)
