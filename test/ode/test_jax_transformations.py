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
Tests for jax transformations.
"""

import numpy as np

from qiskit.quantum_info.operators import Operator
from qiskit_ode.models import HamiltonianModel
from qiskit_ode.signals import Constant, Signal
from qiskit_ode import solve_lmde
from qiskit_ode.dispatch import Array

from .common import TestJaxBase

try:
    from jax import jit, grad
    import jax.numpy as jnp
# pylint: disable=broad-except
except Exception:
    pass


class TestJaxTransformations(TestJaxBase):
    """Class for testing jax transformations of integrated use cases."""

    def setUp(self):
        """Set up a basic parameterized simulation."""

        self.w = 5.0
        self.r = 0.1

        operators = [
            2 * np.pi * self.w * Array(Operator.from_label("Z").data) / 2,
            2 * np.pi * self.r * Array(Operator.from_label("X").data) / 2,
        ]

        ham = HamiltonianModel(operators=operators)

        self.ham = ham

        def param_sim(amp, drive_freq):
            signals = [Constant(1.0), Signal(lambda t: amp, carrier_freq=drive_freq)]

            ham_copy = ham.copy()
            ham_copy.signals = signals

            results = solve_lmde(
                ham_copy,
                t_span=[0.0, 1 / self.r],
                y0=Array([0.0, 1.0], dtype=complex),
                method="jax_odeint",
                atol=1e-10,
                rtol=1e-10,
            )
            return results.y[-1]

        self.param_sim = param_sim

    def test_jit_solve_t_eval_jax_odeint(self):
        """Test compiling with a passed t_eval."""

        def t_eval_param_sim(amp, drive_freq):
            signals = [Constant(1.0), Signal(lambda t: amp, carrier_freq=drive_freq)]

            ham_copy = self.ham.copy()
            ham_copy.signals = signals

            results = solve_lmde(
                ham_copy,
                t_span=[0.0, 1 / self.r],
                y0=Array([0.0, 1.0], dtype=complex),
                method="jax_odeint",
                t_eval=[0.0, 0.5 / self.r, 1 / self.r],
                atol=1e-10,
                rtol=1e-10,
            )
            return results.y.data

        jit_sim = jit(t_eval_param_sim)

        yf = jit_sim(1.0, self.w)
        yf2 = jit_sim(2.0, self.w)

        # simple tests to verify correctness
        self.assertTrue(np.abs(yf[-1][0]) ** 2 > 0.999)
        self.assertTrue(np.abs(yf2[-1][0]) ** 2 < 0.001)

    def test_jit_solve_t_span(self):
        """Test compiling when t_span is influenced by the inputs."""

        def param_sim(amp, drive_freq):
            signals = [Constant(1.0), Signal(lambda t: amp, carrier_freq=drive_freq)]

            ham_copy = self.ham.copy()
            ham_copy.signals = signals

            results = solve_lmde(
                ham_copy,
                t_span=[0.0, (1 / self.r) * amp],
                y0=Array([0.0, 1.0], dtype=complex),
                method="jax_odeint",
                atol=1e-10,
                rtol=1e-10,
            )
            return results.y[-1]

        jit_sim = jit(param_sim)

        yf = jit_sim(1.0, self.w)

        # simple tests to verify correctness
        self.assertTrue(np.abs(yf[0]) ** 2 > 0.999)

    def test_jit_solve(self):
        """Test compiling a parameterized Hamiltonian simulation."""

        jit_sim = jit(self.param_sim)

        # run the simulation twice, make sure it compiles and runs again
        yf = jit_sim(1.0, self.w)
        yf2 = jit_sim(2.0, self.w)

        # simple tests to verify correctness
        self.assertTrue(np.abs(yf[0]) ** 2 > 0.999)
        self.assertTrue(np.abs(yf2[0]) ** 2 < 0.001)

    def test_grad_solve(self):
        """Test computing gradient of a parameterized Hamiltonian simulation."""

        def amp_to_prob(amp):
            return jnp.abs(self.param_sim(amp, self.w)[0]) ** 2

        grad_sim = grad(amp_to_prob)

        grad_p0 = grad_sim(1.0)

        self.assertTrue(np.abs(grad_p0) < 0.001)

    def test_jit_grad_solve(self):
        """Test compiling a computation of a gradient of a parameterized
        Hamiltonian simulation.
        """

        def amp_to_prob(amp):
            return jnp.abs(self.param_sim(amp, self.w)[0]) ** 2

        jit_grad_sim = jit(grad(amp_to_prob))

        grad_p0 = jit_grad_sim(1.0)

        self.assertTrue(np.abs(grad_p0) < 0.001)
