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

"""
Tests for algebraic operations on signals.
"""

import numpy as np

from qiskit_ode.signals import Signal, Constant, DiscreteSignal
from qiskit_ode.signals.signals import SignalSum, DiscreteSignalSum
from qiskit_ode.dispatch import Array

from ..common import QiskitOdeTestCase, TestJaxBase

try:
    from jax import jit, grad
    import jax.numpy as jnp
except ImportError:
    pass

class TestSignalAddition(QiskitOdeTestCase):
    """Testing special handling of signal addition."""

    def test_SignalSum_construction(self):
        """Test correct construction of signal sum."""

        sig_sum = Constant(1.) + Signal(lambda t: t)
        self.assertTrue(isinstance(sig_sum, SignalSum))

        self.assertAllClose(sig_sum(3.), 4.)

    def test_DiscreteSignalSum_construction(self):
        """Verify that DiscreteSignals with the same sample structure produce
        a DiscreteSignalSum.
        """

        sig_sum = (DiscreteSignal(dt=0.5, samples=np.array([1., 2., 3.]), start_time=1.)
                    + DiscreteSignal(dt=0.5, samples=np.array([4., 5., 6.]), start_time=1.))

        self.assertTrue(isinstance(sig_sum, DiscreteSignalSum))
        self.assertAllClose(sig_sum.samples, np.array([[1., 4.], [2., 5.], [3., 6.]]))
        self.assertAllClose(sig_sum.envelope(1.5), np.array([2., 5.]))

        sig_sum2 = sig_sum + DiscreteSignal(dt=0.5, samples=np.array([1., 2., 3.]), start_time=1.)
        self.assertTrue(isinstance(sig_sum2, DiscreteSignalSum))
        self.assertAllClose(sig_sum2.samples, np.array([[1., 4., 1.], [2., 5., 2.], [3., 6., 3.]]))
        self.assertAllClose(sig_sum2.envelope(1.5), np.array([2., 5., 2.]))

    def test_scalar_addition(self):
        """Test addition of a scalar with a signal."""

        sig_sum = 1. + Signal(3., carrier_freq=2.)
        self.assertTrue(isinstance(sig_sum, SignalSum))
        self.assertTrue(isinstance(sig_sum[1], Constant)) # calls __radd__
        self.assertAllClose(sig_sum.envelope(1.5), np.array([3., 1.]))

        sig_sum = Signal(3., carrier_freq=2.) - 1
        self.assertTrue(isinstance(sig_sum, SignalSum))
        self.assertTrue(isinstance(sig_sum[1], Constant)) # calls __radd__
        self.assertAllClose(sig_sum.envelope(1.5), np.array([3., -1.]))


class TestSignalMultiplication(QiskitOdeTestCase):
    """Test special handling of signal multiplication."""

    def test_DiscreteSignal_products(self):
        """Test special handling for products of discrete signals."""

        sig1 = DiscreteSignal(dt=0.5, samples=[1, 2, 3], start_time=0, carrier_freq=3., phase=0.1)
        sig2 = DiscreteSignal(dt=0.5, samples=[4j, 5, 6], start_time=0, carrier_freq=2., phase=3.5)
        sig_prod = sig1 * sig2

        self.assertAllClose(sig_prod.samples, 0.5 * np.array([[4j, -4j], [10, 10], [18, 18]]))
        self.assertAllClose(sig_prod.carrier_freq, np.array([5., 1.]))
        self.assertAllClose(sig_prod.phase, np.array([3.6, -3.4]))

        t_vals = np.array([0.1, 0.2, 1.231])
        self.assertAllClose(sig_prod(t_vals), sig1(t_vals) * sig2(t_vals))

    def test_DiscreteSignalSum_products(self):
        """More advanced test case for discrete signals."""

        sig1 = DiscreteSignal(dt=0.5, samples=[1, 2, 3], start_time=0, carrier_freq=3., phase=0.1)
        sig2 = DiscreteSignal(dt=0.5, samples=[4j, 5, 6], start_time=0, carrier_freq=2., phase=3.5)
        sig3 = DiscreteSignal(dt=0.5, samples=[-1, 2j, 3], start_time=0, carrier_freq=1., phase=1.5)

        sig_prod = (sig1 * sig2) * sig3

        expected_samples = 0.25 * np.array([[-4j, 4j, -4j, 4j],
                                            [20j, 20j, -20j, -20j],
                                            [54, 54, 54, 54]])
        expected_freqs = np.array([6., 2., 4., 0.])
        expected_phases = np.array([5.1, -1.9, 2.1, -4.9])

        self.assertAllClose(sig_prod.samples, expected_samples)
        self.assertAllClose(sig_prod.carrier_freq, expected_freqs)
        self.assertAllClose(sig_prod.phase, expected_phases)

        t_vals = np.array([0.1, 0.2, 1.231])
        self.assertAllClose(sig_prod(t_vals), sig1(t_vals) * sig2(t_vals) * sig3(t_vals))



class TestSignalAdditionJax(TestSignalAddition, TestJaxBase):
    """Jax version of TestSignalAddition."""


class TestSignalMultiplicationJax(TestSignalMultiplication, TestJaxBase):
    """Jax version of TestSignalMultiplication."""
