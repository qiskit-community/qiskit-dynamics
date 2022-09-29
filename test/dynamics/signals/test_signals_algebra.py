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

from qiskit_dynamics.signals import Signal, DiscreteSignal, SignalSum, DiscreteSignalSum

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit, grad
    import jax.numpy as jnp
except ImportError:
    pass


class TestSignalAddition(QiskitDynamicsTestCase):
    """Testing special handling of signal addition."""

    def test_SignalSum_construction(self):
        """Test correct construction of signal sum."""

        sig_sum = 1.0 + Signal(lambda t: t)
        self.assertTrue(isinstance(sig_sum, SignalSum))

        self.assertAllClose(sig_sum(3.0), 4.0)

    def test_empty_SignalSum(self):
        """Test construction of signal sum with empty element."""
        sig_sum = 1.0 + DiscreteSignal(samples=[], dt=1.0)
        self.assertTrue(isinstance(sig_sum, SignalSum))
        self.assertAllClose(sig_sum(3.0), 1.0)


    def test_DiscreteSignalSum_construction(self):
        """Verify that DiscreteSignals with the same sample structure produce
        a DiscreteSignalSum.
        """

        sig_sum = DiscreteSignal(
            dt=0.5, samples=np.array([1.0, 2.0, 3.0]), start_time=1.0
        ) + DiscreteSignal(dt=0.5, samples=np.array([4.0, 5.0, 6.0]), start_time=1.0)

        self.assertTrue(isinstance(sig_sum, DiscreteSignalSum))
        self.assertAllClose(sig_sum.samples, np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
        self.assertAllClose(sig_sum.envelope(1.5), np.array([2.0, 5.0]))

        sig_sum2 = sig_sum + DiscreteSignal(
            dt=0.5, samples=np.array([1.0, 2.0, 3.0]), start_time=1.0
        )
        self.assertTrue(isinstance(sig_sum2, DiscreteSignalSum))
        self.assertAllClose(
            sig_sum2.samples, np.array([[1.0, 4.0, 1.0], [2.0, 5.0, 2.0], [3.0, 6.0, 3.0]])
        )
        self.assertAllClose(sig_sum2.envelope(1.5), np.array([2.0, 5.0, 2.0]))

    def test_scalar_addition(self):
        """Test addition of a scalar with a signal."""

        sig_sum = 1.0 + Signal(3.0, carrier_freq=2.0)
        self.assertTrue(isinstance(sig_sum, SignalSum))
        self.assertTrue(sig_sum[1].is_constant)  # calls __radd__
        self.assertAllClose(sig_sum.envelope(1.5), np.array([3.0, 1.0]))

        sig_sum = Signal(3.0, carrier_freq=2.0) - 1
        self.assertTrue(isinstance(sig_sum, SignalSum))
        self.assertTrue(sig_sum[1].is_constant)  # calls __radd__
        self.assertAllClose(sig_sum.envelope(1.5), np.array([3.0, -1.0]))


class TestSignalMultiplication(QiskitDynamicsTestCase):
    """Test special handling of signal multiplication."""

    def test_DiscreteSignal_products(self):
        """Test special handling for products of discrete signals."""

        sig1 = DiscreteSignal(dt=0.5, samples=[1, 2, 3], start_time=0, carrier_freq=3.0, phase=0.1)
        sig2 = DiscreteSignal(dt=0.5, samples=[4j, 5, 6], start_time=0, carrier_freq=2.0, phase=3.5)
        sig_prod = sig1 * sig2

        self.assertAllClose(sig_prod.samples, 0.5 * np.array([[4j, -4j], [10, 10], [18, 18]]))
        self.assertAllClose(sig_prod.carrier_freq, np.array([5.0, 1.0]))
        self.assertAllClose(sig_prod.phase, np.array([3.6, -3.4]))

        t_vals = np.array([0.1, 0.2, 1.231])
        self.assertAllClose(sig_prod(t_vals), sig1(t_vals) * sig2(t_vals))

    def test_DiscreteSignalSum_products(self):
        """More advanced test case for discrete signals."""

        sig1 = DiscreteSignal(dt=0.5, samples=[1, 2, 3], start_time=0, carrier_freq=3.0, phase=0.1)
        sig2 = DiscreteSignal(dt=0.5, samples=[4j, 5, 6], start_time=0, carrier_freq=2.0, phase=3.5)
        sig3 = DiscreteSignal(
            dt=0.5, samples=[-1, 2j, 3], start_time=0, carrier_freq=1.0, phase=1.5
        )

        sig_prod = (sig1 * sig2) * sig3

        expected_samples = 0.25 * np.array(
            [[-4j, 4j, -4j, 4j], [20j, 20j, -20j, -20j], [54, 54, 54, 54]]
        )
        expected_freqs = np.array([6.0, 2.0, 4.0, 0.0])
        expected_phases = np.array([5.1, -1.9, 2.1, -4.9])

        self.assertAllClose(sig_prod.samples, expected_samples)
        self.assertAllClose(sig_prod.carrier_freq, expected_freqs)
        self.assertAllClose(sig_prod.phase, expected_phases)

        t_vals = np.array([0.1, 0.2, 1.231])
        self.assertAllClose(sig_prod(t_vals), sig1(t_vals) * sig2(t_vals) * sig3(t_vals))

    def test_constant_product(self):
        """Test special handling of constant products."""

        sig_prod = Signal(3.0) * Signal(2.0)

        self.assertTrue(len(sig_prod) == 1)
        self.assertTrue(sig_prod[0].is_constant)
        self.assertAllClose(sig_prod(0.1), 6.0)

    def test_constant_discrete_product(self):
        """Test constant multiplied by DiscreteSignalSum."""

        sig_prod = 3.0 * DiscreteSignal(
            dt=0.5, samples=[1, 2, 3], start_time=0, carrier_freq=3.0, phase=0.1
        )

        self.assertTrue(isinstance(sig_prod, DiscreteSignalSum))
        self.assertAllClose(sig_prod.samples, 3 * np.array([[1], [2], [3]]))
        self.assertAllClose(sig_prod.carrier_freq, np.array([3.0]))
        self.assertAllClose(sig_prod.phase, np.array([0.1]))

        sig_prod = DiscreteSignal(
            dt=0.5, samples=[1, 2, 3], start_time=0, carrier_freq=3.0, phase=0.1
        ) * Signal(3.0)

        self.assertTrue(isinstance(sig_prod, DiscreteSignalSum))
        self.assertAllClose(sig_prod.samples, 3 * np.array([[1], [2], [3]]))
        self.assertAllClose(sig_prod.carrier_freq, np.array([3.0]))
        self.assertAllClose(sig_prod.phase, np.array([0.1]))

    def test_constant_signal_product(self):
        """Test constant multiplied by a general Signal."""

        sig_prod = 3.0 * Signal(lambda t: t, carrier_freq=3.0, phase=0.1)

        self.assertTrue(isinstance(sig_prod, SignalSum))
        self.assertTrue(len(sig_prod) == 1)
        self.assertAllClose(
            sig_prod(2.0), np.real(3.0 * 2.0 * np.exp(1j * (2 * np.pi * 3.0 * 2.0 + 0.1)))
        )

        sig_prod = Signal(lambda t: t, carrier_freq=3.0, phase=0.1) * Signal(3.0)

        self.assertTrue(isinstance(sig_prod, SignalSum))
        self.assertTrue(len(sig_prod) == 1)
        self.assertAllClose(
            sig_prod(2.0), np.real(3.0 * 2.0 * np.exp(1j * (2 * np.pi * 3.0 * 2.0 + 0.1)))
        )

    def test_signal_signal_product(self):
        """Test Signal x Signal."""

        sig1 = Signal(lambda t: t, carrier_freq=3.0, phase=0.1)
        sig2 = Signal(lambda t: t**2, carrier_freq=2.1, phase=1.1)
        sig3 = Signal(lambda t: t**3, carrier_freq=2.1, phase=1.1)

        sig_prod = sig1 * sig2 * sig3

        self.assertTrue(isinstance(sig_prod, SignalSum))
        self.assertTrue(len(sig_prod) == 4)
        t_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.2312])
        s1_vals = sig1.complex_value(t_vals)
        s2_vals = sig2.complex_value(t_vals)
        s3_vals = sig3.complex_value(t_vals)
        expected = 0.25 * (
            s1_vals * s2_vals * s3_vals
            + s1_vals * s2_vals.conj() * s3_vals
            + s1_vals * s2_vals * s3_vals.conj()
            + s1_vals * s2_vals.conj() * s3_vals.conj()
        )
        self.assertAllClose(sig_prod.complex_value(t_vals), expected)

        expected = sig1(t_vals) * sig2(t_vals) * sig3(t_vals)
        self.assertAllClose(sig_prod(t_vals), expected)


class TestSignalAdditionJax(TestSignalAddition, TestJaxBase):
    """Jax version of TestSignalAddition."""


class TestSignalMultiplicationJax(TestSignalMultiplication, TestJaxBase):
    """Jax version of TestSignalMultiplication."""


class TestSignalAlgebraJaxTransformations(QiskitDynamicsTestCase, TestJaxBase):
    """Test cases for jax transformations through signal algebraic operations."""

    def setUp(self):
        self.signal = Signal(lambda t: t**2, carrier_freq=3.0)
        self.constant = Signal(3 * np.pi)
        self.discrete_signal = DiscreteSignal(
            dt=0.5, samples=jnp.ones(20, dtype=complex), carrier_freq=2.0
        )
        self.signal_sum = self.signal + self.discrete_signal
        self.discrete_signal_sum = DiscreteSignalSum.from_SignalSum(
            self.signal_sum, dt=0.5, n_samples=20
        )

    def test_jit_sum(self):
        """Test jitting a function that involves constructing a SignalSum."""

        t_vals = np.array([1.0, 3.0, 0.1232])
        self._test_jit_sum_eval(self.signal, self.signal, t_vals)
        self._test_jit_sum_eval(self.constant, self.signal, t_vals)
        self._test_jit_sum_eval(self.signal, self.signal, t_vals)
        self._test_jit_sum_eval(self.discrete_signal, self.signal, t_vals)
        self._test_jit_sum_eval(self.discrete_signal, self.discrete_signal, t_vals)
        self._test_jit_sum_eval(self.discrete_signal_sum, self.signal_sum, t_vals)

    def test_jit_grad_sum(self):
        """Test that the jit of a grad of a sum can be computed without error."""

        t_vals = 0.233234
        self._test_grad_jit_sum_eval(self.signal, self.signal, t_vals)
        self._test_grad_jit_sum_eval(self.constant, self.signal, t_vals)
        self._test_grad_jit_sum_eval(self.signal, self.signal, t_vals)
        self._test_grad_jit_sum_eval(self.discrete_signal, self.signal, t_vals)
        self._test_grad_jit_sum_eval(self.discrete_signal, self.discrete_signal, t_vals)
        self._test_grad_jit_sum_eval(self.discrete_signal_sum, self.signal_sum, t_vals)

    def test_jit_prod(self):
        """Test jitting a function that involves multiplying signals."""

        t_vals = np.array([1.0, 3.0, 0.1232])
        self._test_jit_prod_eval(self.signal, self.signal, t_vals)
        self._test_jit_prod_eval(self.constant, self.signal, t_vals)
        self._test_jit_prod_eval(self.signal, self.signal, t_vals)
        self._test_jit_prod_eval(self.discrete_signal, self.signal, t_vals)
        self._test_jit_prod_eval(self.discrete_signal, self.discrete_signal, t_vals)
        self._test_jit_prod_eval(self.discrete_signal_sum, self.signal_sum, t_vals)

    def test_jit_grad_prod(self):
        """Test jitting a function that involves constructing a SignalSum."""

        t_vals = 0.233234
        self._test_grad_jit_prod_eval(self.signal, self.signal, t_vals)
        self._test_grad_jit_prod_eval(self.constant, self.signal, t_vals)
        self._test_grad_jit_prod_eval(self.signal, self.signal, t_vals)
        self._test_grad_jit_prod_eval(self.discrete_signal, self.signal, t_vals)
        self._test_grad_jit_prod_eval(self.discrete_signal, self.discrete_signal, t_vals)
        self._test_grad_jit_prod_eval(self.discrete_signal_sum, self.signal_sum, t_vals)

    def _test_jit_sum_eval(self, sig1, sig2, t_vals):
        """jit compilation and evaluation of added signals."""

        def eval_func(t):
            sig_sum = sig1 + sig2
            return sig_sum(t).data

        jit_eval_func = jit(eval_func)
        self.assertAllClose(jit_eval_func(t_vals), eval_func(t_vals))

    def _test_grad_jit_sum_eval(self, sig1, sig2, t):
        """Verify that the grad of the sum of two signals can be compiled."""

        def eval_func(t):
            sig_sum = sig1 + sig2
            return sig_sum(t).data

        jit_eval_func = jit(grad(eval_func))
        jit_eval_func(t)

    def _test_jit_prod_eval(self, sig1, sig2, t_vals):
        """jit compilation and evaluation of added signals."""

        def eval_func(t):
            sig_sum = sig1 * sig2
            return sig_sum(t).data

        jit_eval_func = jit(eval_func)
        self.assertAllClose(jit_eval_func(t_vals), eval_func(t_vals))

    def _test_grad_jit_prod_eval(self, sig1, sig2, t):
        """Verify that the grad of the product of two signals can be compiled."""

        def eval_func(t):
            sig_sum = sig1 * sig2
            return sig_sum(t).data

        jit_eval_func = jit(grad(eval_func))
        jit_eval_func(t)
