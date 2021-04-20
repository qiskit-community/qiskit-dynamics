# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
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
Tests for signals.
"""

import numpy as np

from qiskit_ode.signals import Constant, DiscreteSignal, Signal
from qiskit_ode.signals.signals import SignalSum
from qiskit_ode.dispatch import Array

from ..common import QiskitOdeTestCase, TestJaxBase

try:
    from jax import jit, grad
except ImportError:
    pass


class TestSignals(QiskitOdeTestCase):
    """Tests for signals."""

    def setUp(self):
        pass

    def test_constant(self):
        """Test Constant signal"""

        constant = Constant(0.5)

        self.assertEqual(constant.envelope(0.0), 0.5)
        self.assertEqual(constant.envelope(10.0), 0.5)
        self.assertEqual(constant(0.0), 0.5)
        self.assertEqual(constant(10.0), 0.5)

    def test_signal(self):
        """Test Signal."""

        # Signal with constant amplitude
        signal = Signal(0.25, carrier_freq=0.3)
        self.assertEqual(signal.envelope(0.0), 0.25)
        self.assertEqual(signal.envelope(1.23), 0.25)
        self.assertEqual(signal(0.0), 0.25)
        self.assertEqual(signal(1.0), 0.25 * np.cos(0.3 * 2.0 * np.pi))

        signal = Signal(0.25, carrier_freq=0.3, phase=0.5)
        self.assertEqual(signal(1.0), 0.25 * np.cos(0.3 * 2.0 * np.pi + 0.5))

        # Signal with parabolic amplitude
        signal = Signal(lambda t: 2.0 * t ** 2, carrier_freq=0.1)
        self.assertEqual(signal.envelope(0.0), 0.0)
        self.assertEqual(signal.envelope(3.0), 18.0)
        self.assertEqual(signal(0.0), 0.0)
        self.assertEqual(signal(2.0), 8.0 * np.cos(0.1 * 2.0 * np.pi * 2.0))

        signal = Signal(lambda t: 2.0 * t ** 2, carrier_freq=0.1, phase=-0.1)
        self.assertEqual(signal(2.0), 8.0 * np.cos(0.1 * 2.0 * np.pi * 2.0 - 0.1))

    def test_piecewise_constant(self):
        """Test PWC signal."""

        dt = 1.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.5
        piecewise_const = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        self.assertEqual(piecewise_const.envelope(0.0), 0.0)
        self.assertEqual(piecewise_const.envelope(2.0), 1.0)
        self.assertEqual(piecewise_const(0.0), 0.0)
        self.assertEqual(piecewise_const(3.0), 2.0 * np.cos(0.5 * 2.0 * np.pi * 3.0))

        piecewise_const = DiscreteSignal(
            dt=dt, samples=samples, carrier_freq=carrier_freq, phase=0.5
        )
        self.assertEqual(piecewise_const(3.0), 2.0 * np.cos(0.5 * 2.0 * np.pi * 3.0 + 0.5))

    def test_multiplication(self):
        """Tests the multiplication of signals."""

        # Test Constant
        const1 = Constant(0.3)
        const2 = Constant(0.5)
        self.assertTrue(isinstance(const1 * const2, Signal))
        self.assertEqual((const1 * const2)(0.0), 0.15)
        self.assertEqual((const1 * const2)(10.0), 0.15)

        # Test Signal
        signal1 = Signal(3.0, carrier_freq=0.1)
        signal2 = Signal(lambda t: 2.0 * t ** 2, carrier_freq=0.1)
        self.assertTrue(isinstance(const1 * signal1, Signal))
        self.assertTrue(isinstance(signal1 * const1, Signal))
        self.assertTrue(isinstance(signal1 * signal2, Signal))
        self.assertAllClose((signal1 * signal2).carrier_freq, Array([0.2, 0.0]))
        self.assertEqual((signal1 * const1).carrier_freq, 0.1)
        self.assertAllClose((signal1 * signal2).envelope(0.0), Array([0.0, 0.0]))
        self.assertAllClose((signal1 * signal2).envelope(3.0), 0.5 * Array([3.0 * 18.0, 3.0 * 18.0]))
        self.assertEqual((signal1 * signal2)(0.0), 0.0)
        self.assertAllClose((signal1 * signal2)(2.0), signal1(2.0) * signal2(2.0))

        # Test piecewise constant
        dt = 1.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.5
        pwc1 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        dt = 2.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.1
        pwc2 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        # Test types
        self.assertTrue(len(const1 * pwc1) == 1)
        self.assertTrue(isinstance((const1 * pwc1)[0], Signal))
        self.assertTrue(isinstance(signal1 * pwc1, SignalSum))
        self.assertTrue(isinstance(pwc1 * pwc2, SignalSum))
        self.assertTrue(isinstance((pwc1 * pwc2)[0], Signal))
        self.assertTrue(isinstance((pwc1 * pwc2)[1], Signal))
        self.assertTrue(len(pwc1 * const1) == 1)
        self.assertTrue(isinstance((pwc1 * const1)[0], DiscreteSignal))
        self.assertTrue(isinstance(pwc1 * signal1, SignalSum))
        self.assertTrue(isinstance((pwc1 * signal1)[0], Signal))

        # Test values
        self.assertAllClose((pwc1 * pwc2).carrier_freq, Array([0.5 + 0.1, 0.5 - 0.1]))
        self.assertAllClose((pwc1 * pwc2).envelope(0.0), Array([0.0, 0.0]))
        self.assertAllClose((pwc1 * pwc2).envelope(4.0), 0.5 * Array([1.0, 1.0]))
        self.assertEqual((pwc1 * pwc2)(0.0), 0.0)
        self.assertAllClose((pwc1 * pwc2)(4.0), 1.0 * np.cos(0.6 * 2.0 * np.pi * 4.0))

        # Test phase
        pwc2 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq, phase=0.5)
        self.assertEqual((pwc1 * pwc2)(4.0), 1.0 * np.cos(0.6 * 2.0 * np.pi * 4.0 + 0.5))

    def test_addition(self):
        """Tests the multiplication of signals."""

        # Test Constant
        const1 = Constant(0.3)
        const2 = Constant(0.5)
        self.assertTrue(isinstance(const1 + const2, Signal))
        self.assertEqual((const1 + const2)(0.0), 0.8)

        # Test Signal
        signal1 = Signal(3.0, carrier_freq=0.1)
        signal2 = Signal(lambda t: 2.0 * t ** 2, carrier_freq=0.1)
        self.assertTrue(isinstance(const1 + signal1, Signal))
        self.assertTrue(isinstance(signal1 + const1, Signal))
        self.assertTrue(isinstance(signal1 + signal2, Signal))
        self.assertAllClose((signal1 + signal2).carrier_freq, Array([0.1, 0.1]))
        self.assertAllClose((signal1 + const1).carrier_freq, Array([0.1, 0.0]))
        self.assertAllClose((signal1 + signal2).envelope(0.0), Array([3.0, 0.0]))
        expected = Array([3.0, 2.0 * (3.0)**2])
        self.assertAllClose((signal1 + signal2).envelope(3.0), expected)
        self.assertEqual((signal1 + signal2)(0.0), 3.0)
        self.assertEqual((signal1 + signal2)(2.0), 11.0 * np.cos(0.1 * 2.0 * np.pi * 2.0))

        # Test piecewise constant
        dt = 1.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.5
        pwc1 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        dt = 1.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.1
        pwc2 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        # Test types
        self.assertTrue(isinstance(const1 + pwc1, Signal))
        self.assertTrue(isinstance(signal1 + pwc1, Signal))
        self.assertTrue(isinstance(pwc1 + pwc2, Signal))
        self.assertTrue(isinstance(pwc1 + const1, Signal))
        self.assertTrue(isinstance(pwc1 + signal1, Signal))

        # Test values
        self.assertAllClose((pwc1 + pwc2).carrier_freq, Array([0.5, 0.1]))

        self.assertAllClose((pwc1 + pwc2).envelope(0.0), Array([0.0, 0.0]))
        self.assertAllClose((pwc1 + pwc2).envelope(4.0), Array([1.0, 1.0]))
        self.assertEqual((pwc1 + pwc2)(0.0), 0.0)
        expected = 1.0 * np.cos(0.5 * 2.0 * np.pi * 4.0) + 1.0 * np.cos(0.1 * 2.0 * np.pi * 4.0)
        self.assertAlmostEqual((pwc1 + pwc2)(4.0), expected, places=8)

        # Test phase
        pwc2 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq, phase=0.5)
        self.assertAllClose((pwc1 + pwc2).envelope(4.0), Array([1.0, 1.0]))


class TestSignalsJax(QiskitOdeTestCase, TestJaxBase):
    """Tests with some JAX functionality."""

    def test_jit_DiscreteSignal(self):
        """Verify that jit works through DiscreteSignal."""

        test_sig = DiscreteSignal(dt=1.0, samples=Array([1.0, 2.0, 3.0]))

        jit_eval = jit(lambda t: test_sig(t).data)

        val1 = jit_eval(0.5)
        expected = 1.0
        self.assertEqual(val1, expected)

        val2 = jit_eval(2.41)
        expected = 3.0
        self.assertEqual(val2, expected)

    def test_grad_DiscreteSignal(self):
        """Verify that grad works through DiscreteSignal."""

        def test_func(val):
            sig = DiscreteSignal(dt=1.0, samples=val * Array([1.0, 2.0, 3.0]))
            return np.real(sig(1.5)).data

        grad_func = grad(test_func)

        output = grad_func(3.0)
        expected = 2.0

        self.assertEqual(output, expected)
