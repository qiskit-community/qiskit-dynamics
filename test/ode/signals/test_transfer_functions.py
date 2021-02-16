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
Tests for signal transfer functions.
"""

import numpy as np
from qiskit_ode.signals import Convolution, PiecewiseConstant, Sampler, IQMixer, Signal

from ..common import QiskitOdeTestCase


class TestTransferFunctions(QiskitOdeTestCase):
    """Tests for transfer functions."""

    def setUp(self):
        pass

    def test_convolution(self):
        """Test of convolution function."""
        ts = np.linspace(0, 100, 200)

        def gaussian(t):
            sigma = 4
            return (2. * ts[1] / np.sqrt(2. * np.pi * sigma ** 2) *
                    np.exp(-t ** 2 / (2 * sigma ** 2)))

        # Test the simple convolution of a signal without a carrier
        convolve = Convolution(gaussian)

        samples = [0. if t < 20. or t > 80. else 1. for t in ts]  # Defines a square pulse.
        piecewise_const = PiecewiseConstant(dt=ts[1] - ts[0],
                                            samples=samples,
                                            carrier_freq=0.0,
                                            start_time=0)

        self.assertEqual(piecewise_const.duration, len(ts))
        self.assertEqual(piecewise_const.value(21.0), 1.0)
        self.assertEqual(piecewise_const.value(81.0), 0.0)

        convolved = convolve(piecewise_const)

        self.assertLess(convolved.value(21.0), 1.0)
        self.assertGreater(convolved.value(81.0), 0.0)

        if isinstance(convolved, PiecewiseConstant):
            self.assertEqual(convolved.duration, 2*len(ts)-1)
        else:
            self.fail()

        # Test that the normalization happens properly
        def non_normalized_gaussian(t):
            sigma = 4
            return 20. * np.exp(-t ** 2 / (2 * sigma ** 2))

        convolve = Convolution(non_normalized_gaussian)

        convolved2 = convolve(piecewise_const)

        self.assertAlmostEqual(convolved.value(15.0), convolved2.value(15.0), places=6)
        self.assertAlmostEqual(convolved.value(20.0), convolved2.value(20.0), places=6)
        self.assertAlmostEqual(convolved.value(25.0), convolved2.value(25.0), places=6)
        self.assertAlmostEqual(convolved.value(30.0), convolved2.value(30.0), places=6)

    def test_sampler(self):
        """Test the sampler."""
        dt = 0.5
        signal = PiecewiseConstant(dt=dt, samples=[0.3, 0.5], carrier_freq=0.2)
        sampler = Sampler(dt/2, 4)

        new_signal = sampler(signal)

        self.assertTrue(np.allclose(new_signal.samples, [0.3, 0.3, 0.5, 0.5]))

        signal = PiecewiseConstant(dt=dt, samples=[0.3, 0.4, 0.6, 0.8], carrier_freq=0.2)
        sampler = Sampler(2*dt, 2)

        new_signal = sampler(signal)
        self.assertTrue(np.allclose(new_signal.samples, [0.3, 0.6]))

    def test_iq_mixer(self):
        """Test the IQ mixer by checking we can up-convert to 5GHz"""
        dt = 0.25

        # Sideband at 0.1 GHz
        in_phase = PiecewiseConstant(dt, [1.0] * 200, carrier_freq=0.1, phase=0)
        quadrature = PiecewiseConstant(dt, [1.0] * 200, carrier_freq=0.1, phase=np.pi / 2)

        sampler = Sampler(dt/25, 5000)
        in_phase = sampler(in_phase)
        quadrature = sampler(quadrature)

        mixer = IQMixer(4.9)  # LO at 4.9 GHz
        rf = mixer(in_phase, quadrature)

        # Check max amplitude of the fourier transform
        import scipy.fftpack

        rf_samples = rf.to_pwc(dt/25, 5000).samples
        yf = scipy.fftpack.fft(np.real(rf_samples))
        xf = np.linspace(0.0, 1.0 / (2.0 * dt / 25), len(rf_samples) // 2)

        self.assertAlmostEqual(5.0, xf[np.argmax(np.abs(yf[:len(rf_samples) // 2]))], 2)

        # Test the same using Signals and not PieceWiseConstant
        in_phase = Signal(1.0, carrier_freq=0.1, phase=0)
        quadrature = Signal(1.0, carrier_freq=0.1, phase=-np.pi / 2)
        rf = mixer(in_phase, quadrature)
        dt = 0.01
        samples = rf.to_pwc(dt, 1000).samples

        yf = scipy.fftpack.fft(np.real(samples))
        xf = np.linspace(0.0, 1.0 / (2.0 * dt), len(samples) // 2)

        self.assertAlmostEqual(4.8, xf[np.argmax(np.abs(yf[:len(samples) // 2]))], 1)
