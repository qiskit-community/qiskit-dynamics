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
"""tests for operator_models.py"""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.pulse_new.models.operator_models import OperatorModel
from qiskit.providers.aer.pulse_new.models.signals import Constant, Signal, VectorSignal


class TestOperatorModel(unittest.TestCase):
    """Tests for OperatorModel."""

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

        # define a basic model
        w = 2.
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2,
                     -1j * 2 * np.pi *  r * self.X / 2]
        signals = [Constant(w), Signal(1., w)]

        self.w = 2
        self.r = r
        self.basic_model = OperatorModel(operators=operators, signals=signals)

    def test_frame_operator_errors(self):
        """Check different modes of error raising for frame setting."""

        # 1d array
        try:
            self.basic_model.frame = np.array([1., 1.])
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

        # 2d array
        try:
            self.basic_model.frame = np.array([[1., 0.], [0., 1.]])
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

        # Operator
        try:
            self.basic_model.frame = self.Z
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

    def test_diag_frame_operator_basic_model(self):
        """Test setting a diagonal frame operator for the internally
        set up basic model.
        """

        self._basic_frame_evaluate_test(np.array([1j, -1j]), 1.123)
        self._basic_frame_evaluate_test(np.array([1j, -1j]), np.pi)


    def test_non_diag_frame_operator_basic_model(self):
        """Test setting a non-diagonal frame operator for the internally
        set up basic model.
        """
        self._basic_frame_evaluate_test(-1j * (self.Y + self.Z), 1.123)
        self._basic_frame_evaluate_test(-1j * (self.Y - self.Z), np.pi)


    def _basic_frame_evaluate_test(self, frame_operator, t):
        """Routine for testing setting of valid frame operators using the
        basic_model.
        """

        self.basic_model.frame = frame_operator

        # convert to 2d array
        if isinstance(frame_operator, Operator):
            frame_operator = frame_operator.data
        if isinstance(frame_operator, np.ndarray) and frame_operator.ndim == 1:
            frame_operator = np.diag(frame_operator)

        value = self.basic_model.evaluate(t)

        i2pi = -1j * 2 * np.pi

        U = expm(- frame_operator * t)

        # drive coefficient
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)

        # manually evaluate frame
        expected = (i2pi * self.w * U @ self.Z.data @ U.conj().transpose() / 2 +
                    d_coeff * i2pi * U @ self.X.data @ U.conj().transpose() / 2 -
                    frame_operator)

        self.assertAlmostEqual(value, expected)

    def test_evaluate_no_frame_basic_model(self):
        """Test evaluation without a frame in the basic model."""

        t = 3.21412
        value = self.basic_model.evaluate(t)
        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected =  (i2pi * self.w * self.Z.data / 2 +
                     i2pi * d_coeff * self.X.data / 2)

        self.assertAlmostEqual(value, expected)

    def test_evaluate_in_frame_basis_basic_model(self):
        """Test evaluation in frame basis in the basic_model."""

        frame_op = -1j * (self.X + 0.2 * self.Y + 0.1 * self.Z).data

        # enter the frame given by the -1j * X
        self.basic_model.frame = frame_op

        # get the frame basis that is used in model
        _, U = np.linalg.eigh(1j * frame_op)

        t = 3.21412
        value = self.basic_model.evaluate(t, in_frame_basis=True)

        # compose the frame basis transformation with the exponential
        # frame rotation (this will be multiplied on the right)
        U = expm(frame_op * t) @ U
        Uadj = U.conj().transpose()

        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = Uadj @ (i2pi * self.w * self.Z.data / 2 +
                           i2pi * d_coeff * self.X.data / 2 -
                           frame_op) @ U

        self.assertAlmostEqual(value, expected)


    def test_evaluate_pseudorandom(self):
        """Test evaluate with pseudorandom inputs."""

        rng = np.random.default_rng(30493)
        num_terms = 3
        dim = 5
        b = 1. # bound on size of random terms
        rand_op = (rng.uniform(low=-b, high=b, size=(dim,dim)) +
                   1j*rng.uniform(low=-b, high=b, size=(dim,dim)))
        frame_op = rand_op - rand_op.conj().transpose()
        rand_operators = (rng.uniform(low=-b,high=b, size=(num_terms, dim, dim)) +
                            1j * rng.uniform(low=-b,high=b, size=(num_terms, dim, dim)))

        rand_coeffs = (rng.uniform(low=-b,high=b, size=(num_terms)) +
                            1j * rng.uniform(low=-b,high=b, size=(num_terms)))
        rand_carriers = rng.uniform(low=-b,high=b, size=(num_terms))

        self._test_evaluate(frame_op, rand_operators, rand_coeffs, rand_carriers)

        rng = np.random.default_rng(94818)
        num_terms = 5
        dim = 10
        b = 1. # bound on size of random terms
        rand_op = (rng.uniform(low=-b, high=b, size=(dim,dim)) +
                   1j*rng.uniform(low=-b, high=b, size=(dim,dim)))
        frame_op = rand_op - rand_op.conj().transpose()
        rand_operators = (rng.uniform(low=-b,high=b, size=(num_terms, dim, dim)) +
                            1j * rng.uniform(low=-b,high=b, size=(num_terms, dim, dim)))

        rand_coeffs = (rng.uniform(low=-b,high=b, size=(num_terms)) +
                            1j * rng.uniform(low=-b,high=b, size=(num_terms)))
        rand_carriers = rng.uniform(low=-b,high=b, size=(num_terms))

    def _test_evaluate(self, frame_op, operators, coefficients, carriers):

        vec_sig = VectorSignal(lambda t: coefficients, carriers)
        model = OperatorModel(operators, vec_sig, frame=frame_op)

        value = model.evaluate(1.)
        coeffs = np.real(coefficients * np.exp(1j * 2 * np.pi * carriers * 1.))
        expected = expm(-frame_op) @ np.tensordot(coeffs, operators, axes=1) @ expm(frame_op) - frame_op

        self.assertAlmostEqual(value, expected)

    def test_lmult_rmult_no_frame_basic_model(self):
        """Test evaluation with no frame in the basic model."""

        y0 = np.array([[1., 2.], [0., 4.]])
        t = 3.21412
        value = self.basic_model.lmult(t, y0)
        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        model_expected =  (i2pi * self.w * self.Z.data / 2 +
                           i2pi * d_coeff * self.X.data / 2)

        self.assertAlmostEqual(self.basic_model.lmult(t, y0),
                               model_expected @ y0)
        self.assertAlmostEqual(self.basic_model.rmult(t, y0),
                               y0 @ model_expected)

    def test_signal_setting(self):
        """Test updating the signals."""
        signals = VectorSignal(lambda t: np.array([2 * t, t**2]),
                               np.array([1., 2.]),
                               np.array([0., 0.]))

        self.basic_model.signals = signals


        t = 0.1
        value = self.basic_model.evaluate(t)
        i2pi = -1j * 2 * np.pi
        Z_coeff = (2 * t) * np.cos(2 * np.pi * 1 * t)
        X_coeff = self.r * (t**2) * np.cos(2 * np.pi * 2 * t)
        expected =  (i2pi * Z_coeff * self.Z.data / 2 +
                     i2pi * X_coeff * self.X.data / 2)
        self.assertAlmostEqual(value, expected)

    def test_signal_setting_None(self):
        """Test setting signals to None"""

        self.basic_model.signals = None
        self.assertTrue(self.basic_model.signals is None)

    def test_signal_setting_incorrect_length(self):
        """Test error being raised if signals is the wrong length."""

        try:
            self.basic_model.signals = [Constant(1.)]
        except Exception as e:
            self.assertTrue('same length' in str(e))

    def test_signal_mapping(self):
        """Test behaviour of signal mapping."""

        def sig_map(slope):
            return VectorSignal(lambda t: np.array([slope * t, slope**2 * t]),
                                carrier_freqs=np.array([1., self.w]))

        self.basic_model.signal_mapping = sig_map
        self.basic_model.signals = 2.

        output = self.basic_model.signals.envelope_value(2.)
        expected = np.array([4., 8.])

        self.assertAlmostEqual(output, expected)

        output = self.basic_model.evaluate(2.)
        drive_coeff = self.r * 8 * np.cos(2 * np.pi * self.w * 2.)
        expected = (-1j * 2 * np.pi * 4 * self.Z.data /2 +
                    -1j * 2 * np.pi * drive_coeff * self.X.data /2)

        self.assertAlmostEqual(output, expected)


    def test_drift(self):
        """Test drift evaluation."""

        self.assertAlmostEqual(self.basic_model.drift,
                               -1j * 2 * np.pi * self.w * self.Z.data / 2)

    def test_drift_error_in_frame(self):
        """Test raising of error if drift is requested in a frame."""
        self.basic_model.frame = self.basic_model.drift

        try:
            self.basic_model.drift
        except Exception as e:
            self.assertTrue('ill-defined' in str(e))

    def test_cutoff_freq(self):
        """Test evaluation with a cutoff frequency."""

        # enter frame of drift
        self.basic_model.frame = self.basic_model.drift

        # set cutoff freq to 2 * drive freq (standard RWA)
        self.basic_model.cutoff_freq = 2 * self.w

        # result should just be the X term halved
        eval_rwa = self.basic_model.evaluate(2.)
        expected = -1j * 2 * np.pi * (self.r / 2) * self.X.data / 2
        self.assertAlmostEqual(eval_rwa, expected)

        drive_func = lambda t: t**2 + t**3 * 1j
        self.basic_model.signals = [Constant(self.w),
                                    Signal(drive_func, self.w)]

        # result should now contain both X and Y terms halved
        t = 2.1231 * np.pi
        dRe = np.real(drive_func(t))
        dIm = np.imag(drive_func(t))
        eval_rwa = self.basic_model.evaluate(t)
        expected = (-1j * 2 * np.pi * (self.r / 2) * dRe * self.X.data / 2 +
                    -1j * 2 * np.pi * (self.r / 2) * dIm * self.Y.data / 2)
        self.assertAlmostEqual(eval_rwa, expected)


    def assertAlmostEqual(self, A, B, tol=1e-12):
        self.assertTrue(np.abs(A - B).max() < tol)
