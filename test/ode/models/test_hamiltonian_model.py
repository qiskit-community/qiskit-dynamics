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

"""tests for quantum_models.HamiltonianModel"""

import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info.operators import Operator
from qiskit_ode.models import HamiltonianModel
from qiskit_ode.signals import Constant, Signal, VectorSignal
from qiskit_ode.dispatch import Array
from ..common import QiskitOdeTestCase, TestJaxBase


class TestHamiltonianModel(QiskitOdeTestCase):
    """Tests for HamiltonianModel."""

    def setUp(self):
        self.X = Array(Operator.from_label('X').data)
        self.Y = Array(Operator.from_label('Y').data)
        self.Z = Array(Operator.from_label('Z').data)

        # define a basic hamiltonian
        w = 2.
        r = 0.5
        operators = [2 * np.pi * self.Z / 2,
                     2 * np.pi * r * self.X / 2]
        signals = [Constant(w), Signal(1., w)]

        self.w = w
        self.r = r
        self.basic_hamiltonian = HamiltonianModel(operators=operators,
                                                  signals=signals)

    def _basic_frame_evaluate_test(self, frame_operator, t):
        """Routine for testing setting of valid frame operators using
        basic_hamiltonian.

        Adapted from the version of this test in
        test_operator_models.py, but relative to the way HamiltonianModel
        modifies frame handling.

        Args:
            frame_operator (Array): now assumed to be a Hermitian operator H, with the
                            frame being entered being F=-1j * H
            t (float): time of frame transformation
        """

        self.basic_hamiltonian.frame = frame_operator

        # convert to 2d array
        if isinstance(frame_operator, Operator):
            frame_operator = Array(frame_operator.data)
        if isinstance(frame_operator, Array) and frame_operator.ndim == 1:
            frame_operator = np.diag(frame_operator)

        value = self.basic_hamiltonian.evaluate(t)

        twopi = 2 * np.pi

        # frame is F=-1j * H, and need to compute exp(-F * t)
        U = expm(1j * np.array(frame_operator) * t)

        # drive coefficient
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)

        # manually evaluate frame
        expected = (twopi * self.w * U @ self.Z.data @ U.conj().transpose() / 2 +
                    d_coeff * twopi * U @ self.X.data @ U.conj().transpose() / 2 -
                    frame_operator)

        self.assertAllClose(value, expected)

    def test_diag_frame_operator_basic_hamiltonian(self):
        """Test setting a diagonal frame operator for the internally
        set up basic hamiltonian.
        """

        self._basic_frame_evaluate_test(Array([1., -1.]), 1.123)
        self._basic_frame_evaluate_test(Array([1., -1.]), np.pi)

    def test_non_diag_frame_operator_basic_hamiltonian(self):
        """Test setting a non-diagonal frame operator for the internally
        set up basic model.
        """
        self._basic_frame_evaluate_test(self.Y + self.Z, 1.123)
        self._basic_frame_evaluate_test(self.Y - self.Z, np.pi)

    def test_evaluate_no_frame_basic_hamiltonian(self):
        """Test evaluation without a frame in the basic model."""

        t = 3.21412
        value = self.basic_hamiltonian.evaluate(t)
        twopi = 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = (twopi * self.w * self.Z.data / 2 +
                    twopi * d_coeff * self.X.data / 2)

        self.assertAllClose(value, expected)

    def test_evaluate_in_frame_basis_basic_hamiltonian(self):
        """Test evaluation in frame basis in the basic_model."""

        frame_op = (self.X + 0.2 * self.Y + 0.1 * self.Z).data

        # enter the frame given by the -1j * X
        self.basic_hamiltonian.frame = frame_op

        # get the frame basis used in model. Note that the Frame object
        # orders the basis according to the ordering of eigh
        _, U = np.linalg.eigh(frame_op)

        t = 3.21412
        value = self.basic_hamiltonian.evaluate(t, in_frame_basis=True)

        # compose the frame basis transformation with the exponential
        # frame rotation (this will be multiplied on the right)
        U = expm(-1j * frame_op * t) @ U
        Uadj = U.conj().transpose()

        twopi = 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = Uadj @ (twopi * self.w * self.Z.data / 2 +
                           twopi * d_coeff * self.X.data / 2 -
                           frame_op) @ U

        self.assertAllClose(value, expected)

    def test_evaluate_pseudorandom(self):
        """Test evaluate with pseudorandom inputs."""

        rng = np.random.default_rng(30493)
        num_terms = 3
        dim = 5
        b = 1.  # bound on size of random terms

        # random hermitian frame operator
        rand_op = (rng.uniform(low=-b, high=b, size=(dim, dim)) +
                   1j * rng.uniform(low=-b, high=b, size=(dim, dim)))
        frame_op = Array(rand_op + rand_op.conj().transpose())

        # random hermitian operators
        randoperators = (rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) +
                         1j * rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)))
        randoperators = Array(randoperators + randoperators.conj().transpose([0, 2, 1]))

        rand_coeffs = (rng.uniform(low=-b, high=b, size=(num_terms)) +
                       1j * rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_carriers = Array(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = Array(rng.uniform(low=-b, high=b, size=(num_terms)))

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

        rng = np.random.default_rng(94818)
        num_terms = 5
        dim = 10
        b = 1.  # bound on size of random terms

        # random hermitian frame operator
        rand_op = (rng.uniform(low=-b, high=b, size=(dim, dim)) +
                   1j * rng.uniform(low=-b, high=b, size=(dim, dim)))
        frame_op = Array(rand_op + rand_op.conj().transpose())

        # random hermitian operators
        randoperators = (rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) +
                         1j * rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)))
        randoperators = Array(randoperators + randoperators.conj().transpose([0, 2, 1]))

        rand_coeffs = (rng.uniform(low=-b, high=b, size=(num_terms)) +
                       1j * rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_carriers = Array(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = Array(rng.uniform(low=-b, high=b, size=(num_terms)))

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

    def _test_evaluate(self, frame_op, operators, coefficients, carriers, phases):

        vec_sig = VectorSignal(lambda t: coefficients, carriers, phases)
        model = HamiltonianModel(operators, vec_sig, frame=frame_op)

        value = model.evaluate(1.)
        coeffs = np.real(coefficients * np.exp(1j * 2 * np.pi * carriers * 1. + 1j * phases))
        expected = (expm(1j * np.array(frame_op)) @ np.tensordot(coeffs, operators, axes=1) @
                    expm(-1j * np.array(frame_op)) - frame_op)

        self.assertAllClose(value, expected)

    def test_cutoff_freq(self):
        """Test evaluation with a cutoff frequency."""

        # enter frame of drift
        self.basic_hamiltonian.frame = self.basic_hamiltonian.drift

        # set cutoff freq to 2 * drive freq (standard RWA)
        self.basic_hamiltonian.cutoff_freq = 2 * self.w

        # result should just be the X term halved
        eval_rwa = self.basic_hamiltonian.evaluate(2.)
        expected = 2 * np.pi * (self.r / 2) * self.X.data / 2
        self.assertAllClose(eval_rwa, expected)

        def drive_func(t):
            return t**2 + t**3 * 1j

        self.basic_hamiltonian.signals = [Constant(self.w),
                                          Signal(drive_func, self.w)]

        # result should now contain both X and Y terms halved
        t = 2.1231 * np.pi
        dRe = np.real(drive_func(t))
        dIm = np.imag(drive_func(t))
        eval_rwa = self.basic_hamiltonian.evaluate(t)
        expected = (2 * np.pi * (self.r / 2) * dRe * self.X.data / 2 +
                    2 * np.pi * (self.r / 2) * dIm * self.Y.data / 2)
        self.assertAllClose(eval_rwa, expected)


class TestHamiltonianModelJax(TestHamiltonianModel, TestJaxBase):
    """Jax version of TestHamiltonianModel tests.

    Note: This class has no body but contains tests due to inheritance.
    """
