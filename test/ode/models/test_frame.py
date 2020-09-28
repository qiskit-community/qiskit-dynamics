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
"""tests for frame.py"""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.providers.aer.pulse_new.models.frame import Frame
from qiskit.quantum_info.operators import Operator

class TestFrame(unittest.TestCase):
    """Tests for Frame."""

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

    def test_instantiation_errors(self):
        """Check different modes of error raising for frame setting."""

        # 1d array
        try:
            frame = Frame(np.array([1., 1.]))
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

        # 2d array
        try:
            frame = Frame(np.array([[1., 0.], [0., 1.]]))
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

        # Operator
        try:
            frame = Frame(self.Z)
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

    def test_state_into_frame_basis(self):
        """Test state_into_frame_basis."""

        rng = np.random.default_rng(10933)
        rand_op = (rng.uniform(low=-10, high=10, size=(6,6)) +
                   1j*rng.uniform(low=-10, high=10, size=(6,6)))

        frame_op = rand_op - rand_op.conj().transpose()
        frame = Frame(frame_op)

        _, U = np.linalg.eigh(1j * frame_op)

        t = 1312.132
        y0 = (rng.uniform(low=-10, high=10, size=(6,6)) +
              1j*rng.uniform(low=-10, high=10, size=(6,6)))

        val = frame.state_into_frame_basis(y0)
        expected = U.conj().transpose() @ y0
        self.assertAlmostEqual(val, expected)

        val = frame.state_out_of_frame_basis(y0)
        expected = U @ y0
        self.assertAlmostEqual(val, expected)

    def test_operator_into_frame_basis(self):
        """Test state_into_frame_basis."""

        rng = np.random.default_rng(98747)
        rand_op = (rng.uniform(low=-10, high=10, size=(10,10)) +
                   1j*rng.uniform(low=-10, high=10, size=(10,10)))

        frame_op = rand_op - rand_op.conj().transpose()
        frame = Frame(frame_op)

        _, U = np.linalg.eigh(1j * frame_op)
        Uadj = U.conj().transpose()

        t = 1312.132
        y0 = (rng.uniform(low=-10, high=10, size=(10,10)) +
              1j*rng.uniform(low=-10, high=10, size=(10,10)))

        val = frame.operator_into_frame_basis(y0)
        expected = U.conj().transpose() @ y0 @ U
        self.assertAlmostEqual(val, expected)

        val = frame.operator_out_of_frame_basis(y0)
        expected = U @ y0 @ Uadj
        self.assertAlmostEqual(val, expected)

    def test_state_transformations_no_frame(self):
        """Test frame transformations with no frame."""

        frame = Frame(np.zeros(2))

        t = 0.123
        y = np.array([1., 1j])
        out = frame.state_into_frame(t, y)
        self.assertAlmostEqual(out, y)
        out = frame.state_out_of_frame(t, y)
        self.assertAlmostEqual(out, y)

        t = 100.12498
        y = np.eye(2)
        out = frame.state_into_frame(t, y)
        self.assertAlmostEqual(out, y)
        out = frame.state_out_of_frame(t, y)
        self.assertAlmostEqual(out, y)

    def test_state_into_frame_2_level(self):
        """Test state_into_frame with a non-trival frame."""
        frame_op = -1j * np.pi * (self.X + 0.1 * self.Y + 12. * self.Z).data
        t = 1312.132
        y0 = np.array([[1., 2.], [3., 4.]])

        self._test_state_into_frame(t, frame_op, y0)
        self._test_state_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_into_frame(t, frame_op, y0, y_in_frame_basis=True,
                                                     return_in_frame_basis=True)

    def test_state_into_frame_pseudo_random(self):
        """Test state_into_frame with pseudo-random matrices."""
        rng = np.random.default_rng(30493)
        rand_op = (rng.uniform(low=-10, high=10, size=(5,5)) +
                   1j*rng.uniform(low=-10, high=10, size=(5,5)))

        frame_op = rand_op - rand_op.conj().transpose()

        t = 1312.132
        y0 = (rng.uniform(low=-10, high=10, size=(5,5)) +
              1j*rng.uniform(low=-10, high=10, size=(5,5)))

        self._test_state_into_frame(t, frame_op, y0)
        self._test_state_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_into_frame(t, frame_op, y0, y_in_frame_basis=True,
                                                     return_in_frame_basis=True)

    def _test_state_into_frame(self,
                               t,
                               frame_op,
                               y,
                               y_in_frame_basis=False,
                               return_in_frame_basis=False):

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals

        frame = Frame(frame_op)

        value = frame.state_into_frame(t, y,
                                       y_in_frame_basis,
                                       return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = U.conj().transpose() @ expected

        expected = np.diag(np.exp(-t * evals)) @ expected

        if not return_in_frame_basis:
            expected = U @ expected

        self.assertAlmostEqual(value, expected, tol=1e-10)

    def test_state_out_of_frame_2_level(self):
        """Test state_out_of_frame with a non-trival frame."""
        frame_op = -1j * np.pi * (3.1 * self.X + 1.1 * self.Y +
                                  12. * self.Z).data
        t = 122.132
        y0 = np.array([[1., 2.], [3., 4.]])
        self._test_state_out_of_frame(t, frame_op, y0)
        self._test_state_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_out_of_frame(t, frame_op, y0, y_in_frame_basis=True,
                                                     return_in_frame_basis=True)

    def test_state_out_of_frame_pseudo_random(self):
        """Test state_out_of_frame with pseudo-random matrices."""
        rng = np.random.default_rng(1382)
        rand_op = (rng.uniform(low=-10, high=10, size=(6,6)) +
                   1j*rng.uniform(low=-10, high=10, size=(6,6)))

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = (rng.uniform(low=-10, high=10, size=(6,6)) +
              1j*rng.uniform(low=-10, high=10, size=(6,6)))

        self._test_state_out_of_frame(t, frame_op, y0)
        self._test_state_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_out_of_frame(t, frame_op, y0, y_in_frame_basis=True,
                                                     return_in_frame_basis=True)

    def _test_state_out_of_frame(self,
                                 t,
                                 frame_op,
                                 y,
                                 y_in_frame_basis=False,
                                 return_in_frame_basis=False):

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals

        frame = Frame(frame_op)

        value = frame.state_out_of_frame(t, y,
                                         y_in_frame_basis,
                                         return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = U.conj().transpose() @ expected

        expected = np.diag(np.exp(t * evals)) @ expected

        if not return_in_frame_basis:
            expected = U @ expected

        self.assertAlmostEqual(value, expected, tol=1e-10)

    def test_operator_into_frame(self):
        """Test operator_into_frame."""
        rng = np.random.default_rng(94994)
        rand_op = (rng.uniform(low=-10, high=10, size=(6,6)) +
                   1j*rng.uniform(low=-10, high=10, size=(6,6)))

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = (rng.uniform(low=-10, high=10, size=(6,6)) +
              1j*rng.uniform(low=-10, high=10, size=(6,6)))

        self._test_operator_into_frame(t, frame_op, y0)
        self._test_operator_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_operator_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_operator_into_frame(t, frame_op, y0, y_in_frame_basis=True,
                                                     return_in_frame_basis=True)


    def _test_operator_into_frame(self,
                                  t,
                                  frame_op,
                                  y,
                                  y_in_frame_basis=False,
                                  return_in_frame_basis=False):

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals
        Uadj = U.conj().transpose()

        frame = Frame(frame_op)

        value = frame.operator_into_frame(t, y,
                                          y_in_frame_basis,
                                          return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = np.diag(np.exp(-t * evals)) @ expected @ np.diag(np.exp(t * evals))

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAlmostEqual(value, expected, tol=1e-10)

    def test_operator_into_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(37164093)
        rand_op = (rng.uniform(low=-10, high=10, size=(6,6)) +
                   1j*rng.uniform(low=-10, high=10, size=(6,6)))

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = (rng.uniform(low=-10, high=10, size=(6,6)) +
              1j*rng.uniform(low=-10, high=10, size=(6,6)))

        self._test_operator_out_of_frame(t, frame_op, y0)
        self._test_operator_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_operator_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_operator_out_of_frame(t, frame_op, y0, y_in_frame_basis=True,
                                                         return_in_frame_basis=True)

    def _test_operator_out_of_frame(self,
                                    t,
                                    frame_op,
                                    y,
                                    y_in_frame_basis=False,
                                    return_in_frame_basis=False):

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals
        Uadj = U.conj().transpose()

        frame = Frame(frame_op)

        value = frame.operator_out_of_frame(t, y,
                                            y_in_frame_basis,
                                            return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = np.diag(np.exp(t * evals)) @ expected @ np.diag(np.exp(-t * evals))

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAlmostEqual(value, expected, tol=1e-10)

    def test_generator_into_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(111)
        rand_op = (rng.uniform(low=-10, high=10, size=(6,6)) +
                   1j*rng.uniform(low=-10, high=10, size=(6,6)))

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = (rng.uniform(low=-10, high=10, size=(6,6)) +
              1j*rng.uniform(low=-10, high=10, size=(6,6)))

        self._test_generator_into_frame(t, frame_op, y0)
        self._test_generator_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_generator_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_generator_into_frame(t, frame_op, y0, y_in_frame_basis=True,
                                                         return_in_frame_basis=True)

    def _test_generator_into_frame(self,
                                   t,
                                   frame_op,
                                   y,
                                   y_in_frame_basis=False,
                                   return_in_frame_basis=False):
        """Helper function for testing generator_into_frame."""
        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals
        Uadj = U.conj().transpose()

        frame = Frame(frame_op)

        value = frame.generator_into_frame(t, y,
                                            y_in_frame_basis,
                                            return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = np.diag(np.exp(-t * evals)) @ expected @ np.diag(np.exp(t * evals))
        expected = expected - np.diag(evals)

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAlmostEqual(value, expected, tol=1e-10)

    def test_generator_out_of_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(111)
        rand_op = (rng.uniform(low=-10, high=10, size=(6,6)) +
                   1j*rng.uniform(low=-10, high=10, size=(6,6)))

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = (rng.uniform(low=-10, high=10, size=(6,6)) +
              1j*rng.uniform(low=-10, high=10, size=(6,6)))

        self._test_generator_out_of_frame(t, frame_op, y0)
        self._test_generator_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_generator_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_generator_out_of_frame(t, frame_op, y0, y_in_frame_basis=True,
                                                         return_in_frame_basis=True)

    def _test_generator_out_of_frame(self,
                                     t,
                                     frame_op,
                                     y,
                                     y_in_frame_basis=False,
                                     return_in_frame_basis=False):
        """Helper function for testing generator_into_frame."""
        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals
        Uadj = U.conj().transpose()

        frame = Frame(frame_op)

        value = frame.generator_out_of_frame(t, y,
                                            y_in_frame_basis,
                                            return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = np.diag(np.exp(t * evals)) @ expected @ np.diag(np.exp(-t * evals))
        expected = expected + np.diag(evals)

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAlmostEqual(value, expected, tol=1e-10)

    def test_operators_into_frame_basis_with_cutoff_no_cutoff(self):
        """Test function for construction of operators with cutoff,
        without a cutoff.
        """

        # no cutoff with already diagonal frame
        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = np.array([self.X.data, self.Y.data, self.Z.data])
        carrier_freqs = np.array([1., 2., 3.])

        frame = Frame(frame_op)

        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       carrier_freqs=carrier_freqs)

        self.assertAlmostEqual(ops_w_cutoff, operators)
        self.assertAlmostEqual(ops_w_conj_cutoff, operators)

        # same test but with frame given as a 2d array
        # in this case diagonalization will occur, causing eigenvalues to
        # be sorted in ascending order. This will flip Y and Z
        frame_op = -1j * np.pi * np.array([[1., 0], [0, -1.]])
        carrier_freqs = np.array([1., 2., 3.])

        frame = Frame(frame_op)

        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       carrier_freqs=carrier_freqs)


        expected_ops = np.array([self.X.data, -self.Y.data, -self.Z.data])

        self.assertAlmostEqual(ops_w_cutoff, expected_ops)
        self.assertAlmostEqual(ops_w_conj_cutoff, expected_ops)

    def test_operators_into_frame_basis_with_cutoff(self):
        """Test function for construction of operators with cutoff."""

        # cutoff test
        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = np.array([self.X.data, self.Y.data, self.Z.data])
        carrier_freqs = np.array([1., 2., 3.])
        cutoff_freq = 3.

        frame = Frame(frame_op)

        cutoff_mat = np.array([[[1, 1],
                                [1, 1]],
                               [[1, 0],
                                [1, 1]],
                               [[0, 0],
                                [1, 0]]
                               ])

        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       cutoff_freq=cutoff_freq,
                                                                                       carrier_freqs=carrier_freqs)

        ops_w_cutoff_expect = cutoff_mat * operators
        ops_w_conj_cutoff_expect = cutoff_mat.transpose([0, 2, 1]) * operators

        self.assertAlmostEqual(ops_w_cutoff, ops_w_cutoff_expect)
        self.assertAlmostEqual(ops_w_conj_cutoff, ops_w_conj_cutoff_expect)

        # same test with lower cutoff
        cutoff_freq = 2.

        cutoff_mat = np.array([[[1, 0],
                                [1, 1]],
                               [[0, 0],
                                [1, 0]],
                               [[0, 0],
                                [0, 0]]
                               ])

        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       cutoff_freq=cutoff_freq,
                                                                                       carrier_freqs=carrier_freqs)

        ops_w_cutoff_expect = cutoff_mat * operators
        ops_w_conj_cutoff_expect = cutoff_mat.transpose([0, 2, 1]) * operators

        self.assertAlmostEqual(ops_w_cutoff, ops_w_cutoff_expect)
        self.assertAlmostEqual(ops_w_conj_cutoff, ops_w_conj_cutoff_expect)


    def assertAlmostEqual(self, A, B, tol=1e-14):
        diff = np.abs(A - B).max()
        self.assertTrue(diff < tol)
