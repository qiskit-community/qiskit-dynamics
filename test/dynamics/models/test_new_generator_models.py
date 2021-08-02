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

"""Tests for generator_models.py after
beginning to use OperatorCollections. """

from qiskit_dynamics.signals.signals import SignalList
import numpy as np
import numpy.random as rand
from scipy.linalg import expm
from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import GeneratorModel, Frame
from qiskit_dynamics.models.operator_collections import (
    DenseOperatorCollection,
    DenseLindbladCollection,
)
from qiskit_dynamics.signals import Signal
from qiskit_dynamics.dispatch import Array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestDenseOperatorCollection(QiskitDynamicsTestCase):
    """Tests for GeneratorModel."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

        # define a basic model
        w = 2.0
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2, -1j * 2 * np.pi * r * self.X / 2]
        signals = [w, Signal(1.0, w)]

        self.w = 2
        self.r = r
        self.basic_model = GeneratorModel(operators=operators, signals=signals)

    def test_frame_operator_errors(self):
        """Check different modes of error raising for frame setting."""

        # 1d array
        try:
            self.basic_model.frame = Array([1.0, 1.0])
        except QiskitError as e:
            self.assertTrue("anti-Hermitian" in str(e))

        # 2d array
        try:
            self.basic_model.frame = Array([[1.0, 0.0], [0.0, 1.0]])
        except QiskitError as e:
            self.assertTrue("anti-Hermitian" in str(e))

        # Operator
        try:
            self.basic_model.frame = self.Z
        except QiskitError as e:
            self.assertTrue("anti-Hermitian" in str(e))

    def test_diag_frame_operator_basic_model(self):
        """Test setting a diagonal frame operator for the internally
        set up basic model.
        """

        self._basic_frame_evaluate_test(Array([1j, -1j]), 1.123)
        self._basic_frame_evaluate_test(Array([1j, -1j]), np.pi)

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
            frame_operator = Array(frame_operator.data)
        if isinstance(frame_operator, Array) and frame_operator.ndim == 1:
            frame_operator = np.diag(frame_operator)

        value = self.basic_model(t)

        i2pi = -1j * 2 * np.pi

        U = expm(-np.array(frame_operator) * t)

        # drive coefficient
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)

        # manually evaluate frame
        expected = (
            i2pi * self.w * U @ self.Z.data @ U.conj().transpose() / 2
            + d_coeff * i2pi * U @ self.X.data @ U.conj().transpose() / 2
            - frame_operator
        )

        self.assertAllClose(value, expected)

    def test_evaluate_no_frame_basic_model(self):
        """Test evaluation without a frame in the basic model."""

        t = 3.21412
        value = self.basic_model(t)
        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = i2pi * self.w * self.Z.data / 2 + i2pi * d_coeff * self.X.data / 2

        self.assertAllClose(value, expected)

    def test_evaluate_in_frame_basis_basic_model(self):
        """Test evaluation in frame basis in the basic_model."""

        frame_op = -1j * (self.X + 0.2 * self.Y + 0.1 * self.Z).data

        # enter the frame given by the -1j * X
        self.basic_model.frame = frame_op

        # get the frame basis that is used in model
        _, U = np.linalg.eigh(1j * frame_op)

        t = 3.21412
        value = self.basic_model(t, in_frame_basis=True)

        # compose the frame basis transformation with the exponential
        # frame rotation (this will be multiplied on the right)
        U = expm(np.array(frame_op) * t) @ U
        Uadj = U.conj().transpose()

        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = (
            Uadj
            @ (i2pi * self.w * self.Z.data / 2 + i2pi * d_coeff * self.X.data / 2 - frame_op)
            @ U
        )

        self.assertAllClose(value, expected)

    def test_evaluate_pseudorandom(self):
        """Test evaluate with pseudorandom inputs."""

        rng = np.random.default_rng(30493)
        num_terms = 3
        dim = 5
        b = 1.0  # bound on size of random terms
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = Array(rand_op - rand_op.conj().transpose())
        randoperators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms, dim, dim)
        )

        rand_coeffs = Array(
            rng.uniform(low=-b, high=b, size=(num_terms))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms))
        )
        rand_carriers = Array(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = Array(rng.uniform(low=-b, high=b, size=(num_terms)))

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

        rng = np.random.default_rng(94818)
        num_terms = 5
        dim = 10
        b = 1.0  # bound on size of random terms
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = Array(rand_op - rand_op.conj().transpose())
        randoperators = Array(
            rng.uniform(low=-b, high=b, size=(num_terms, dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms, dim, dim))
        )

        rand_coeffs = Array(
            rng.uniform(low=-b, high=b, size=(num_terms))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms))
        )
        rand_carriers = Array(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = Array(rng.uniform(low=-b, high=b, size=(num_terms)))

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

    def _test_evaluate(self, frame_op, operators, coefficients, carriers, phases):

        sig_list = []
        for coeff, freq, phase in zip(coefficients, carriers, phases):

            def get_env_func(coeff=coeff):
                # pylint: disable=unused-argument
                def env(t):
                    return coeff

                return env

            sig_list.append(Signal(get_env_func(), freq, phase))
        model = GeneratorModel(operators, signals=sig_list)
        model.frame = frame_op

        value = model(1.0)
        coeffs = np.real(coefficients * np.exp(1j * 2 * np.pi * carriers * 1.0 + 1j * phases))

        self.assertAllClose(
            model.frame.operator_out_of_frame_basis(
                model._operator_collection.evaluate_generator(coeffs)
            ),
            np.tensordot(coeffs, operators, axes=1) - frame_op,
        )

        expected = (
            expm(-np.array(frame_op))
            @ np.tensordot(coeffs, operators, axes=1)
            @ expm(np.array(frame_op))
            - frame_op
        )

        self.assertAllClose(value, expected)

    def test_signal_setting(self):
        """Test updating the signals."""

        signals = [Signal(lambda t: 2 * t, 1.0), Signal(lambda t: t ** 2, 2.0)]
        self.basic_model.signals = signals

        t = 0.1
        value = self.basic_model(t)
        i2pi = -1j * 2 * np.pi
        Z_coeff = (2 * t) * np.cos(2 * np.pi * 1 * t)
        X_coeff = self.r * (t ** 2) * np.cos(2 * np.pi * 2 * t)
        expected = i2pi * Z_coeff * self.Z.data / 2 + i2pi * X_coeff * self.X.data / 2
        self.assertAllClose(value, expected)

    def test_signal_setting_None(self):
        """Test setting signals to None"""

        self.basic_model.signals = None
        self.assertTrue(self.basic_model.signals is None)

    def test_signal_setting_incorrect_length(self):
        """Test error being raised if signals is the wrong length."""

        try:
            self.basic_model.signals = [1.0]
        except QiskitError as e:
            self.assertTrue("same length" in str(e))

    def test_known_values_basic_functionality(self):
        """Test for checking that with known operators that
        the Model returns the analyticlly known values."""

        test_operator_list = Array([self.X, self.Y, self.Z])
        signals = SignalList([Signal(1, j / 3) for j in range(3)])
        simple_model = GeneratorModel(test_operator_list, drift=None, signals=signals, frame=None)

        res = simple_model.evaluate_generator(2)
        self.assertAllClose(res, Array([[-0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 0.5 + 0j]]))

        simple_model.drift = np.eye(2)
        res = simple_model.evaluate_generator(2)
        self.assertAllClose(res, Array([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]))
        simple_model.drift = None

    def test_frame_basis_transformation(self):
        """Test for checking that the frame basis transformations,
        pre- and post-rotation routines, as well as taking operators
        out of the frame basis are producing the analytically known
        answers."""
        test_operator_list = Array([self.X, self.Y, self.Z])
        signals = SignalList([Signal(1, j / 3) for j in range(3)])
        simple_model = GeneratorModel(test_operator_list, drift=None, signals=signals, frame=None)
        simple_model.drift = np.eye(2)
        fop = Array([[0, 1j], [1j, 0]])
        simple_model.frame = fop
        res = simple_model(2, in_frame_basis=False)
        expected = (
            expm(np.array(-2 * fop))
            @ (Array([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]) - fop)
            @ expm(np.array(2 * fop))
        )
        self.assertAllClose(res, expected)

        res = simple_model(2, in_frame_basis=True)
        expected = (
            np.array([[1, 1], [1, -1]])
            / np.sqrt(2)
            @ expected
            @ np.array([[1, 1], [1, -1]])
            / np.sqrt(2)
        )

        self.assertAllClose(res, expected)

        simple_model.drift = None

    def test_order_of_application_cases(self):
        """Test to see if the (nontrivial) setter methods
        of GeneratorModel are (a) working with all possible
        working types of input, (b) whether adding properties
        as part of the constructor phase or afterwards makes
        a difference, and (c) if added after the constructor
        phase, if the order in which they're set matters."""

        paulis = Array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
        extra = Array(np.eye(2))
        state = Array([0.2, 0.5])

        t = 2

        sarr = [Signal(1, j / 3) for j in range(3)]
        sigvals = np.real(SignalList(sarr).complex_value(t))

        farr = Array(np.array([[3j, 2j], [2j, 0]]))
        farr2 = Array(np.array([[1j, 2], [-2, 3j]]))
        evals, evect = np.linalg.eig(farr)
        diafarr = np.diag(evals)

        paulis_in_frame_basis = np.conjugate(np.transpose(evect)) @ paulis @ evect

        ## Run checks without frame for now
        gm1 = GeneratorModel(paulis, extra, sarr)
        gm2 = GeneratorModel(paulis, extra)
        gm2.signals = sarr
        gm3 = GeneratorModel(paulis, extra)
        gm3.signals = SignalList(sarr)

        # All should be the same, because there are no frames involved
        t11 = gm1.evaluate_generator(t, False)
        t12 = gm1.evaluate_generator(t, True)
        t21 = gm2.evaluate_generator(t, False)
        t22 = gm2.evaluate_generator(t, True)
        t31 = gm3.evaluate_generator(t, False)
        t32 = gm3.evaluate_generator(t, True)
        t_analytical = Array([[0.5, 1.0 + 0.5j], [1.0 - 0.5j, 1.5]])

        self.assertAllClose(t11, t12)
        self.assertAllClose(t11, t21)
        self.assertAllClose(t11, t22)
        self.assertAllClose(t11, t31)
        self.assertAllClose(t11, t32)
        self.assertAllClose(t11, t_analytical)

        # now work with a specific statevector
        ts1 = gm1.evaluate_rhs(t, state, in_frame_basis=False)
        ts2 = gm1.evaluate_rhs(t, state, in_frame_basis=True)
        ts_analytical = Array([0.6 + 0.25j, 0.95 - 0.1j])

        self.assertAllClose(ts1, ts2)
        self.assertAllClose(ts1, ts_analytical)

        ## Now, run checks with frame
        # If passing a frame in the first place, operators must be in frame basis.abs
        # Testing at the same time whether having Drift = None is an issue.
        gm1 = GeneratorModel(paulis, signals=sarr, frame=Frame(farr))
        gm2 = GeneratorModel(paulis, frame=farr)
        gm2.signals = SignalList(sarr)
        gm3 = GeneratorModel(paulis, frame=farr)
        gm3.signals = sarr
        # Does adding a frame after make a difference?
        # If so, does it make a difference if we add signals or the frame first?
        gm4 = GeneratorModel(paulis)
        gm4.signals = sarr
        gm4.frame = farr
        gm5 = GeneratorModel(paulis)
        gm5.frame = farr
        gm5.signals = sarr
        gm6 = GeneratorModel(paulis, signals=sarr)
        gm6.frame = farr
        # If we go to one frame, then transform back, does this make a difference?
        gm7 = GeneratorModel(paulis, signals=sarr)
        gm7.frame = farr2
        gm7.frame = farr

        t_in_frame_actual = Array(
            np.diag(np.exp(-t * evals))
            @ (np.tensordot(sigvals, paulis_in_frame_basis, axes=1) - diafarr)
            @ np.diag(np.exp(t * evals))
        )
        tf1 = gm1.evaluate_generator(t, in_frame_basis=True)
        tf2 = gm2.evaluate_generator(t, in_frame_basis=True)
        tf3 = gm3.evaluate_generator(t, in_frame_basis=True)
        tf4 = gm4.evaluate_generator(t, in_frame_basis=True)
        tf5 = gm5.evaluate_generator(t, in_frame_basis=True)
        tf6 = gm6.evaluate_generator(t, in_frame_basis=True)
        tf7 = gm7.evaluate_generator(t, in_frame_basis=True)

        self.assertAllClose(t_in_frame_actual, tf1)
        self.assertAllClose(tf1, tf2)
        self.assertAllClose(tf1, tf3)
        self.assertAllClose(tf1, tf4)
        self.assertAllClose(tf1, tf5)
        self.assertAllClose(tf1, tf6)
        self.assertAllClose(tf1, tf7)

    def test_vectorization_pseudorandom(self):
        """Test for whether evaluating a model at m different
        states, each with an n-length statevector, by passing
        an (m,n) Array provides the same value as passing each
        (n) Array individually."""
        rand.seed(9231)
        n = 8
        k = 4
        m = 2
        t = rand.rand()
        sig_list = SignalList([Signal(rand.rand(), rand.rand()) for j in range(k)])
        normal_states = rand.uniform(-1, 1, (n))
        vectorized_states = rand.uniform(-1, 1, (n, m))

        operators = rand.uniform(-1, 1, (k, n, n))

        gm = GeneratorModel(operators, drift=None, signals=sig_list)
        self.assertTrue(gm.evaluate_rhs(t, normal_states).shape == (n,))
        self.assertTrue(gm.evaluate_rhs(t, vectorized_states).shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                gm.evaluate_rhs(t, vectorized_states)[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i]),
            )

        farr = rand.uniform(-1, 1, (n, n))
        farr = farr - np.conjugate(np.transpose(farr))

        gm.frame = farr

        self.assertTrue(gm.evaluate_rhs(t, normal_states).shape == (n,))
        self.assertTrue(gm.evaluate_rhs(t, vectorized_states).shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                gm.evaluate_rhs(t, vectorized_states)[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i]),
            )

        vectorized_result = gm.evaluate_rhs(t, vectorized_states, in_frame_basis=True)

        self.assertTrue(gm.evaluate_rhs(t, normal_states, in_frame_basis=True).shape == (n,))
        self.assertTrue(vectorized_result.shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                vectorized_result[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i], in_frame_basis=True),
            )


class TestDenseOperatorCollectionJax(TestDenseOperatorCollection, TestJaxBase):
    """Jax version of TestGeneratorModel tests.

    Note: This class has no body but contains tests due to inheritance.
    """