# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may obtain a copy of this license
# in the LICENSE.txt file in the root directory of this source tree or at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this copyright notice, and modified
# files need to carry a notice indicating that they have been altered from the originals.
# pylint: disable=invalid-name,no-member

"""Tests for generator_models.py."""

from functools import partial

from scipy.linalg import expm
import numpy as np
import numpy.random as rand
from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics.arraylias.alias import _isArrayLike
from qiskit_dynamics.models import GeneratorModel, RotatingFrame
from qiskit_dynamics.models.generator_model import (
    _static_operator_into_frame_basis,
    _operators_into_frame_basis,
)
from qiskit_dynamics.signals import Signal, SignalList
from ..common import QiskitDynamicsTestCase, test_array_backends


class TestGeneratorModelErrors(QiskitDynamicsTestCase):
    """Test deliberate error modes."""

    def test_both_static_operator_operators_None(self):
        """Test errors raising for a mal-formed GeneratorModel."""

        with self.assertRaisesRegex(QiskitError, "at least one of static_operator or operators"):
            GeneratorModel(static_operator=None, operators=None)

    def test_operators_None_signals_not_None(self):
        """Test setting signals with operators being None."""

        with self.assertRaisesRegex(QiskitError, "Signals must be None if operators is None."):
            GeneratorModel(
                static_operator=np.array([[1.0, 0.0], [0.0, -1.0]]), operators=None, signals=[1.0]
            )

        # test after initial instantiation
        model = GeneratorModel(static_operator=np.array([[1.0, 0.0], [0.0, -1.0]]))
        with self.assertRaisesRegex(QiskitError, "Signals must be None if operators is None."):
            model.signals = [1.0]

    def test_operators_signals_length_mismatch(self):
        """Test setting operators and signals to incompatible lengths."""
        with self.assertRaisesRegex(QiskitError, "same length as operators."):
            GeneratorModel(operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]), signals=[1.0, 1.0])

        # test after initial instantiation
        model = GeneratorModel(operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]))
        with self.assertRaisesRegex(QiskitError, "same length as operators."):
            model.signals = [1.0, 1.0]

    def test_signals_bad_format(self):
        """Test setting signals in an unacceptable format."""
        with self.assertRaisesRegex(QiskitError, "unaccepted format."):
            GeneratorModel(operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]), signals=lambda t: t)

        # test after initial instantiation
        model = GeneratorModel(operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]))
        with self.assertRaisesRegex(QiskitError, "unaccepted format."):
            model.signals = lambda t: t


@partial(test_array_backends, array_libraries=["numpy", "jax", "jax_sparse", "scipy_sparse"])
class Test_into_frame_basis_functions:
    """Tests for _static_operator_into_frame_basis and _operators_into_frame_basis."""

    def test_all_None(self):
        """Test all arguments being None."""

        static_operator = _static_operator_into_frame_basis(
            None, RotatingFrame(None), array_library=self.array_library()
        )
        operators = _operators_into_frame_basis(
            None, RotatingFrame(None), array_library=self.array_library()
        )

        self.assertTrue(static_operator is None)
        self.assertTrue(operators is None)

    def test_array_inputs_diagonal_frame(self):
        """Test correct handling when operators are arrays for diagonal frames."""

        static_operator = -1j * np.array([[1.0, 0.0], [0.0, -1.0]])
        operators = -1j * np.array([[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1j], [1j, 0.0]]])
        rotating_frame = RotatingFrame(-1j * np.array([1.0, -1.0]))

        out_static = _static_operator_into_frame_basis(
            static_operator, rotating_frame=rotating_frame, array_library=self.array_library()
        )
        out_operators = _operators_into_frame_basis(
            operators, rotating_frame=rotating_frame, array_library=self.array_library()
        )

        self.assertArrayType(out_static)
        self.assertArrayType(out_operators)

        self.assertAllClose(out_operators, operators)
        self.assertAllClose(out_static, np.zeros((2, 2)))

    def test_array_inputs_pseudo_random(self):
        """Test correct handling when operators are pseudo random arrays."""

        rng = np.random.default_rng(34233)
        num_terms = 3
        dim = 5
        b = 1.0  # bound on size of random terms
        static_operator = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        operators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )

        rotating_frame = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        rotating_frame = RotatingFrame(rotating_frame - rotating_frame.conj().transpose())

        out_static = _static_operator_into_frame_basis(
            static_operator, rotating_frame=rotating_frame, array_library=self.array_library()
        )
        out_operators = _operators_into_frame_basis(
            operators, rotating_frame=rotating_frame, array_library=self.array_library()
        )

        self.assertArrayType(out_static)
        self.assertArrayType(out_operators)

        expected_static = rotating_frame.operator_into_frame_basis(
            static_operator - rotating_frame.frame_operator
        )
        expected_operators = [rotating_frame.operator_into_frame_basis(op) for op in operators]

        self.assertAllClose(out_static, expected_static)
        self.assertAllClose(out_operators, expected_operators)


@partial(test_array_backends, array_libraries=["numpy", "jax", "jax_sparse", "scipy_sparse"])
class TestGeneratorModel:
    """Tests for GeneratorModel."""

    def setUp(self):
        """Build simple model elements."""
        self.X = Operator.from_label("X").data
        self.Y = Operator.from_label("Y").data
        self.Z = Operator.from_label("Z").data

        # define basic model elements
        self.w = 2.0
        self.r = 0.5
        self.operators = [-1j * 2 * np.pi * self.Z / 2, -1j * 2 * np.pi * self.r * self.X / 2]
        self.signals = [self.w, Signal(1.0, self.w)]

        self.basic_model = GeneratorModel(
            operators=self.operators, signals=self.signals, array_library=self.array_library()
        )

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
        basic_model = GeneratorModel(
            operators=self.operators,
            signals=self.signals,
            rotating_frame=frame_operator,
            array_library=self.array_library(),
        )

        # convert to 2d array
        if isinstance(frame_operator, Operator):
            frame_operator = frame_operator.data

        if _isArrayLike(frame_operator) and frame_operator.ndim == 1:
            frame_operator = np.diag(frame_operator)

        value = basic_model(t)

        i2pi = -1j * 2 * np.pi

        U = expm(-np.array(frame_operator) * t)

        # drive coefficient
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)

        # manually evaluate frame
        expected = (
            i2pi * self.w * U @ self.Z @ U.conj().transpose() / 2
            + d_coeff * i2pi * U @ self.X @ U.conj().transpose() / 2
            - frame_operator
        )

        self.assertArrayType(value)
        self.assertAllClose(value, expected)

    def test_evaluate_no_frame_basic_model(self):
        """Test evaluation without a frame in the basic model."""

        t = 3.21412
        value = self.basic_model(t)
        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = i2pi * self.w * self.Z / 2 + i2pi * d_coeff * self.X / 2

        self.assertArrayType(value)
        self.assertAllClose(value, expected)

    def test_evaluate_generator_in_frame_basis_basic_model(self):
        """Test generator evaluation in frame basis in the basic_model."""

        frame_op = -1j * (self.X + 0.2 * self.Y + 0.1 * self.Z)

        basic_model = GeneratorModel(
            operators=self.operators,
            signals=self.signals,
            rotating_frame=frame_op,
            array_library=self.array_library(),
        )

        # set to operate in frame basis
        basic_model.in_frame_basis = True

        # get the frame basis that is used in model
        _, U = unp.linalg.eigh(1j * frame_op)

        t = 3.21412
        value = basic_model(t)

        # compose the frame basis transformation with the exponential
        # frame rotation (this will be multiplied on the right)
        U = expm(np.array(frame_op) * t) @ U
        Uadj = U.conj().transpose()

        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = Uadj @ (i2pi * self.w * self.Z / 2 + i2pi * d_coeff * self.X / 2 - frame_op) @ U

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
        frame_op = rand_op - rand_op.conj().transpose()
        randoperators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms, dim, dim)
        )

        rand_coeffs = rng.uniform(low=-b, high=b, size=num_terms) + 1j * rng.uniform(
            low=-b, high=b, size=num_terms
        )
        rand_carriers = rng.uniform(low=-b, high=b, size=num_terms)
        rand_phases = rng.uniform(low=-b, high=b, size=num_terms)

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

        rng = np.random.default_rng(94818)
        num_terms = 5
        dim = 10
        b = 1.0  # bound on size of random terms
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = rand_op - rand_op.conj().transpose()
        randoperators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms, dim, dim)
        )

        rand_coeffs = rng.uniform(low=-b, high=b, size=num_terms) + 1j * rng.uniform(
            low=-b, high=b, size=num_terms
        )
        rand_carriers = rng.uniform(low=-b, high=b, size=num_terms)
        rand_phases = rng.uniform(low=-b, high=b, size=num_terms)

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

    def _test_evaluate(self, frame_op, operators, coefficients, carriers, phases):
        """Test evaluation of both dense and sparse."""
        sig_list = []
        for coeff, freq, phase in zip(coefficients, carriers, phases):

            def get_env_func(coeff=coeff):
                # pylint: disable=unused-argument
                def env(t):
                    return coeff

                return env

            sig_list.append(Signal(get_env_func(), freq, phase))
        model = GeneratorModel(
            operators=operators,
            signals=sig_list,
            rotating_frame=frame_op,
            array_library=self.array_library(),
        )

        value = model(1.0)
        coeffs = np.real(coefficients * np.exp(1j * 2 * np.pi * carriers * 1.0 + 1j * phases))

        self.assertAllClose(
            model.rotating_frame.operator_out_of_frame_basis(
                model._operator_collection.evaluate(coeffs)
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

        signals = [Signal(lambda t: 2 * t, 1.0), Signal(lambda t: t**2, 2.0)]
        self.basic_model.signals = signals

        t = 0.1
        value = self.basic_model(t)
        i2pi = -1j * 2 * np.pi
        Z_coeff = (2 * t) * np.cos(2 * np.pi * 1 * t)
        X_coeff = self.r * (t**2) * np.cos(2 * np.pi * 2 * t)
        expected = i2pi * Z_coeff * self.Z / 2 + i2pi * X_coeff * self.X / 2
        self.assertArrayType(value)
        self.assertAllClose(value, expected)

    def test_signal_setting_None(self):
        """Test setting signals to None"""

        self.basic_model.signals = None
        self.assertTrue(self.basic_model.signals is None)

    def test_evaluate_analytic(self):
        """Test for checking that with known operators that
        the Model returns the analyticlly known values."""

        test_operator_list = [self.X, self.Y, self.Z]
        signals = SignalList([Signal(1, j / 3) for j in range(3)])
        simple_model = GeneratorModel(
            operators=test_operator_list,
            static_operator=None,
            signals=signals,
            rotating_frame=None,
            array_library=self.array_library(),
        )

        res = simple_model.evaluate(2)
        self.assertArrayType(res)
        self.assertAllClose(res, np.array([[-0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 0.5 + 0j]]))

        simple_model = GeneratorModel(
            operators=test_operator_list,
            static_operator=self.asarray(np.eye(2)),
            signals=signals,
            rotating_frame=None,
            array_library=self.array_library(),
        )
        res = simple_model.evaluate(2)
        self.assertArrayType(res)
        self.assertAllClose(res, np.array([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]))

    def test_evaluate_in_frame_basis_analytic(self):
        """Tests evaluation in rotating frame against analytic values,
        both in specified basis and in frame basis."""
        test_operator_list = [self.X, self.Y, self.Z]
        signals = SignalList([Signal(1, j / 3) for j in range(3)])
        fop = np.array([[0, 1j], [1j, 0]])
        simple_model = GeneratorModel(
            operators=test_operator_list,
            static_operator=self.asarray(np.eye(2)),
            signals=signals,
            rotating_frame=fop,
            array_library=self.array_library(),
        )
        res = simple_model(2.0)
        expected = (
            expm(np.array(-2 * fop))
            @ (np.array([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]) - fop)
            @ expm(np.array(2 * fop))
        )
        self.assertArrayType(res)
        self.assertAllClose(res, expected)

        simple_model.in_frame_basis = True
        res = simple_model(2)
        expected = (
            np.array([[1, 1], [1, -1]])
            / np.sqrt(2)
            @ expected
            @ np.array([[1, 1], [1, -1]])
            / np.sqrt(2)
        )
        self.assertArrayType(res)
        self.assertAllClose(res, expected)

    def test_order_of_assigning_properties(self):
        """Tests whether setting signals in the constructor is the same as constructing without
        and then assigning them.
        """

        paulis = np.array([self.X, self.Y, self.Z])
        extra = np.eye(2)
        state = np.array([0.2, 0.5])

        t = 2

        sarr = [Signal(1, j / 3) for j in range(3)]
        sigvals = SignalList(sarr)(t)

        farr = np.array([[3j, 2j], [2j, 0]])
        evals, evect = np.linalg.eig(farr)
        diafarr = np.diag(evals)

        paulis_in_frame_basis = np.conjugate(np.transpose(evect)) @ paulis @ evect

        ## Run checks without rotating frame for now
        gm1 = GeneratorModel(
            operators=paulis,
            signals=sarr,
            static_operator=extra,
            array_library=self.array_library(),
        )
        gm2 = GeneratorModel(
            operators=paulis, static_operator=extra, array_library=self.array_library()
        )
        gm2.signals = sarr
        gm3 = GeneratorModel(
            operators=paulis, static_operator=extra, array_library=self.array_library()
        )
        gm3.signals = SignalList(sarr)

        # All should be the same, because there are no frames involved
        t11 = gm1.evaluate(t)
        gm1.in_frame_basis = True
        t12 = gm1.evaluate(t)
        t21 = gm2.evaluate(t)
        gm2.in_frame_basis = True
        t22 = gm2.evaluate(t)
        t31 = gm3.evaluate(t)
        gm3.in_frame_basis = True
        t32 = gm3.evaluate(t)
        t_analytical = np.array([[0.5, 1.0 + 0.5j], [1.0 - 0.5j, 1.5]])

        self.assertArrayType(t11)
        self.assertArrayType(t12)
        self.assertArrayType(t21)
        self.assertArrayType(t22)
        self.assertArrayType(t31)
        self.assertArrayType(t32)

        self.assertAllClose(t11, t12)
        self.assertAllClose(t11, t21)
        self.assertAllClose(t11, t22)
        self.assertAllClose(t11, t31)
        self.assertAllClose(t11, t32)
        self.assertAllClose(t11, t_analytical)

        # now work with a specific statevector
        ts1 = gm1.evaluate_rhs(t, state)
        gm1.in_frame_basis = True
        ts2 = gm1.evaluate_rhs(t, state)
        ts_analytical = np.array([0.6 + 0.25j, 0.95 - 0.1j])

        self.assertArrayType(ts1)
        self.assertArrayType(ts2)

        self.assertAllClose(ts1, ts2)
        self.assertAllClose(ts1, ts_analytical)
        self.assertAllClose(gm1(t) @ state, ts1)
        gm1.in_frame_basis = False
        self.assertAllClose(gm1(t) @ state, ts1)

        ## Now, run checks with frame
        # If passing a frame in the first place, operators must be in frame basis.
        # Testing at the same time whether having static_operator = None is an issue.
        gm1 = GeneratorModel(
            operators=paulis,
            signals=sarr,
            rotating_frame=RotatingFrame(farr),
            in_frame_basis=True,
            array_library=self.array_library(),
        )
        gm2 = GeneratorModel(
            operators=paulis,
            rotating_frame=farr,
            in_frame_basis=True,
            array_library=self.array_library(),
        )
        gm2.signals = SignalList(sarr)
        gm3 = GeneratorModel(
            operators=paulis,
            rotating_frame=farr,
            in_frame_basis=True,
            array_library=self.array_library(),
        )
        gm3.signals = sarr

        t_in_frame_actual = (
            np.diag(np.exp(-t * evals))
            @ (np.tensordot(sigvals, paulis_in_frame_basis, axes=1) - diafarr)
            @ np.diag(np.exp(t * evals))
        )
        tf1 = gm1.evaluate(t)
        tf2 = gm2.evaluate(t)
        tf3 = gm3.evaluate(t)

        self.assertArrayType(tf1)
        self.assertArrayType(tf2)
        self.assertArrayType(tf3)
        self.assertAllClose(t_in_frame_actual, tf1)
        self.assertAllClose(tf1, tf2)
        self.assertAllClose(tf1, tf3)

    def test_evaluate_rhs_lmult_equivalent_analytic(self):
        """Tests whether evaluate(t) @ state == evaluate_rhs(t,state)
        for analytically known values."""

        paulis = np.array([self.X, self.Y, self.Z])
        extra = np.eye(2)

        t = 2

        farr = np.array([[3j, 2j], [2j, 0]])
        evals, evect = np.linalg.eig(farr)
        diafarr = np.diag(evals)

        paulis_in_frame_basis = np.conjugate(np.transpose(evect)) @ paulis @ evect
        extra_in_basis = evect.T.conj() @ extra @ evect

        sarr = [Signal(1, j / 3) for j in range(3)]
        sigvals = SignalList(sarr)(t)

        t_in_frame_actual = (
            np.diag(np.exp(-t * evals))
            @ (np.tensordot(sigvals, paulis_in_frame_basis, axes=1) + extra_in_basis - diafarr)
            @ np.diag(np.exp(t * evals))
        )

        state = np.array([0.3, 0.1])
        state_in_frame_basis = np.conjugate(np.transpose(evect)) @ state

        gm1 = GeneratorModel(
            operators=paulis,
            signals=sarr,
            rotating_frame=farr,
            static_operator=extra,
            in_frame_basis=True,
            array_library=self.array_library(),
        )
        res = gm1(t)
        self.assertArrayType(res)
        self.assertAllClose(res, t_in_frame_actual)
        self.assertAllClose(
            gm1(t, state_in_frame_basis),
            t_in_frame_actual @ state_in_frame_basis,
        )

        t_not_in_frame_actual = (
            expm(np.array(-t * farr))
            @ (np.tensordot(sigvals, paulis, axes=1) + extra - farr)
            @ expm(np.array(t * farr))
        )

        gm2 = GeneratorModel(
            operators=paulis,
            signals=sarr,
            rotating_frame=farr,
            static_operator=extra,
            in_frame_basis=False,
            array_library=self.array_library(),
        )
        gm1.in_frame_basis = False
        res = gm2(t)
        self.assertArrayType(res)
        self.assertAllClose(res, t_not_in_frame_actual)
        self.assertAllClose(gm1(t, state), t_not_in_frame_actual @ state)

    def test_evaluate_rhs_vectorized_pseudorandom(self):
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

        gm = GeneratorModel(
            operators=operators,
            static_operator=None,
            signals=sig_list,
            array_library=self.array_library(),
        )
        self.assertTrue(gm.evaluate_rhs(t, normal_states).shape == (n,))
        self.assertTrue(gm.evaluate_rhs(t, vectorized_states).shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                gm.evaluate_rhs(t, vectorized_states)[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i]),
            )

        # add pseudo frame to test
        farr = rand.uniform(-1, 1, (n, n))
        farr = farr - np.conjugate(np.transpose(farr))
        gm = gm = GeneratorModel(
            operators=operators,
            static_operator=None,
            signals=sig_list,
            rotating_frame=farr,
            array_library=self.array_library(),
        )

        self.assertTrue(gm.evaluate_rhs(t, normal_states).shape == (n,))
        self.assertTrue(gm.evaluate_rhs(t, vectorized_states).shape == (n, m))
        vectorized_result = gm.evaluate_rhs(t, vectorized_states)
        for i in range(m):
            self.assertAllClose(
                vectorized_result[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i]),
            )

        gm.in_frame_basis = True
        vectorized_result = gm(t, vectorized_states)

        self.assertTrue(gm(t, normal_states).shape == (n,))
        self.assertTrue(vectorized_result.shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                vectorized_result[:, i],
                gm(t, vectorized_states[:, i]),
            )

    def test_evaluate_static(self):
        """Test evaluation of a GeneratorModel with only a static component."""

        static_model = GeneratorModel(static_operator=self.X, array_library=self.array_library())
        self.assertAllClose(self.X, static_model(1.0))

        # now with frame
        frame_op = -1j * (self.Z + 1.232 * self.Y)
        static_model = GeneratorModel(
            static_operator=self.X, rotating_frame=frame_op, array_library=self.array_library()
        )
        t = 1.2312
        expected = expm(-frame_op * t) @ (self.X - frame_op) @ expm(frame_op * t)
        res = static_model(t)
        self.assertArrayType(res)
        self.assertAllClose(expected, res)

    def test_get_operators_when_None(self):
        """Test getting operators when None."""

        model = GeneratorModel(
            static_operator=np.array([[1.0, 0.0], [0.0, -1.0]]), array_library=self.array_library()
        )
        self.assertTrue(model.operators is None)


@partial(test_array_backends, array_libraries=["jax", "jax_sparse"])
class TestGeneratorModelJAXTransformations:
    """Test GeneratorModel under JAX transformations."""

    def setUp(self):
        """Build simple model elements."""
        self.X = Operator.from_label("X").data
        self.Y = Operator.from_label("Y").data
        self.Z = Operator.from_label("Z").data

        # define basic model elements
        self.w = 2.0
        self.r = 0.5
        self.operators = [-1j * 2 * np.pi * self.Z / 2, -1j * 2 * np.pi * self.r * self.X / 2]
        self.signals = [self.w, Signal(1.0, self.w)]

        self.basic_model = GeneratorModel(
            operators=self.operators, signals=self.signals, array_library=self.array_library()
        )

        self.basic_model_w_frame = GeneratorModel(
            operators=self.operators,
            signals=self.signals,
            rotating_frame=np.array([[3j, 2j], [2j, 0]]),
            array_library=self.array_library(),
        )

    def test_jitable_funcs(self):
        """Tests whether all functions are jitable. Checks if having a frame makes a difference, as
        well as all jax-compatible evaluation_modes.
        """
        from jax import jit

        jit(self.basic_model.evaluate)(1.0)
        jit(self.basic_model.evaluate_rhs)(1.0, np.array([0.2, 0.4]))

        jit(self.basic_model_w_frame.evaluate)(1.0)
        jit(self.basic_model_w_frame.evaluate_rhs)(1.0, np.array([0.2, 0.4]))

    def test_gradable_funcs(self):
        """Tests whether all functions are gradable. Checks if having a frame makes a difference, as
        well as all jax-compatible evaluation_modes.
        """
        self.jit_grad(self.basic_model.evaluate)(1.0)
        self.jit_grad(self.basic_model.evaluate_rhs)(1.0, np.array([0.2, 0.4]))

        self.jit_grad(self.basic_model_w_frame.evaluate)(1.0)
        self.jit_grad(self.basic_model_w_frame.evaluate_rhs)(1.0, np.array([0.2, 0.4]))
