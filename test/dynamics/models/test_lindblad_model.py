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
# pylint: disable=invalid-name,redundant-keyword-arg

"""Tests for qiskit_dynamics.models.lindblad_models.py. Most
of the actual calculation checking is handled at the level of a
models.operator_collection.DenseLindbladOperatorCollection test."""

import numpy as np

from scipy.linalg import expm

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import LindbladModel
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.array import Array
from qiskit_dynamics.type_utils import to_array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestLindbladModelErrors(QiskitDynamicsTestCase):
    """Test error raising for LindbladModel."""

    def test_all_operators_None(self):
        """Test error raised if no operators set."""

        with self.assertRaisesRegex(QiskitError, "requires at least one of"):
            LindbladModel()

    def test_operators_None_signals_not_None(self):
        """Test setting signals with operators being None."""

        # test Hamiltonian signals
        with self.assertRaisesRegex(QiskitError, "Hamiltonian signals must be None"):
            LindbladModel(
                static_hamiltonian=np.array([[1.0, 0.0], [0.0, -1.0]]), hamiltonian_signals=[1.0]
            )

        # test after initial instantiation
        model = LindbladModel(static_hamiltonian=np.array([[1.0, 0.0], [0.0, -1.0]]))
        with self.assertRaisesRegex(QiskitError, "Hamiltonian signals must be None"):
            model.signals = ([1.0], None)

        # test dissipator signals
        with self.assertRaisesRegex(QiskitError, "Dissipator signals must be None"):
            LindbladModel(
                static_hamiltonian=np.array([[1.0, 0.0], [0.0, -1.0]]), dissipator_signals=[1.0]
            )

        # test after initial instantiation
        model = LindbladModel(static_hamiltonian=np.array([[1.0, 0.0], [0.0, -1.0]]))
        with self.assertRaisesRegex(QiskitError, "Dissipator signals must be None"):
            model.signals = (None, [1.0])

    def test_operators_signals_length_mismatch(self):
        """Test setting operators and signals to incompatible lengths."""

        # Test Hamiltonian signals
        with self.assertRaisesRegex(QiskitError, "same length"):
            LindbladModel(
                hamiltonian_operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]),
                hamiltonian_signals=[1.0, 1.0],
            )

        # test after initial instantiation
        model = LindbladModel(hamiltonian_operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]))
        with self.assertRaisesRegex(QiskitError, "same length"):
            model.signals = ([1.0, 1.0], None)

        # Test dissipator signals
        with self.assertRaisesRegex(QiskitError, "same length"):
            LindbladModel(
                dissipator_operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]),
                dissipator_signals=[1.0, 1.0],
            )

        # test after initial instantiation
        model = LindbladModel(dissipator_operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]))
        with self.assertRaisesRegex(QiskitError, "same length"):
            model.signals = (None, [1.0, 1.0])

    def test_signals_bad_format(self):
        """Test setting signals in an unacceptable format."""

        # test Hamiltonian signals
        with self.assertRaisesRegex(QiskitError, "unaccepted format."):
            LindbladModel(
                hamiltonian_operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]),
                hamiltonian_signals=lambda t: t,
            )

        # test after initial instantiation
        model = LindbladModel(hamiltonian_operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]))
        with self.assertRaisesRegex(QiskitError, "unaccepted format."):
            model.signals = (lambda t: t, None)

        # test dissipator signals
        with self.assertRaisesRegex(QiskitError, "unaccepted format."):
            LindbladModel(
                dissipator_operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]),
                dissipator_signals=lambda t: t,
            )

        # test after initial instantiation
        model = LindbladModel(dissipator_operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]))
        with self.assertRaisesRegex(QiskitError, "unaccepted format."):
            model.signals = (None, lambda t: t)


class TestLindbladModelValidation(QiskitDynamicsTestCase):
    """Test validation handling of LindbladModel."""

    def test_operators_not_hermitian(self):
        """Test raising error if hamiltonian_operators are not Hermitian."""

        hamiltonian_operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]

        with self.assertRaisesRegex(QiskitError, "hamiltonian_operators must be Hermitian."):
            LindbladModel(hamiltonian_operators=hamiltonian_operators)

    def test_static_operator_not_hermitian(self):
        """Test raising error if static_hamiltonian is not Hermitian."""

        static_hamiltonian = np.array([[0.0, 1.0], [0.0, 0.0]])
        hamiltonian_operators = [np.array([[0.0, 1.0], [1.0, 0.0]])]

        with self.assertRaisesRegex(QiskitError, "static_hamiltonian must be Hermitian."):
            LindbladModel(
                hamiltonian_operators=hamiltonian_operators, static_hamiltonian=static_hamiltonian
            )

    def test_validate_false(self):
        """Verify setting validate=False avoids error raising."""

        lindblad_model = LindbladModel(
            hamiltonian_operators=[np.array([[0.0, 1.0], [0.0, 0.0]])],
            hamiltonian_signals=[1.0],
            validate=False,
        )

        self.assertAllClose(lindblad_model(1.0, np.eye(2)), np.zeros(2))


class TestLindbladModel(QiskitDynamicsTestCase):
    """Tests for LindbladModel."""

    def setUp(self):
        self.X = Operator.from_label("X").data
        self.Y = Operator.from_label("Y").data
        self.Z = Operator.from_label("Z").data

        # define a basic hamiltonian
        w = 2.0
        r = 0.5
        ham_operators = [2 * np.pi * self.Z / 2, 2 * np.pi * r * self.X / 2]
        ham_signals = [w, Signal(1.0, w)]

        self.w = w
        self.r = r

        static_dissipators = Array([[[0.0, 0.0], [1.0, 0.0]]])

        self.basic_lindblad = LindbladModel(
            hamiltonian_operators=ham_operators,
            hamiltonian_signals=ham_signals,
            static_dissipators=static_dissipators,
            evaluation_mode=self.evaluation_mode,
        )

    @property
    def evaluation_mode(self):
        """Evaluation mode to use for tests, useful for inheritance."""
        return "dense"

    def test_basic_lindblad_lmult(self):
        """Test lmult method of Lindblad generator OperatorModel."""
        A = Array([[1.0, 2.0], [3.0, 4.0]])

        t = 1.123
        ham = (
            2 * np.pi * self.w * self.Z / 2
            + 2 * np.pi * self.r * np.cos(2 * np.pi * self.w * t) * self.X / 2
        )
        sm = Array([[0.0, 0.0], [1.0, 0.0]])

        expected = self._evaluate_lindblad_rhs(A, ham, [sm])
        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            A = A.flatten(order="F")

        value = self.basic_lindblad(t, A)
        self.assertAllClose(expected, value)

    def test_evaluate_only_dissipators(self):
        """Test evaluation with just dissipators."""

        model = LindbladModel(
            dissipator_operators=[self.X],
            dissipator_signals=[1.0],
            evaluation_mode=self.evaluation_mode,
        )

        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

        expected = self._evaluate_lindblad_rhs(
            rho, ham=np.zeros((2, 2), dtype=complex), dissipators=[self.X]
        )
        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            rho = rho.flatten(order="F")

        self.assertAllClose(model(1.0, rho), expected)

    def test_evaluate_only_static_dissipators(self):
        """Test evaluation with just dissipators."""

        model = LindbladModel(
            static_dissipators=[self.X, self.Y], evaluation_mode=self.evaluation_mode
        )

        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        expected = self._evaluate_lindblad_rhs(
            rho, ham=np.zeros((2, 2), dtype=complex), dissipators=[self.X, self.Y]
        )
        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            rho = rho.flatten(order="F")

        self.assertAllClose(model(1.0, rho), expected)

    def test_evaluate_only_static_hamiltonian(self):
        """Test evaluation with just static hamiltonian."""

        model = LindbladModel(static_hamiltonian=self.X, evaluation_mode=self.evaluation_mode)

        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        expected = self._evaluate_lindblad_rhs(rho, ham=self.X)
        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            rho = rho.flatten(order="F")

        self.assertAllClose(model(1.0, rho), expected)

    def test_evaluate_only_hamiltonian_operators(self):
        """Test evaluation with just hamiltonian operators."""

        model = LindbladModel(
            hamiltonian_operators=[self.X],
            hamiltonian_signals=[1.0],
            evaluation_mode=self.evaluation_mode,
        )

        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        expected = self._evaluate_lindblad_rhs(rho, ham=self.X)
        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            rho = rho.flatten(order="F")

        self.assertAllClose(model(1.0, rho), expected)

    def test_lindblad_pseudorandom(self):
        """Test various evaluation modes of LindbladModel with structureless pseudorandom
        model parameters.
        """
        rng = np.random.default_rng(9848)
        dim = 10
        num_ham = 4
        num_diss = 3

        b = 1.0  # bound on size of random terms

        # generate random hamiltonian
        randoperators = rng.uniform(low=-b, high=b, size=(num_ham, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_ham, dim, dim)
        )
        rand_ham_ops = Array(randoperators + randoperators.conj().transpose([0, 2, 1]))

        # generate random hamiltonian coefficients
        rand_ham_coeffs = rng.uniform(low=-b, high=b, size=(num_ham)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_ham)
        )
        rand_ham_carriers = Array(rng.uniform(low=-b, high=b, size=(num_ham)))
        rand_ham_phases = Array(rng.uniform(low=-b, high=b, size=(num_ham)))

        ham_sigs = []
        for coeff, freq, phase in zip(rand_ham_coeffs, rand_ham_carriers, rand_ham_phases):
            ham_sigs.append(Signal(coeff, freq, phase))

        ham_sigs = SignalList(ham_sigs)

        # generate random static dissipators
        rand_static_diss = Array(
            rng.uniform(low=-b, high=b, size=(num_diss, dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(num_diss, dim, dim))
        )

        # generate random dissipators
        rand_diss = Array(
            rng.uniform(low=-b, high=b, size=(num_diss, dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(num_diss, dim, dim))
        )

        # random dissipator coefficients
        rand_diss_coeffs = rng.uniform(low=-b, high=b, size=(num_diss)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_diss)
        )
        rand_diss_carriers = Array(rng.uniform(low=-b, high=b, size=(num_diss)))
        rand_diss_phases = Array(rng.uniform(low=-b, high=b, size=(num_diss)))

        diss_sigs = []
        for coeff, freq, phase in zip(rand_diss_coeffs, rand_diss_carriers, rand_diss_phases):
            diss_sigs.append(Signal(coeff, freq, phase))

        diss_sigs = SignalList(diss_sigs)

        # random anti-hermitian frame operator
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = Array(rand_op - rand_op.conj().transpose())
        evect = np.linalg.eigh(1j * frame_op)[1]
        into_frame_basis = lambda x: evect.T.conj() @ x @ evect

        # construct model
        lindblad_model = LindbladModel(
            hamiltonian_operators=rand_ham_ops,
            hamiltonian_signals=ham_sigs,
            static_dissipators=rand_static_diss,
            dissipator_operators=rand_diss,
            dissipator_signals=diss_sigs,
            evaluation_mode=self.evaluation_mode,
        )
        lindblad_model.rotating_frame = frame_op

        t = rng.uniform(low=-b, high=b)
        # test storage of operators in class
        diss_coeffs = np.real(
            rand_diss_coeffs
            * np.exp(1j * 2 * np.pi * rand_diss_carriers * t + 1j * rand_diss_phases)
        )

        ham_coeffs = np.real(
            rand_ham_coeffs * np.exp(1j * 2 * np.pi * rand_ham_carriers * t + 1j * rand_ham_phases)
        )
        ham = np.tensordot(ham_coeffs, rand_ham_ops, axes=1)

        self.assertAllClose(ham_coeffs, ham_sigs(t))
        self.assertAllClose(diss_coeffs, diss_sigs(t))
        # lindblad model is in frame basis here
        lindblad_model.in_frame_basis = True
        self.assertAllClose(
            into_frame_basis(rand_diss),
            to_array(lindblad_model.dissipator_operators),
        )
        self.assertAllClose(
            into_frame_basis(rand_ham_ops),
            to_array(lindblad_model.hamiltonian_operators),
        )
        self.assertAllClose(
            into_frame_basis(-1j * frame_op),
            to_array(lindblad_model.static_hamiltonian),
        )
        lindblad_model.in_frame_basis = False
        self.assertAllClose(-1j * frame_op, lindblad_model.static_hamiltonian)

        # evaluation tests
        A = Array(
            rng.uniform(low=-b, high=b, size=(dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(dim, dim))
        )

        expected = self._evaluate_lindblad_rhs(
            A,
            ham,
            static_dissipators=rand_static_diss,
            dissipators=rand_diss,
            dissipator_coeffs=diss_coeffs,
            frame_op=frame_op,
            t=t,
        )
        expected_in_frame_basis = self._evaluate_lindblad_rhs(
            lindblad_model.rotating_frame.operator_out_of_frame_basis(A),
            ham,
            static_dissipators=rand_static_diss,
            dissipators=rand_diss,
            dissipator_coeffs=diss_coeffs,
            frame_op=frame_op,
            t=t,
        )
        expected_in_frame_basis = lindblad_model.rotating_frame.operator_into_frame_basis(
            expected_in_frame_basis
        )

        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            expected_in_frame_basis = expected_in_frame_basis.flatten(order="F")
            A = A.flatten(order="F")

        value = lindblad_model(t, A)
        self.assertAllClose(expected, value)

        lindblad_model.in_frame_basis = True
        value_in_frame_basis = lindblad_model(t, A)
        self.assertAllClose(expected, value)
        self.assertAllClose(expected_in_frame_basis, value_in_frame_basis)

    def test_dissipator_consistency(self):
        """Test consistent evaluation with different ways of specifying dissipators."""
        rng = np.random.default_rng(1231)
        dim = 8
        num_diss = 4

        b = 1.0  # bound on size of random terms

        # generate random dissipators
        rand_diss = Array(
            rng.uniform(low=-b, high=b, size=(num_diss, dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(num_diss, dim, dim))
        )

        static_model = LindbladModel(
            static_dissipators=rand_diss, evaluation_mode=self.evaluation_mode
        )
        non_static_model = LindbladModel(
            dissipator_operators=rand_diss,
            dissipator_signals=[1.0] * num_diss,
            evaluation_mode=self.evaluation_mode,
        )

        rand_input = Array(
            rng.uniform(low=-b, high=b, size=(dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(dim, dim))
        )
        if "vectorized" in self.evaluation_mode:
            rand_input = rand_input.flatten(order="F")

        self.assertAllClose(static_model(0.0, rand_input), non_static_model(0.0, rand_input))

    # pylint: disable=no-self-use,too-many-arguments
    def _evaluate_lindblad_rhs(
        self,
        A,
        ham,
        static_dissipators=None,
        dissipators=None,
        dissipator_coeffs=None,
        frame_op=None,
        t=0.0,
    ):
        """Evaluate the Lindblad equation

        frame_op assumed anti-Hermitian

        Note: here we force everything into numpy arrays as these parts of
        the test are just for confirmation
        """
        # if a frame operator is given, transform the model pieces into
        # the frame
        if frame_op is not None:
            frame_op = np.array(frame_op)
            U = expm(-t * frame_op)
            Uadj = U.conj().transpose()

            ham = U @ ham @ Uadj - 1j * frame_op

            if static_dissipators is not None:
                static_dissipators = np.array(static_dissipators)
                static_dissipators = np.array([U @ D @ Uadj for D in static_dissipators])

            if dissipators is not None:
                dissipators = np.array(dissipators)
                dissipators = np.array([U @ D @ Uadj for D in dissipators])

        ham = np.array(ham)
        A = np.array(A)
        ham_part = -1j * (ham @ A - A @ ham)

        if static_dissipators is None and dissipators is None:
            return ham_part

        diss_part = np.zeros_like(A)
        if static_dissipators is not None:
            # force numpy here if using JAX
            static_dissipators = np.array(static_dissipators)
            for D in static_dissipators:
                Dadj = D.conj().transpose()
                DadjD = Dadj @ D
                diss_part += D @ A @ Dadj - 0.5 * (DadjD @ A + A @ DadjD)

        if dissipators is not None:
            # force numpy here if using JAX
            dissipators = np.array(dissipators)
            if dissipator_coeffs is None:
                dissipator_coeffs = np.ones(len(dissipators))
            else:
                dissipator_coeffs = np.array(dissipator_coeffs)

            for c, D in zip(dissipator_coeffs, dissipators):
                Dadj = D.conj().transpose()
                DadjD = Dadj @ D
                diss_part += c * (D @ A @ Dadj - 0.5 * (DadjD @ A + A @ DadjD))

        return ham_part + diss_part

    def test_get_operators_when_None(self):
        """Test getting various operators when None."""

        model = LindbladModel(
            static_hamiltonian=np.array([[1.0, 0.0], [0.0, -1.0]]),
            evaluation_mode=self.evaluation_mode,
        )
        self.assertTrue(model.hamiltonian_operators is None)
        self.assertTrue(model.static_dissipators is None)
        self.assertTrue(model.dissipator_operators is None)

        model = LindbladModel(
            hamiltonian_operators=[np.array([[1.0, 0.0], [0.0, -1.0]])],
            evaluation_mode=self.evaluation_mode,
        )
        self.assertTrue(model.static_hamiltonian is None)


class TestLindbladModelJax(TestLindbladModel, TestJaxBase):
    """Jax version of TestLindbladModel tests."""

    def test_jitable_funcs(self):
        """Tests whether all functions are jitable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        rho = Array(np.array([[0.2, 0.4], [0.6, 0.8]]))
        if "vectorized" in self.evaluation_mode:
            rho = rho.flatten(order="F")

        self.jit_wrap(self.basic_lindblad.evaluate_rhs)(1.0, rho)

        self.basic_lindblad.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))
        self.jit_wrap(self.basic_lindblad.evaluate_rhs)(1.0, rho)

    def test_gradable_funcs(self):
        """Tests whether all functions are gradable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""

        rho = Array(np.array([[0.2, 0.4], [0.6, 0.8]]))
        if "vectorized" in self.evaluation_mode:
            rho = rho.flatten(order="F")

        self.jit_grad_wrap(self.basic_lindblad.evaluate_rhs)(1.0, rho)

        self.basic_lindblad.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))
        self.jit_grad_wrap(self.basic_lindblad.evaluate_rhs)(1.0, rho)


class TestLindbladModelSparse(TestLindbladModel):
    """Sparse-mode tests."""

    @property
    def evaluation_mode(self):
        """Evaluation mode to use for tests."""
        return "sparse"

    def test_switch_modes_and_evaluate(self):
        """Test construction of a model, switching modes, and evaluating."""

        model = LindbladModel(
            static_hamiltonian=self.Z,
            hamiltonian_operators=[self.X],
            hamiltonian_signals=[1.0],
            evaluation_mode=self.evaluation_mode,
        )
        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        ham = self.Z + self.X
        expected = -1j * (ham @ rho - rho @ ham)
        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            rho = rho.flatten(order="F")

        self.assertAllClose(model(1.0, rho), expected)

        if "vectorized" in self.evaluation_mode:
            model.evaluation_mode = "dense_vectorized"
        else:
            model.evaluation_mode = "dense"

        self.assertAllClose(model(1.0, rho), expected)

    def test_frame_change_sparse(self):
        """Test setting a frame after instantiation in sparse mode and evaluating."""
        model = LindbladModel(
            static_hamiltonian=self.Z,
            hamiltonian_operators=[self.X],
            hamiltonian_signals=[1.0],
            evaluation_mode=self.evaluation_mode,
        )

        # test non-diagonal frame
        model.rotating_frame = self.Z
        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        ham = expm(1j * self.Z) @ self.X @ expm(-1j * self.Z)
        expected = -1j * (ham @ rho - rho @ ham)
        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            rho = rho.flatten(order="F")
        self.assertAllClose(expected, model(1.0, rho))

        # test diagonal frame
        model.rotating_frame = np.array([1.0, -1.0])
        self.assertAllClose(expected, model(1.0, rho))

    def test_switching_to_sparse_with_frame(self):
        """Test switching to sparse with a frame already set."""

        model = LindbladModel(
            static_hamiltonian=self.Z,
            hamiltonian_operators=[self.X],
            hamiltonian_signals=[1.0],
            rotating_frame=np.array([1.0, -1.0]),
        )

        model.evaluation_mode = self.evaluation_mode

        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        ham = expm(1j * self.Z) @ self.X @ expm(-1j * self.Z)
        expected = -1j * (ham @ rho - rho @ ham)
        if "vectorized" in self.evaluation_mode:
            expected = expected.flatten(order="F")
            rho = rho.flatten(order="F")

        self.assertAllClose(expected, model(1.0, rho))


class TestLindbladModelJAXSparse(TestLindbladModelSparse, TestLindbladModelJax):
    """JAX sparse-mode tests."""


class TestLindbladModelDenseVectorized(TestLindbladModel):
    """Test class for dense vectorized mode operation of LindbladModel."""

    @property
    def evaluation_mode(self):
        return "dense_vectorized"


class TestLindbladModelSparseVectorized(TestLindbladModelSparse):
    """Test class for sparse vectorized mode operation of LindbladModel."""

    @property
    def evaluation_mode(self):
        return "sparse_vectorized"


class TestLindbladModelDenseVectorizedJax(TestLindbladModelJax):
    """Test class for jax dense vectorized mode operation of LindbladModel."""

    @property
    def evaluation_mode(self):
        return "dense_vectorized"


class TestLindbladModelSparseVectorizedJax(TestLindbladModelJax):
    """Test class for jax sparse vectorized mode operation of LindbladModel."""

    @property
    def evaluation_mode(self):
        return "sparse_vectorized"


def get_const_func(const):
    """Helper function for defining a constant function."""
    # pylint: disable=unused-argument
    def env(t):
        return const

    return env
