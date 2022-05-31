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
Tests for solver classes module.
"""

import numpy as np
from scipy.linalg import expm
from ddt import ddt, data, unpack

from qiskit import QiskitError
from qiskit.quantum_info import Operator, Statevector, SuperOp, DensityMatrix

from qiskit_dynamics import Solver, Signal, solve_lmde
from qiskit_dynamics.models import HamiltonianModel, LindbladModel, rotating_wave_approximation
from qiskit_dynamics.array import Array
from qiskit_dynamics.type_utils import to_array

from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestSolverDeprecations(QiskitDynamicsTestCase):
    """Test deprecation warnings and deprecated behaviour."""

    def setUp(self):
        self.X = Operator.from_label("X")

    def test_deprecated_signals_at_construction(self):
        """Test deprecation warning raised when signals passed to constructor."""

        with self.assertWarnsRegex(DeprecationWarning, "deprecated arguments"):
            Solver(hamiltonian_operators=[self.X], hamiltonian_signals=[1.0])

        with self.assertWarnsRegex(DeprecationWarning, "deprecated arguments"):
            Solver(dissipator_operators=[self.X], dissipator_signals=[1.0])

    def test_deprecated_signals_property(self):
        """Test deprecation warning raised when setting or getting signals property."""

        solver = Solver(hamiltonian_operators=[self.X])

        with self.assertWarnsRegex(DeprecationWarning, "signals property is deprecated"):
            solver.signals = [1.0]

        with self.assertWarnsRegex(DeprecationWarning, "signals property is deprecated"):
            solver.signals  # pylint: disable=pointless-statement

    def test_copy_deprecated(self):
        """Test copy method raises deprecation warning."""

        solver = Solver(hamiltonian_operators=[self.X])

        with self.assertWarnsRegex(DeprecationWarning, "copy method is deprecated"):
            solver.copy()

    def test_no_signals_to_solve(self):
        """Test raising of deprecation warning if no signals passed to solve,
        and signals present in model.
        """

        with self.assertWarnsRegex(DeprecationWarning, "deprecated arguments"):
            solver = Solver(hamiltonian_operators=[self.X], hamiltonian_signals=[1.0])

        with self.assertWarnsRegex(DeprecationWarning, "No signals specified to solve"):
            solver.solve(t_span=[0.0, 0.1], y0=np.array([0.0, 1.0]))

        with self.assertWarnsRegex(DeprecationWarning, "deprecated arguments"):
            solver = Solver(
                hamiltonian_operators=[self.X],
                hamiltonian_signals=[1.0],
                dissipator_operators=[self.X],
                dissipator_signals=[1.0],
            )

        with self.assertWarnsRegex(DeprecationWarning, "No signals specified to solve"):
            solver.solve(t_span=[0.0, 0.1], y0=np.array([[0.0, 0.0], [0.0, 1.0]]))

    def test_replacing_signals_in_solve(self):
        """Test that the signals stored in the signals attribute are correctly replaced."""

        with self.assertWarnsRegex(DeprecationWarning, "deprecated arguments"):
            solver = Solver(
                hamiltonian_operators=[self.X],
                hamiltonian_signals=[1.0],
            )

        y0 = np.array([1.0, 0.0])
        results = solver.solve(t_span=[0.0, 0.1], y0=y0, signals=[2.0], atol=1e-10, rtol=1e-10)
        expected = expm(-1j * 0.1 * 2 * self.X.data) @ y0
        self.assertAllClose(results.y[-1], expected)

        with self.assertWarnsRegex(DeprecationWarning, "signals property is deprecated"):
            self.assertTrue(solver.signals[0] == 1.0)


class TestSolverValidation(QiskitDynamicsTestCase):
    """Test validation checks."""

    def test_hamiltonian_operators_array_not_hermitian(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]

        with self.assertRaisesRegex(QiskitError, "operators must be Hermitian."):
            Solver(hamiltonian_operators=operators)

    def test_validation_override_hamiltonian(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]
        solver = Solver(hamiltonian_operators=operators, validate=False)

        self.assertAllClose(solver.model.operators, operators)

    def test_hamiltonian_operators_array_not_hermitian_lindblad(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]

        with self.assertRaisesRegex(QiskitError, "operators must be Hermitian"):
            Solver(hamiltonian_operators=operators, static_dissipators=operators)

    def test_validation_override_lindblad(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]
        solver = Solver(
            hamiltonian_operators=operators, static_dissipators=operators, validate=False
        )

        self.assertAllClose(solver.model.hamiltonian_operators, operators)

    def test_incompatible_input_lengths(self):
        """Test exception raised if args to solve are incompatible lengths."""

        solver = Solver(hamiltonian_operators=[np.array([[0.0, 1.0], [1.0, 0.0]])])

        error_str = "y0 specifies 3 inputs, but signals is of length 2"
        with self.assertRaisesRegex(QiskitError, error_str):
            solver.solve(t_span=[0.0, 1.0], y0=[np.array([0.0, 1.0])] * 3, signals=[[1.0]] * 2)

        error_str = "y0 specifies 4 inputs, but t_span is of length 2"
        with self.assertRaisesRegex(QiskitError, error_str):
            solver.solve(t_span=[[0.0, 1.0]] * 2, y0=[np.array([0.0, 1.0])] * 4, signals=[1.0])


class TestSolverExceptions(QiskitDynamicsTestCase):
    """Tests for Solver exception raising based on input types."""

    def setUp(self):
        X = Operator.from_label("X")
        self.ham_solver = Solver(hamiltonian_operators=[X])

        self.lindblad_solver = Solver(hamiltonian_operators=[X], static_dissipators=[X])

        self.vec_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[X],
            evaluation_mode="dense_vectorized",
        )

    def test_hamiltonian_shape_error(self):
        """Test error raising if invalid shape for Hamiltonian model."""

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.ham_solver.solve(t_span=[0.0, 1.0], y0=np.array([1.0, 0.0, 0.0]), signals=[1.0])

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.ham_solver.solve(
                t_span=[0.0, 1.0], y0=np.array([[[1.0, 0.0, 0.0]]]), signals=[1.0]
            )

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.ham_solver.solve(
                t_span=[0.0, 1.0], y0=Statevector(np.array([1.0, 0.0, 0.0])), signals=[1.0]
            )

    def test_lindblad_shape_error(self):
        """Test error raising if invalid shape for Lindblad model."""

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.lindblad_solver.solve(
                t_span=[0.0, 1.0], y0=np.array([1.0, 0.0, 0.0]), signals=[1.0]
            )

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.lindblad_solver.solve(
                t_span=[0.0, 1.0], y0=np.array([[[1.0, 0.0, 0.0]]]), signals=[1.0]
            )

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.lindblad_solver.solve(
                t_span=[0.0, 1.0], y0=Statevector(np.array([1.0, 0.0, 0.0])), signals=[1.0]
            )

    def test_vectorized_lindblad_shape_error(self):
        """Test error raising if invalid shape for vectorized Lindblad model."""

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.vec_lindblad_solver.solve(
                t_span=[0.0, 1.0], y0=np.array([[1.0, 0.0], [0.0, 1.0]]), signals=[1.0]
            )

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.vec_lindblad_solver.solve(
                t_span=[0.0, 1.0], y0=DensityMatrix(np.array([1.0, 0.0, 0.0])), signals=[1.0]
            )

        with self.assertRaisesRegex(QiskitError, "Shape mismatch"):
            self.vec_lindblad_solver.solve(
                t_span=[0.0, 1.0], y0=Statevector(np.array([1.0, 0.0, 0.0])), signals=[1.0]
            )

    def test_non_vectorized_SuperOp_error(self):
        """Test SuperOp simulation attempt for non-vectorized Lindblad model."""

        with self.assertRaisesRegex(QiskitError, "Simulating SuperOp"):
            self.lindblad_solver.solve(t_span=[0.0, 1.0], y0=SuperOp(np.eye(4)), signals=[1.0])


class TestSolverSignalHandling(QiskitDynamicsTestCase):
    """Test correct handling of signals arguments to solve."""

    def setUp(self):
        """Set up some simple models."""
        X = 2 * np.pi * Operator.from_label("X") / 2
        Z = 2 * np.pi * Operator.from_label("Z") / 2
        self.X = X
        self.Z = Z

        self.ham_model = HamiltonianModel(
            operators=[X],
            static_operator=5 * Z,
            rotating_frame=5 * Z,
        )

        self.lindblad_model = LindbladModel(
            hamiltonian_operators=[X],
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.td_lindblad_model = LindbladModel(
            hamiltonian_operators=[X],
            static_dissipators=[0.01 * X],
            dissipator_operators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

    def test_hamiltonian_model(self):
        """Test for solver with only Hamiltonian information."""

        ham_solver = Solver(
            hamiltonian_operators=[self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
        )

        y0 = np.array([0.0, 1.0])
        t_span = [0.0, 1.42]
        signals = [Signal(3.0)]

        res1 = ham_solver.solve(t_span=t_span, y0=y0, signals=signals)

        self.ham_model.signals = signals
        res2 = solve_lmde(generator=self.ham_model, t_span=t_span, y0=y0)

        self.assertAllClose(res1.y, res2.y)

    def test_lindblad_model(self):
        """Test for solver with only static lindblad information."""

        lindblad_solver = Solver(
            hamiltonian_operators=[self.X],
            static_dissipators=[0.01 * self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
        )

        y0 = np.array([[0.0, 1.0], [0.0, 1.0]])
        t_span = [0.0, 1.42]
        signals = [Signal(3.0)]

        res1 = lindblad_solver.solve(t_span=t_span, y0=y0, signals=signals)

        self.lindblad_model.signals = (signals, None)
        res2 = solve_lmde(generator=self.lindblad_model, t_span=t_span, y0=y0)

        self.assertAllClose(res1.y, res2.y)

    def test_td_lindblad_model(self):
        """Test for solver with time-dependent lindblad information."""

        td_lindblad_solver = Solver(
            hamiltonian_operators=[self.X],
            static_dissipators=[0.01 * self.X],
            dissipator_operators=[0.01 * self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
        )

        y0 = np.array([[0.0, 1.0], [0.0, 1.0]])
        t_span = [0.0, 1.42]
        signals = ([Signal(3.0)], [Signal(2.0)])

        res1 = td_lindblad_solver.solve(t_span=t_span, y0=y0, signals=signals)

        self.td_lindblad_model.signals = signals
        res2 = solve_lmde(generator=self.td_lindblad_model, t_span=t_span, y0=y0)

        self.assertAllClose(res1.y, res2.y)

    def test_rwa_ham_model(self):
        """Test correct handling of RWA for a Hamiltonian model."""

        rwa_ham_solver = Solver(
            hamiltonian_operators=[self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
            rwa_cutoff_freq=5.0,
            rwa_carrier_freqs=[5.0],
        )

        y0 = np.array([0.0, 1.0])
        t_span = [0.0, 1.0]
        signals = [Signal(1.0, carrier_freq=5.0)]

        res1 = rwa_ham_solver.solve(t_span=t_span, y0=y0, signals=signals)

        self.ham_model.signals = signals
        rwa_ham_model = rotating_wave_approximation(self.ham_model, cutoff_freq=5.0)
        res2 = solve_lmde(generator=rwa_ham_model, t_span=t_span, y0=y0)

        self.assertAllClose(res1.y, res2.y)

    def test_rwa_lindblad_model(self):
        """Test correct handling of RWA for Lindblad model without
        time-dependent dissipator terms.
        """

        rwa_lindblad_solver = Solver(
            hamiltonian_operators=[self.X],
            static_dissipators=[0.01 * self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
            rwa_cutoff_freq=5.0,
            rwa_carrier_freqs=[5.0],
        )

        y0 = np.array([[0.0, 0.0], [0.0, 1.0]])
        t_span = [0.0, 1.0]
        signals = [Signal(1.0, carrier_freq=5.0)]

        res1 = rwa_lindblad_solver.solve(t_span=t_span, y0=y0, signals=signals)

        self.lindblad_model.signals = (signals, None)
        rwa_lindblad_model = rotating_wave_approximation(self.lindblad_model, cutoff_freq=5.0)
        res2 = solve_lmde(generator=rwa_lindblad_model, t_span=t_span, y0=y0)

        self.assertAllClose(res1.y, res2.y)

    def test_rwa_td_lindblad_model(self):
        """Test correct handling of RWA for Lindblad model with
        time-dependent dissipator terms.
        """

        rwa_td_lindblad_solver = Solver(
            hamiltonian_operators=[self.X],
            static_dissipators=[0.01 * self.X],
            dissipator_operators=[0.01 * self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
            rwa_cutoff_freq=5.0,
            rwa_carrier_freqs=([5.0], [5.0]),
        )

        y0 = np.array([[0.0, 0.0], [0.0, 1.0]])
        t_span = [0.0, 1.0]
        signals = ([Signal(1.0, carrier_freq=5.0)], [Signal(1.0, carrier_freq=5.0)])

        res1 = rwa_td_lindblad_solver.solve(t_span=t_span, y0=y0, signals=signals)

        self.td_lindblad_model.signals = signals
        rwa_td_lindblad_model = rotating_wave_approximation(self.td_lindblad_model, cutoff_freq=5.0)
        res2 = solve_lmde(generator=rwa_td_lindblad_model, t_span=t_span, y0=y0)

        self.assertAllClose(res1.y, res2.y)


class TestSolverSimulation(QiskitDynamicsTestCase):
    """Test cases for correct simulation for Solver class."""

    def setUp(self):
        """Set up some simple models."""
        X = 2 * np.pi * Operator.from_label("X") / 2
        Z = 2 * np.pi * Operator.from_label("Z") / 2
        self.X = X
        self.Z = Z
        self.ham_solver = Solver(
            hamiltonian_operators=[X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.rwa_ham_solver = Solver(
            hamiltonian_operators=[X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            rwa_cutoff_freq=2 * 5.0,
            rwa_carrier_freqs=[5.0],
        )

        self.lindblad_solver = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.vec_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            evaluation_mode="dense_vectorized",
        )

        # lindblad solver with no dissipation for testing
        self.vec_lindblad_solver_no_diss = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[0.0 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            evaluation_mode="dense_vectorized",
        )
        self.method = "DOP853"

    def test_state_dims_preservation(self):
        """Test that state shapes are correctly preserved."""

        # Hamiltonian only model
        solver = Solver(static_hamiltonian=np.zeros((6, 6)))
        y0 = Statevector(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dims=(2, 3))
        yf = solver.solve(t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, Statevector))
        self.assertTrue(yf.dims() == (2, 3))

        y0 = DensityMatrix(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dims=(2, 3))
        yf = solver.solve(t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, DensityMatrix))
        self.assertTrue(yf.dims() == (2, 3))

        # model with Lindblad terms
        solver = Solver(static_dissipators=np.zeros((1, 6, 6)))
        y0 = Statevector(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dims=(2, 3))
        yf = solver.solve(t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, DensityMatrix))
        self.assertTrue(yf.dims() == (2, 3))

        y0 = DensityMatrix(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dims=(2, 3))
        yf = solver.solve(t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, DensityMatrix))
        self.assertTrue(yf.dims() == (2, 3))

        # SuperOp
        solver.model.evaluation_mode = "dense_vectorized"
        y0 = SuperOp(np.eye(36), input_dims=(2, 3), output_dims=(3, 2))
        yf = solver.solve(t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, SuperOp))
        self.assertTrue(yf.input_dims() == (2, 3) and yf.output_dims() == (3, 2))

    def test_convert_results(self):
        """Test convert_results option in a Lindblad simulation of a Statevector."""

        results = self.lindblad_solver.solve(
            t_span=[0.0, 1.0],
            y0=Statevector([0.0, 1.0]),
            signals=[Signal(1.0, 5.0)],
            convert_results=False,
            method=self.method,
        )
        self.assertTrue(not isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(results.y[-1].shape == (2, 2))
        self.assertTrue(results.y[-1][0, 0] > 0.99 and results.y[-1][0, 0] < 0.999)

    def test_lindblad_solve_statevector(self):
        """Test correct conversion of Statevector to DensityMatrix."""

        results = self.lindblad_solver.solve(
            t_span=[0.0, 1.0],
            y0=Statevector([0.0, 1.0]),
            signals=[Signal(1.0, 5.0)],
            method=self.method,
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(results.y[-1].data[0, 0] > 0.99 and results.y[-1].data[0, 0] < 0.999)

    def test_vec_lindblad_statevector(self):
        """Test correct conversion of Statevector to DensityMatrix and vectorized solving."""

        results = self.vec_lindblad_solver.solve(
            t_span=[0.0, 1.0],
            y0=Statevector([0.0, 1.0]),
            signals=[Signal(1.0, 5.0)],
            method=self.method,
        )
        results2 = self.lindblad_solver.solve(
            t_span=[0.0, 1.0],
            y0=Statevector([0.0, 1.0]),
            signals=[Signal(1.0, 5.0)],
            method=self.method,
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertAllClose(results.y[-1].data, results2.y[-1].data)

    def test_array_vectorized_lindblad(self):
        """Test Lindblad solver is array-vectorized."""
        results = self.lindblad_solver.solve(
            t_span=[0.0, 1.0],
            y0=np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]]),
            signals=[Signal(1.0, 5.0)],
            method=self.method,
        )
        self.assertTrue(results.y[-1][0, 0, 0] > 0.99 and results.y[-1][0, 0, 0] < 0.999)
        self.assertTrue(results.y[-1][1, 1, 1] > 0.99 and results.y[-1][1, 1, 1] < 0.999)

    def test_rwa_hamiltonian(self):
        """Test perfect inversion for pi pulse with RWA."""
        results = self.rwa_ham_solver.solve(
            t_span=[0.0, 1.0],
            y0=np.array([0.0, 1.0]),
            signals=[Signal(1.0, 5.0)],
            atol=1e-10,
            rtol=1e-10,
            method=self.method,
        )
        self.assertTrue(np.abs(results.y[-1][0]) > (1 - 1e-8))

    def test_hamiltonian_DensityMatrix(self):
        """Test correct conjugation of Hamiltonian-based density matrix simulation."""
        results = self.ham_solver.solve(
            t_span=[0.0, 1.0],
            y0=DensityMatrix(np.array([0.0, 1.0])),
            signals=[Signal(1.0, 5.0)],
            atol=1e-10,
            rtol=1e-10,
            method=self.method,
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(np.abs(results.y[-1].data[0, 0]) > 0.999)

    def test_hamiltonian_SuperOp(self):
        """Test Hamiltonian-based SuperOp simulation."""
        results = self.rwa_ham_solver.solve(
            t_span=[0.0, 1.0],
            y0=SuperOp(np.eye(4)),
            signals=[Signal(1.0, 5.0)],
            atol=1e-10,
            rtol=1e-10,
            method=self.method,
        )
        self.assertTrue(isinstance(results.y[-1], SuperOp))
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAllClose(results.y[-1].data, np.kron(X, X))

    def test_hamiltonian_lindblad_SuperOp_consistency(self):
        """Test Hamiltonian-based SuperOp simulation."""
        results = self.ham_solver.solve(
            t_span=[0.0, 0.432],
            y0=SuperOp(np.eye(4)),
            signals=[Signal(1.0, 5.0)],
            atol=1e-10,
            rtol=1e-10,
            method=self.method,
        )
        results2 = self.vec_lindblad_solver_no_diss.solve(
            t_span=[0.0, 0.432],
            y0=SuperOp(np.eye(4)),
            signals=[Signal(1.0, 5.0)],
            atol=1e-10,
            rtol=1e-10,
        )
        self.assertAllClose(results.y[-1].data, results2.y[-1].data)

    def test_lindblad_solver_consistency(self):
        """Test consistency of lindblad solver with dissipators specified
        as constant v.s. non-constant.
        """
        lindblad_solver2 = Solver(
            hamiltonian_operators=[self.X],
            dissipator_operators=[self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
        )

        results = lindblad_solver2.solve(
            t_span=[0.0, 1.0],
            y0=Statevector([0.0, 1.0]),
            signals=([Signal(1.0, 5.0)], [0.01**2]),
            method=self.method,
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(results.y[-1].data[0, 0] > 0.99 and results.y[-1].data[0, 0] < 0.999)


class TestSolverSimulationJax(TestSolverSimulation, TestJaxBase):
    """JAX version of TestSolverSimulation."""

    def setUp(self):
        """Set method to 'jax_odeint' to speed up running of jax version of tests."""
        super().setUp()
        self.method = "jax_odeint"

    def test_transform_through_construction_when_validate_false(self):
        """Test that a function building a Solver can be compiled if validate=False."""

        Z = to_array(self.Z)
        X = to_array(self.X)

        def func(a):
            solver = Solver(
                static_hamiltonian=5 * Z,
                hamiltonian_operators=[X],
                rotating_frame=5 * Z,
                validate=False,
            )
            yf = solver.solve(
                t_span=np.array([0.0, 0.1]),
                y0=np.array([0.0, 1.0]),
                signals=[Signal(Array(a), 5.0)],
                method=self.method,
            ).y[-1]
            return yf

        jit_func = self.jit_wrap(func)
        self.assertAllClose(jit_func(2.0), func(2.0))

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)

    def test_jit_solve(self):
        """Test jitting setting signals and solving."""

        def func(a):
            yf = self.ham_solver.solve(
                t_span=np.array([0.0, 1.0]),
                y0=np.array([0.0, 1.0]),
                signals=[Signal(lambda t: a, 5.0)],
                method=self.method,
            ).y[-1]
            return yf

        jit_func = self.jit_wrap(func)
        self.assertAllClose(jit_func(2.0), func(2.0))

    def test_jit_grad_solve(self):
        """Test jitting setting signals and solving."""

        X = Operator.from_label("X")
        solver = Solver(hamiltonian_operators=[X], dissipator_operators=[X])

        def func(a):
            yf = solver.solve(
                t_span=[0.0, 1.0],
                y0=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=complex),
                signals=([Signal(lambda t: a, 5.0)], [1.0]),
                method=self.method,
            ).y[-1]
            return yf

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)


@ddt
class TestSolverListSimulation(QiskitDynamicsTestCase):
    """Test cases for Solver class using list simulation."""

    def setUp(self):
        """Set up some simple models."""
        X = 2 * np.pi * Operator.from_label("X") / 2
        Z = 2 * np.pi * Operator.from_label("Z") / 2
        self.X = X
        self.Z = Z

        self.static_ham_solver = Solver(
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.ham_solver = Solver(
            hamiltonian_operators=[X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.static_lindblad_solver = Solver(
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.lindblad_solver = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.td_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[0.01 * X],
            dissipator_operators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

    @unpack
    @data(("static_ham_solver",), ("static_lindblad_solver",))
    def test_static_solver(self, model):
        """Test doing lists of simulations for a solver with only static information."""

        model = "static_lindblad_solver"
        solver = getattr(self, model)

        y0 = [Statevector([0.0, 1.0]), Statevector([1.0, 0.0])]
        t_span = [0.0, 0.4232]

        results = solver.solve(t_span=t_span, y0=y0)

        res0 = solver.solve(t_span=t_span, y0=y0[0])
        res1 = solver.solve(t_span=t_span, y0=y0[1])

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])

    @unpack
    @data(("ham_solver",), ("lindblad_solver",))
    def test_list_hamiltonian_sim_case1(self, model):
        """Test doing lists of simulations for a solvers whose time dependence only
        comes from the Hamiltonian part.
        """

        solver = getattr(self, model)

        y0 = [Statevector([0.0, 1.0]), Statevector([1.0, 0.0])]
        t_span = [0.0, 0.4232]
        signals = [Signal(1.0, 5.0)]

        results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        res0 = solver.solve(t_span=t_span, y0=y0[0], signals=signals)
        res1 = solver.solve(t_span=t_span, y0=y0[1], signals=signals)

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])

    @unpack
    @data(("ham_solver",), ("lindblad_solver",))
    def test_list_hamiltonian_sim_case2(self, model):
        """Test doing lists of simulations for a solvers whose time dependence only
        comes from the Hamiltonian part.
        """

        solver = getattr(self, model)

        y0 = [Statevector([0.0, 1.0]), Statevector([1.0, 0.0])]
        t_span = [0.0, 0.4232]
        signals = [[Signal(1.0, 5.0)], [Signal(0.5, 5.0)]]

        results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        res0 = solver.solve(t_span=t_span, y0=y0[0], signals=signals[0])
        res1 = solver.solve(t_span=t_span, y0=y0[1], signals=signals[1])

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])

    @unpack
    @data(("ham_solver",), ("lindblad_solver",))
    def test_list_hamiltonian_sim_case3(self, model):
        """Test doing lists of simulations for a solvers whose time dependence only
        comes from the Hamiltonian part.
        """

        solver = getattr(self, model)

        y0 = [Statevector([0.0, 1.0]), Statevector([1.0, 0.0])]
        t_span = [[0.0, 0.4232], [0.0, 1.23]]
        signals = [Signal(1.0, 5.0)]

        results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        res0 = solver.solve(t_span=t_span[0], y0=y0[0], signals=signals)
        res1 = solver.solve(t_span=t_span[1], y0=y0[1], signals=signals)

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])

    @unpack
    @data(("ham_solver",), ("lindblad_solver",))
    def test_list_hamiltonian_sim_case4(self, model):
        """Test doing lists of simulations for a solvers whose time dependence only
        comes from the Hamiltonian part.
        """

        solver = getattr(self, model)

        y0 = Statevector([0.0, 1.0])
        t_span = [0.0, 0.4232]
        signals = [[Signal(1.0, 5.0)], [Signal(0.5, 5.0)]]

        results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        res0 = solver.solve(t_span=t_span, y0=y0, signals=signals[0])
        res1 = solver.solve(t_span=t_span, y0=y0, signals=signals[1])

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])

    def test_list_td_lindblad_sim_case1(self):
        """Test doing lists of simulations for solver with time-dependent dissipators."""

        solver = self.td_lindblad_solver

        y0 = [Statevector([0.0, 1.0]), Statevector([1.0, 0.0])]
        t_span = [0.0, 0.4232]
        signals = ([Signal(1.0, 5.0)], [1.0])

        results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        res0 = solver.solve(t_span=t_span, y0=y0[0], signals=signals)
        res1 = solver.solve(t_span=t_span, y0=y0[1], signals=signals)

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])

    def test_list_td_lindblad_sim_case2(self):
        """Test doing lists of simulations for solver with time-dependent dissipators."""

        solver = self.td_lindblad_solver

        y0 = [Statevector([0.0, 1.0]), Statevector([1.0, 0.0])]
        t_span = [0.0, 0.4232]
        signals = [([Signal(1.0, 5.0)], [1.0]), ([Signal(0.5, 5.0)], [2.0])]

        results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        res0 = solver.solve(t_span=t_span, y0=y0[0], signals=signals[0])
        res1 = solver.solve(t_span=t_span, y0=y0[1], signals=signals[1])

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])

    def test_list_td_lindblad_sim_case3(self):
        """Test doing lists of simulations for solver with time-dependent dissipators."""

        solver = self.td_lindblad_solver

        y0 = [Statevector([0.0, 1.0]), Statevector([1.0, 0.0])]
        t_span = [[0.0, 0.4232], [0.0, 1.23]]
        signals = ([Signal(1.0, 5.0)], [1.0])

        results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        res0 = solver.solve(t_span=t_span[0], y0=y0[0], signals=signals)
        res1 = solver.solve(t_span=t_span[1], y0=y0[1], signals=signals)

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])

    def test_list_td_lindblad_sim_case4(self):
        """Test doing lists of simulations for solver with time-dependent dissipators."""

        solver = self.td_lindblad_solver

        y0 = Statevector([0.0, 1.0])
        t_span = [0.0, 0.4232]
        signals = [([Signal(1.0, 5.0)], [1.0]), ([Signal(0.5, 5.0)], [2.0])]

        results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        res0 = solver.solve(t_span=t_span, y0=y0, signals=signals[0])
        res1 = solver.solve(t_span=t_span, y0=y0, signals=signals[1])

        self.assertAllClose(results[0].y[-1], res0.y[-1])
        self.assertAllClose(results[1].y[-1], res1.y[-1])
