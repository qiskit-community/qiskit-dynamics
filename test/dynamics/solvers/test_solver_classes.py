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
# pylint: disable=invalid-name,no-member

"""
Tests for solver classes module.
"""

from functools import partial

import numpy as np
import sympy as sym
from ddt import ddt, data, unpack

from qiskit import pulse, QiskitError
from qiskit.quantum_info import Operator, Statevector, SuperOp, DensityMatrix

from qiskit_dynamics import Solver, Signal, DiscreteSignal, solve_lmde
from qiskit_dynamics.models import HamiltonianModel, LindbladModel, rotating_wave_approximation
from qiskit_dynamics.solvers.solver_classes import organize_signals_to_channels

from ..common import QiskitDynamicsTestCase, test_array_backends

try:
    from jax import jit
except ImportError:
    pass


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


class TestPulseSolverValidation(QiskitDynamicsTestCase):
    """Tests for validation when constructing with pulse simulation parameters."""

    def setUp(self):
        self.X = Operator.from_label("X")

    def test_channels_len(self):
        """Test hamiltonian_channels or dissipator_channels length error."""

        error_str = "hamiltonian_channels must have same length"
        with self.assertRaisesRegex(QiskitError, error_str):
            Solver(hamiltonian_operators=[self.X] * 2, hamiltonian_channels=["d0"])

        with self.assertRaisesRegex(QiskitError, error_str):
            Solver(hamiltonian_operators=None, hamiltonian_channels=["d0"])

        error_str = "dissipator_channels must have same length"
        with self.assertRaisesRegex(QiskitError, error_str):
            Solver(dissipator_operators=[self.X] * 2, dissipator_channels=["d0"])

        with self.assertRaisesRegex(QiskitError, error_str):
            Solver(dissipator_operators=None, dissipator_channels=["d0"])

    def test_channel_carrier_freq_not_specified(self):
        """Test cases when a channel carrier freq not specified."""

        error_str = "Channel 'd0' does not have carrier frequency specified"
        with self.assertRaisesRegex(QiskitError, error_str):
            Solver(hamiltonian_operators=[self.X], hamiltonian_channels=["d0"])

        with self.assertRaisesRegex(QiskitError, error_str):
            Solver(
                hamiltonian_operators=[self.X],
                hamiltonian_channels=["d0"],
                channel_carrier_freqs={"d1": 1.0},
            )

        with self.assertRaisesRegex(QiskitError, error_str):
            Solver(
                dissipator_operators=[self.X],
                dissipator_channels=["d0"],
                channel_carrier_freqs={"d1": 1.0},
            )

    def test_dt_not_specified(self):
        """Test error if sufficient channel information provided but dt not specified."""

        with self.assertRaisesRegex(QiskitError, "dt must be specified"):
            Solver(
                dissipator_operators=[self.X],
                dissipator_channels=["d0"],
                channel_carrier_freqs={"d0": 1.0},
            )

    def test_solve_not_pulse_configured(self):
        """Test error raised if Schedule passed to solve but Solver not configured for
        pulse simulation.
        """

        solver = Solver(static_hamiltonian=self.X)

        with pulse.build() as sched:
            pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))

        with self.assertRaisesRegex(QiskitError, "not configured for pulse"):
            solver.solve(t_span=[0.0, 1.0], y0=np.array([0.0, 1.0]), signals=sched)


class Testorganize_signals_to_channels(QiskitDynamicsTestCase):
    """Test helper function organize_signals_to_channels."""

    def test_hamiltonian_model(self):
        """Test for HamiltonianModel case."""
        output = organize_signals_to_channels(
            all_signals=["a", "b", "c", "d"],
            all_channels=["d0", "d1", "d2", "d3"],
            model_class=HamiltonianModel,
            hamiltonian_channels=["d1", "d2", "d0", "d3"],
            dissipator_channels=None,
        )

        self.assertTrue(output == ["b", "c", "a", "d"])

    def test_lindblad_model(self):
        """Test for LindbladModel case."""
        output = organize_signals_to_channels(
            all_signals=["a", "b", "c", "d"],
            all_channels=["d0", "d1", "d2", "d3"],
            model_class=LindbladModel,
            hamiltonian_channels=["d1", "d2"],
            dissipator_channels=["d0", "d3"],
        )

        self.assertTrue(output == (["b", "c"], ["a", "d"]))


class TestSolverExceptions(QiskitDynamicsTestCase):
    """Tests for Solver exception raising based on input types."""

    def setUp(self):
        X = Operator.from_label("X")
        self.ham_solver = Solver(hamiltonian_operators=[X])

        self.lindblad_solver = Solver(hamiltonian_operators=[X], static_dissipators=[X])

        self.vec_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[X],
            vectorized=True,
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

        res1 = ham_solver.solve(t_span=t_span, y0=y0, signals=signals, atol=1e-11, rtol=1e-11)

        self.ham_model.signals = signals
        res2 = solve_lmde(generator=self.ham_model, t_span=t_span, y0=y0, atol=1e-11, rtol=1e-11)

        self.assertAllClose(res1.y[-1], res2.y[-1], atol=1e-8, rtol=1e-8)

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

        res1 = lindblad_solver.solve(t_span=t_span, y0=y0, signals=signals, atol=1e-12, rtol=1e-12)

        self.lindblad_model.signals = (signals, None)
        res2 = solve_lmde(
            generator=self.lindblad_model, t_span=t_span, y0=y0, atol=1e-12, rtol=1e-12
        )

        self.assertAllClose(res1.y[-1], res2.y[-1], atol=1e-8, rtol=1e-8)

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

        res1 = td_lindblad_solver.solve(
            t_span=t_span, y0=y0, signals=signals, atol=1e-12, rtol=1e-12
        )

        self.td_lindblad_model.signals = signals
        res2 = solve_lmde(
            generator=self.td_lindblad_model, t_span=t_span, y0=y0, atol=1e-12, rtol=1e-12
        )

        self.assertAllClose(res1.y[-1], res2.y[-1], atol=1e-8, rtol=1e-8)

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

        res1 = rwa_ham_solver.solve(t_span=t_span, y0=y0, signals=signals, atol=1e-12, rtol=1e-12)

        self.ham_model.signals = signals
        rwa_ham_model = rotating_wave_approximation(self.ham_model, cutoff_freq=5.0)
        res2 = solve_lmde(generator=rwa_ham_model, t_span=t_span, y0=y0, atol=1e-12, rtol=1e-12)

        self.assertAllClose(res1.y[-1], res2.y[-1], atol=1e-8, rtol=1e-8)

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

        res1 = rwa_lindblad_solver.solve(
            t_span=t_span, y0=y0, signals=signals, atol=1e-12, rtol=1e-12
        )

        self.lindblad_model.signals = (signals, None)
        rwa_lindblad_model = rotating_wave_approximation(self.lindblad_model, cutoff_freq=5.0)
        res2 = solve_lmde(
            generator=rwa_lindblad_model, t_span=t_span, y0=y0, atol=1e-12, rtol=1e-12
        )

        self.assertAllClose(res1.y[-1], res2.y[-1], atol=1e-8, rtol=1e-8)

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

        res1 = rwa_td_lindblad_solver.solve(
            t_span=t_span, y0=y0, signals=signals, atol=1e-12, rtol=1e-12
        )

        self.td_lindblad_model.signals = signals
        rwa_td_lindblad_model = rotating_wave_approximation(self.td_lindblad_model, cutoff_freq=5.0)
        res2 = solve_lmde(
            generator=rwa_td_lindblad_model, t_span=t_span, y0=y0, atol=1e-12, rtol=1e-12
        )

        self.assertAllClose(res1.y[-1], res2.y[-1], atol=1e-8, rtol=1e-8)

    def test_signals_are_None(self):
        """Test the model signals return to being None after simulation."""
        ham_solver = Solver(hamiltonian_operators=[self.X])
        ham_solver.solve(signals=[1.0], t_span=[0.0, 0.01], y0=np.array([0.0, 1.0], dtype=complex))
        self.assertTrue(ham_solver.model.signals is None)

        lindblad_solver = Solver(hamiltonian_operators=[self.X], static_dissipators=[self.X])
        lindblad_solver.solve(signals=[1.0], t_span=[0.0, 0.01], y0=np.eye(2, dtype=complex))
        self.assertTrue(lindblad_solver.model.signals == (None, None))

        td_lindblad_solver = Solver(hamiltonian_operators=[self.X], dissipator_operators=[self.X])
        td_lindblad_solver.solve(
            signals=([1.0], [1.0]), t_span=[0.0, 0.01], y0=np.eye(2, dtype=complex)
        )
        self.assertTrue(td_lindblad_solver.model.signals == (None, None))


@partial(test_array_backends, array_libraries=["numpy", "jax"])
class TestSolverSimulation:
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
            vectorized=True,
        )

        # lindblad solver with no dissipation for testing
        self.vec_lindblad_solver_no_diss = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[0.0 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            vectorized=True,
        )

        if self.array_library() == "numpy":
            self.method = "DOP853"
        elif self.array_library() == "jax":
            self.method = "jax_odeint"

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
        solver = Solver(static_dissipators=np.zeros((1, 6, 6)), vectorized=True)
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
            atol=1e-12,
            rtol=1e-12,
        )
        results2 = self.lindblad_solver.solve(
            t_span=[0.0, 1.0],
            y0=Statevector([0.0, 1.0]),
            signals=[Signal(1.0, 5.0)],
            method=self.method,
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertAllClose(results.y[-1].data, results2.y[-1].data, atol=1e-8, rtol=1e-8)

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

    def test_hamiltonian_array(self):
        """Test correct simulation of an array for Hamiltonian-based simulation."""
        results = self.ham_solver.solve(
            t_span=[0.0, 1.0],
            y0=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
            signals=[Signal(1.0, 5.0)],
            atol=1e-10,
            rtol=1e-10,
            method=self.method,
        )
        self.assertTrue(
            (np.abs(results.y[-1][0, 1]) > 0.999) and (np.abs(results.y[-1][1, 0]) > 0.999)
        )

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
            atol=1e-12,
            rtol=1e-12,
            method=self.method,
        )
        results2 = self.vec_lindblad_solver_no_diss.solve(
            t_span=[0.0, 0.432],
            y0=SuperOp(np.eye(4)),
            signals=[Signal(1.0, 5.0)],
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertAllClose(results.y[-1].data, results2.y[-1].data, atol=1e-8, rtol=1e-8)

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


@partial(test_array_backends, array_libraries=["jax", "jax_sparse"])
class TestSolverSimulationJAXTransformations:
    """Test Solver class within JAX transformations."""

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
            array_library=self.array_library(),
        )

    def test_jit_solve(self):
        """Test jitting setting signals and solving."""

        def func(a):
            yf = self.ham_solver.solve(
                t_span=np.array([0.0, 1.0]),
                y0=np.array([0.0, 1.0], dtype=complex),
                signals=[Signal(lambda t: a, 5.0)],
                method="jax_odeint",
            ).y[-1]
            return yf

        self.assertAllClose(jit(func)(2.0), func(2.0))

    def test_jit_grad_solve(self):
        """Test jitting setting signals and solving."""

        X = Operator.from_label("X")
        solver = Solver(hamiltonian_operators=[X], dissipator_operators=[X])

        def func(a):
            yf = solver.solve(
                t_span=[0.0, 1.0],
                y0=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=complex),
                signals=([Signal(lambda t: a, 5.0)], [1.0]),
                method="jax_odeint",
            ).y[-1]
            return yf

        self.jit_grad(func)(1.0)


@partial(test_array_backends, array_libraries=["jax"])
class TestSolverConstructionJAXTransformations:
    """Test construction of Solver within a function to be JAX transformed."""

    def test_transform_through_construction_when_validate_false(self):
        """Test that a function building a Solver can be compiled if validate=False."""

        Z = np.array([[1.0, 0.0], [0.0, -1.0]])
        X = np.array([[0.0, 1.0], [1.0, 0.0]])

        def func(a):
            solver = Solver(
                static_hamiltonian=5 * Z,
                hamiltonian_operators=[X],
                rotating_frame=5 * Z,
                validate=False,
                array_library=self.array_library(),
            )
            yf = solver.solve(
                t_span=np.array([0.0, 0.1]),
                y0=np.array([0.0, 1.0]),
                signals=[Signal(a, 5.0)],
                method="jax_odeint",
            ).y[-1]
            return yf

        self.assertAllClose(jit(func)(2.0), func(2.0))

        # validate that grad can be compiled and evaluated
        self.jit_grad(func)(1.0)


@partial(test_array_backends, array_libraries=["numpy", "jax"])
@ddt
class TestPulseSimulation:
    """Test simulation of pulse schedules."""

    def setUp(self):
        """Set up some simple models."""
        X = 2 * np.pi * Operator.from_label("X") / 2
        Z = 2 * np.pi * Operator.from_label("Z") / 2
        self.X = X
        self.Z = Z

        self.static_ham_solver = Solver(
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            dt=0.1,
            array_library=self.array_library(),
        )

        self.ham_solver = Solver(
            hamiltonian_operators=[X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 5.0},
            dt=0.1,
            array_library=self.array_library(),
        )

        self.static_lindblad_solver = Solver(
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            dt=0.1,
            array_library=self.array_library(),
        )

        self.lindblad_solver = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 5.0},
            dt=0.1,
            array_library=self.array_library(),
        )

        self.ham_solver_2_channels = Solver(
            hamiltonian_operators=[X, Z],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            hamiltonian_channels=["d0", "d1"],
            channel_carrier_freqs={"d0": 5.0, "d1": 3.1},
            dt=0.1,
            array_library=self.array_library(),
        )

        self.td_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            static_dissipators=[0.01 * X],
            dissipator_operators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=np.diag(5 * Z),
            hamiltonian_channels=["d0"],
            dissipator_channels=["d1"],
            channel_carrier_freqs={"d0": 5.0, "d1": 3.1},
            dt=0.1,
            array_library=self.array_library(),
            vectorized=True,
        )

        if self.array_library() == "numpy":
            self.method = "DOP853"
        elif self.array_library() == "jax":
            self.method = "jax_odeint"

    @unpack
    @data(("static_ham_solver",), ("static_lindblad_solver",))
    def test_static_simulation(self, model):
        """Test pulse simulation with a static model."""

        # construct schedule
        with pulse.build() as sched:
            pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))

        self._compare_schedule_to_signals(
            solver=getattr(self, model),
            t_span=[0.0, 1.0],
            y0=Statevector([0.0, 1.0]),
            schedules=sched,
            signals=None,
            test_tol=1e-8,
            atol=1e-11,
            rtol=1e-11,
        )

    @unpack
    @data(("ham_solver",), ("lindblad_solver",))
    def test_one_channel_simulation(self, model):
        """Test pulse simulation with models with one channel."""

        # construct schedule
        with pulse.build() as sched:
            pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))
            pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))

        # construct equivalent DiscreteSignal manually
        constant_samples = np.ones(5, dtype=float) * 0.9
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        samples = np.append(constant_samples, gauss_samples * phase)
        sig = DiscreteSignal(dt=0.1, samples=samples, carrier_freq=5.0)

        self._compare_schedule_to_signals(
            solver=getattr(self, model),
            t_span=[0.0, 1.0],
            y0=Statevector([1.0, 0.0]),
            schedules=sched,
            signals=[sig],
            test_tol=1e-8,
            atol=1e-11,
            rtol=1e-11,
        )

    @unpack
    @data(("ham_solver_2_channels",), ("td_lindblad_solver",))
    def test_two_channel_list_simulation(self, model):
        """Test pulse simulation with models with two channels, with solve arguments specified as
        a list.
        """

        # construct schedule0
        with pulse.build() as sched0:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(1))

        # construct equivalent DiscreteSignal manually
        constant_samples = np.ones(5, dtype=float) * 0.9
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        samples0 = np.append(np.append(constant_samples, gauss_samples * phase), np.zeros(5))
        samples1 = np.append(np.zeros(10), gauss_samples)
        sig00 = DiscreteSignal(dt=0.1, samples=samples0, carrier_freq=5.0)
        sig01 = DiscreteSignal(dt=0.1, samples=samples1, carrier_freq=3.1)

        # construct schedule1
        with pulse.build() as sched1:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.8), pulse.DriveChannel(0))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.973, sigma=1.0), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.973, sigma=1.0), pulse.DriveChannel(1))

        # construct equivalent DiscreteSignal manually
        constant_samples = np.ones(5, dtype=float) * 0.8
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.973, sigma=1.0).get_waveform().samples
        samples0 = np.append(np.append(constant_samples, gauss_samples * phase), np.zeros(5))
        samples1 = np.append(np.zeros(10), gauss_samples)
        sig10 = DiscreteSignal(dt=0.1, samples=samples0, carrier_freq=5.0)
        sig11 = DiscreteSignal(dt=0.1, samples=samples1, carrier_freq=3.1)

        signals = None
        if "lindblad" in model:
            signals = [([sig00], [sig01]), ([sig10], [sig11])]
        else:
            signals = [[sig00, sig01], [sig10, sig11]]

        self._compare_schedule_to_signals(
            solver=getattr(self, model),
            t_span=[0.0, 1.5],
            y0=[Statevector([1.0, 0.0]), DensityMatrix([0.0, 1.0])],
            schedules=[sched0, sched1],
            signals=signals,
            test_tol=1e-8,
            atol=1e-12,
            rtol=1e-12,
        )

    @unpack
    @data(("ham_solver_2_channels",), ("td_lindblad_solver",))
    def test_two_channel_SuperOp_simulation(self, model):
        """Test pulse simulation with models with two channels when simulating a SuperOp."""

        # construct schedule
        with pulse.build() as sched:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(1))

        # construct equivalent DiscreteSignal manually
        constant_samples = np.ones(5, dtype=float) * 0.9
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        samples0 = np.append(np.append(constant_samples, gauss_samples * phase), np.zeros(5))
        samples1 = np.append(np.zeros(10), gauss_samples)
        sig0 = DiscreteSignal(dt=0.1, samples=samples0, carrier_freq=5.0)
        sig1 = DiscreteSignal(dt=0.1, samples=samples1, carrier_freq=3.1)

        signals = None
        if "lindblad" in model:
            signals = ([sig0], [sig1])
        else:
            signals = [sig0, sig1]

        self._compare_schedule_to_signals(
            solver=getattr(self, model),
            t_span=[0.0, 1.5],
            y0=SuperOp(np.eye(4, dtype=complex)),
            schedules=sched,
            signals=signals,
            test_tol=1e-8,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_4_channel_schedule(self):
        """Test Solver with 4 channels."""

        dt = 0.05
        big_solver = Solver(
            hamiltonian_operators=[self.X, self.Z],
            static_dissipators=[0.01 * self.X],
            dissipator_operators=[0.01 * self.X, self.Z],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
            hamiltonian_channels=["d0", "d2"],
            dissipator_channels=["d1", "d3"],
            channel_carrier_freqs={"d0": 5.0, "d1": 3.1, "d2": 0, "d3": 4.0},
            dt=dt,
            array_library=self.array_library(),
        )

        with pulse.build() as schedule:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(1))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(2))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(3))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))

        # construct samples
        constant_samples = np.ones(5, dtype=float) * 0.9
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        samples0 = np.append(np.append(constant_samples, gauss_samples * phase), np.zeros(15))
        phase2 = np.exp(2 * 1j * np.pi / 2.98)
        samples0 = np.append(samples0, gauss_samples * phase2)
        samples1 = np.append(np.append(np.zeros(10), gauss_samples), np.zeros(15))
        samples2 = np.append(np.zeros(15), np.append(gauss_samples, np.zeros(10)))
        samples3 = np.append(np.zeros(20), np.append(gauss_samples, np.zeros(5)))
        sig0 = DiscreteSignal(dt=dt, samples=samples0, carrier_freq=5.0)
        sig1 = DiscreteSignal(dt=dt, samples=samples1, carrier_freq=3.1)
        sig2 = DiscreteSignal(dt=dt, samples=samples2, carrier_freq=0.0)
        sig3 = DiscreteSignal(dt=dt, samples=samples3, carrier_freq=4.0)

        signals = ([sig0, sig2], [sig1, sig3])

        self._compare_schedule_to_signals(
            solver=big_solver,
            t_span=[0.0, 30 * dt],
            y0=Statevector([1.0, 0.0]),
            schedules=schedule,
            signals=signals,
            test_tol=1e-8,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_rwa_ham_solver(self):
        """Test RWA for pulse solver with hamiltonian information."""

        ham_pulse_solver = Solver(
            hamiltonian_operators=[self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 5.0},
            dt=0.1,
            rwa_cutoff_freq=1.5 * 5.0,
            array_library=self.array_library(),
        )

        ham_solver = Solver(
            hamiltonian_operators=[self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
            rwa_cutoff_freq=1.5 * 5.0,
            rwa_carrier_freqs=[5.0],
            array_library=self.array_library(),
        )

        with pulse.build() as schedule:
            pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))

        sig = Signal(0.9, carrier_freq=5.0)

        res_pulse = ham_pulse_solver.solve(
            t_span=[0, 0.4],
            y0=Statevector([0.0, 1.0]),
            signals=schedule,
            method=self.method,
            atol=1e-12,
            rtol=1e-12,
        )
        res_signal = ham_solver.solve(
            t_span=[0, 0.4],
            y0=Statevector([0.0, 1.0]),
            signals=[sig],
            method=self.method,
            atol=1e-12,
            rtol=1e-12,
        )

        self.assertAllClose(res_pulse.y[-1], res_signal.y[-1], atol=1e-8, rtol=1e-8)

    def test_rwa_lindblad_solver(self):
        """Test RWA for pulse solver with Lindblad information."""

        lindblad_pulse_solver = Solver(
            hamiltonian_operators=[self.X],
            static_dissipators=[0.01 * self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 5.0},
            dt=0.1,
            rwa_cutoff_freq=1.5 * 5.0,
            array_library=self.array_library(),
        )

        lindblad_solver = Solver(
            hamiltonian_operators=[self.X],
            static_dissipators=[0.01 * self.X],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
            rwa_cutoff_freq=1.5 * 5.0,
            rwa_carrier_freqs=[5.0],
            array_library=self.array_library(),
        )

        with pulse.build() as schedule:
            pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))

        sig = Signal(0.9, carrier_freq=5.0)

        res_pulse = lindblad_pulse_solver.solve(
            t_span=[0, 0.4],
            y0=Statevector([0.0, 1.0]),
            signals=schedule,
            method=self.method,
            convert_results=False,
            atol=1e-12,
            rtol=1e-12,
        )
        res_signal = lindblad_solver.solve(
            t_span=[0, 0.4],
            y0=Statevector([0.0, 1.0]),
            signals=[sig],
            method=self.method,
            convert_results=False,
            atol=1e-12,
            rtol=1e-12,
        )

        self.assertAllClose(res_pulse.y[-1], res_signal.y[-1], atol=1e-8, rtol=1e-8)

    def test_list_simulation_mixing_types(self):
        """Test correct formatting when input states have the same shape.

        This catches an edge case bug that occurred during implementation.
        It passes two initial states with different type handling but with the same shape.
        """

        # construct schedule0
        with pulse.build() as sched0:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(1))

        # construct equivalent DiscreteSignal manually
        constant_samples = np.ones(5, dtype=float) * 0.9
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        samples0 = np.append(np.append(constant_samples, gauss_samples * phase), np.zeros(5))
        samples1 = np.append(np.zeros(10), gauss_samples)
        sig00 = DiscreteSignal(dt=0.1, samples=samples0, carrier_freq=5.0)
        sig01 = DiscreteSignal(dt=0.1, samples=samples1, carrier_freq=3.1)

        # construct schedule1
        with pulse.build() as sched1:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.8), pulse.DriveChannel(0))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.973, sigma=1.0), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.973, sigma=1.0), pulse.DriveChannel(1))

        # construct equivalent DiscreteSignal manually
        constant_samples = np.ones(5, dtype=float) * 0.8
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.973, sigma=1.0).get_waveform().samples
        samples0 = np.append(np.append(constant_samples, gauss_samples * phase), np.zeros(5))
        samples1 = np.append(np.zeros(10), gauss_samples)
        sig10 = DiscreteSignal(dt=0.1, samples=samples0, carrier_freq=5.0)
        sig11 = DiscreteSignal(dt=0.1, samples=samples1, carrier_freq=3.1)

        signals = [[sig00, sig01], [sig10, sig11]]

        self._compare_schedule_to_signals(
            solver=self.ham_solver_2_channels,
            t_span=[0.0, 1.5],
            y0=[np.eye(2, dtype=complex), DensityMatrix([0.0, 1.0])],
            schedules=[sched0, sched1],
            signals=signals,
            test_tol=1e-8,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_channel_without_instructions(self):
        """Test pulse simulation with a channel in the model having no instructions."""

        # construct schedule
        with pulse.build() as sched:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))

        # construct equivalent DiscreteSignal manually
        sig0 = DiscreteSignal(dt=0.1, samples=np.ones(5, dtype=float) * 0.9, carrier_freq=5.0)
        sig1 = DiscreteSignal(dt=0.1, samples=[], carrier_freq=3.1)
        signals = [sig0, sig1]

        self._compare_schedule_to_signals(
            solver=self.ham_solver_2_channels,
            t_span=[0.0, 0.5],
            y0=np.eye(2),
            schedules=sched,
            signals=signals,
            test_tol=1e-8,
            atol=1e-11,
            rtol=1e-11,
        )

    def _compare_schedule_to_signals(
        self, solver, t_span, y0, schedules, signals, test_tol=1e-14, **kwargs
    ):
        """Run comparison of schedule simulation to manually build signals.

        The original expectation of this function was that schedules and signals should represent
        exactly the same time-dependence, so that the solutions agree exactly regardless of
        tolerance. Nevertheless, the test_tol argument enables modification of the tolerance
        of the tests here, in situations in which exact correspondence of time-dependence
        cannot be gauranteed (even machine-epsilon rounding errors based on order of operations
        is enough to break this assumption in practical simulation settings).
        """

        pulse_results = solver.solve(
            t_span=t_span,
            y0=y0,
            signals=schedules,
            convert_results=False,
            method=self.method,
            **kwargs,
        )
        signal_results = solver.solve(
            t_span=t_span,
            y0=y0,
            signals=signals,
            convert_results=False,
            method=self.method,
            **kwargs,
        )

        if not isinstance(pulse_results, list):
            pulse_results = [pulse_results]
            signal_results = [signal_results]

        for pulse_res, signal_res in zip(pulse_results, signal_results):
            self.assertAllClose(pulse_res.y[-1], signal_res.y[-1], atol=test_tol, rtol=test_tol)


@partial(test_array_backends, array_libraries=["jax"])
class TestPulseSimulationJAXPeculiarities:
    """Test class for technical issues of pulse simulation with JAX."""

    def setUp(self):
        """Set up some simple models."""
        X = 2 * np.pi * Operator.from_label("X") / 2
        Z = 2 * np.pi * Operator.from_label("Z") / 2

        self.ham_solver = Solver(
            hamiltonian_operators=[X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 5.0},
            dt=0.1,
            array_library=self.array_library(),
        )

    def test_t_eval_t_span_jax_odeint(self):
        """Test internal jitting works when specifying t_eval and t_span. This catches a bug
        that was missed on initial implementation. This test just checks that it runs,
        with the correctness assumed to be verified elsewhere.
        """

        with pulse.build() as sched:
            pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))

        self.ham_solver.solve(
            signals=sched,
            t_span=[0.0, 0.1],
            y0=np.array([0.0, 1.0]),
            t_eval=[0.0, 0.05, 0.1],
            method="jax_odeint",
        )

    def test_t_eval_t_span_diffrax(self):
        """Test internal jitting works when specifying t_eval and t_span for diffrax.
        This catches a bug that was missed on initial implementation. This test just checks
        that it runs, with the correctness assumed to be verified elsewhere.
        """

        from diffrax import Dopri5, PIDController

        with pulse.build() as sched:
            pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))

        self.ham_solver.solve(
            signals=sched,
            t_span=[0.0, 0.1],
            y0=np.array([0.0, 1.0]),
            t_eval=[0.0, 0.05, 0.1],
            method=Dopri5(),
            stepsize_controller=PIDController(rtol=1e-10, atol=1e-10),
            max_steps=1000000,
        )

    def test_t_eval_t_span_jax_expm(self):
        """Test internal jitting works when specifying t_eval and t_span for jax_expm,
        which serves as a place-holder for internally developed fixed-step solvers.
        This catches a bug that was missed on initial implementation. This test just checks
        that it runs, with the correctness assumed to be verified elsewhere.
        """

        with pulse.build() as sched:
            pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))

        self.ham_solver.solve(
            signals=sched,
            t_span=[0.0, 0.1],
            y0=np.array([0.0, 1.0]),
            t_eval=[0.0, 0.05, 0.1],
            method="jax_expm",
            max_dt=0.05,
        )

    def test_jit_solve_with_internal_jit(self):
        """Test jitting solver with internal jitting works.
        Internal jitting should be avoided when using jitting.
        This test checks that using _solve_list not _solve_schedule_list_jax
        when jitting solvers.solve.
        """

        def constant_pulse(amp):
            parameters = {"amp": amp}
            _time, _amp, _duration = sym.symbols("t, amp, duration")
            envelope_expr = _amp * sym.Piecewise(
                (1, sym.And(_time >= 0, _time <= _duration)), (0, True)
            )
            valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0
            # we need to set disable_validation True to enable jax-jitting.
            pulse.SymbolicPulse.disable_validation = True
            return pulse.SymbolicPulse(
                pulse_type="Constant",
                duration=5,
                parameters=parameters,
                envelope=envelope_expr,
                valid_amp_conditions=valid_amp_conditions_expr,
            )

        def func(amp):
            with pulse.build() as sched:
                pulse.play(constant_pulse(amp), pulse.DriveChannel(0))

            self.ham_solver.solve(
                signals=sched,
                t_span=[0.0, 0.1],
                y0=np.array([0.0, 1.0]),
                method="jax_odeint",
            )

        jit(func)(0.1)


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

        y0 = [np.eye(2, dtype=complex), DensityMatrix([1.0, 0.0])]
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
