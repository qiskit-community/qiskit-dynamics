# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
Test DynamicsBackend.
"""

import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError, pulse, QuantumCircuit
from qiskit.transpiler import Target
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.result.models import ExperimentResult, ExperimentResultData

from qiskit_dynamics import Solver, DynamicsBackend
from qiskit_dynamics.backend import default_experiment_result_function
from ..common import QiskitDynamicsTestCase


class TestDynamicsBackendValidation(QiskitDynamicsTestCase):
    """Test validation checks."""

    def setUp(self):
        """Build simple simulator for multiple tests."""

        solver = Solver(
            static_hamiltonian=np.array([[1.0, 0.0], [0.0, -1.0]]),
            hamiltonian_operators=[np.array([[0.0, 1.0], [1.0, 0.0]])],
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 1.0},
            dt=1.0,
        )

        self.simple_backend = DynamicsBackend(solver=solver)

    def test_solver_not_configured_for_pulse(self):
        """Test error is raised if solver not configured for pulse simulation."""

        solver = Solver(
            static_hamiltonian=np.array([[1.0, 0.0], [0.0, -1.0]]),
            hamiltonian_operators=[np.array([[0.0, 1.0], [1.0, 0.0]])],
        )

        with self.assertRaisesRegex(QiskitError, "not configured for Pulse"):
            DynamicsBackend(solver=solver)

    def test_run_input_error(self):
        """Test submission of invalid run input."""

        with self.assertRaisesRegex(QiskitError, "not supported by DynamicsBackend.run."):
            self.simple_backend.run([1])

    def test_subsystem_dims_inconsistency(self):
        """Test that setting subsystem_dims inconsistently with solver.model.dim raises error."""

        with self.assertRaisesRegex(QiskitError, "inconsistent"):
            self.simple_backend.set_options(subsystem_dims=[4])

    def test_max_outcome_level_error(self):
        """Test that invalid max_outcome_level results in error."""

        with self.assertRaisesRegex(QiskitError, "must be a positive integer"):
            self.simple_backend.set_options(max_outcome_level=0)

        with self.assertRaisesRegex(QiskitError, "must be a positive integer"):
            self.simple_backend.set_options(max_outcome_level="hi")

    def test_no_measurements_in_schedule(self):
        """Test that running a schedule with no measurements raises an error."""

        with pulse.build() as schedule:
            pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))

        with self.assertRaisesRegex(QiskitError, "At least one measurement"):
            self.simple_backend.run(schedule)

    def test_no_measurements_with_memory_slots_in_schedule(self):
        """Test that running a schedule without measurements saving results in a MemorySlot
        raises an error."""

        with pulse.build() as schedule:
            pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
            pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.RegisterSlot(0))

        with self.assertRaisesRegex(QiskitError, "At least one measurement"):
            self.simple_backend.run(schedule)

    def test_multiple_measurements_in_schedule(self):
        """Test error raising when attempting to run a schedule with multiple measurements."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))
                pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        with self.assertRaisesRegex(QiskitError, "only supports measurements at one time"):
            self.simple_backend.run(schedule)

    def test_measure_nonexistant_subsystem(self):
        """Attempt to measure subsystem that doesn't exist."""

        with pulse.build() as schedule:
            pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
            pulse.acquire(duration=1, qubit_or_channel=1, register=pulse.MemorySlot(0))

        with self.assertRaisesRegex(QiskitError, "Attempted to measure subsystem 1"):
            self.simple_backend.run(schedule)

    def test_invalid_initial_state(self):
        """Test setting an invalid initial state."""

        with self.assertRaisesRegex(QiskitError, "initial_state must be either"):
            self.simple_backend.set_options(initial_state=1)

    def test_invalid_meas_level(self):
        """Test setting an invalid meas_level."""

        with self.assertRaisesRegex(QiskitError, "Only meas_level 1 and 2 are supported"):
            self.simple_backend.set_options(meas_level=0)

    def test_invalid_meas_return(self):
        """Test setting an invalid meas_return."""

        with self.assertRaisesRegex(QiskitError, "meas_return must be either 'single' or 'avg'"):
            self.simple_backend.set_options(meas_return="combined")

    def test_invalid_iq_width(self):
        """Test setting an invalid iq_width."""

        with self.assertRaisesRegex(QiskitError, "must be a positive float"):
            self.simple_backend.set_options(iq_width=0)
        with self.assertRaisesRegex(QiskitError, "must be a positive float"):
            self.simple_backend.set_options(iq_width="hi")

    def test_invalid_iq_centers(self):
        """Test setting an invalid iq_centers."""

        with self.assertRaisesRegex(QiskitError, "iq_centers option must be either None or"):
            self.simple_backend.set_options(iq_centers=[[0]])

        with self.assertRaisesRegex(QiskitError, "iq_centers option is not consistent"):
            self.simple_backend.set_options(subsystem_dims=[2])
            self.simple_backend.set_options(iq_centers=[[[1, 0], [0, 1], [1, 1]]])

        with self.assertRaisesRegex(QiskitError, "iq_centers option is not consistent"):
            self.simple_backend.set_options(subsystem_dims=[2])
            self.simple_backend.set_options(iq_centers=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]])

    def test_invalid_experiment_result_function(self):
        """Test setting a non-callable experiment_result_function."""

        with self.assertRaisesRegex(QiskitError, "must be callable."):
            self.simple_backend.set_options(experiment_result_function=1)
    
    def test_invalid_control_channel_map(self):
        """Test setting an invalid control_channel_map raises an error."""

        with self.assertRaisesRegex(QiskitError, "None or a dictionary"):
            self.simple_backend.set_options(control_channel_map=1)
        
        with self.assertRaisesRegex(QiskitError, "values must be of type int"):
            self.simple_backend.set_options(control_channel_map={(0, 1): "3"})


class TestDynamicsBackend(QiskitDynamicsTestCase):
    """Tests ensuring basic workflows work correctly for DynamicsBackend."""

    def setUp(self):
        """Build reusable models."""

        static_ham = 2 * np.pi * 5 * np.array([[-1.0, 0.0], [0.0, 1.0]]) / 2
        drive_op = 2 * np.pi * 0.1 * np.array([[0.0, 1.0], [1.0, 0.0]]) / 2

        solver = Solver(
            static_hamiltonian=static_ham,
            hamiltonian_operators=[drive_op],
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 5.0},
            dt=0.1,
            rotating_frame=static_ham,
        )

        self.simple_solver = solver
        self.simple_backend = DynamicsBackend(solver=solver)

        ident = np.eye(2, dtype=complex)
        static_ham_2q = (
            2 * np.pi * 4.99 * np.kron(ident, np.array([[-1.0, 0.0], [0.0, 1.0]])) / 2
            + 2 * np.pi * 5.01 * np.kron(np.array([[-1.0, 0.0], [0.0, 1.0]]), ident) / 2
            + 2
            * np.pi
            * 0.002
            * np.kron(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]]))
            + 2
            * np.pi
            * 0.002
            * np.kron(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 0.0]]))
        )
        drive_op0 = 2 * np.pi * 0.1 * np.kron(ident, np.array([[0.0, 1.0], [1.0, 0.0]])) / 2
        drive_op1 = 2 * np.pi * 0.1 * np.kron(np.array([[0.0, 1.0], [1.0, 0.0]]), ident) / 2
        solver_2q = Solver(
            static_hamiltonian=static_ham_2q,
            hamiltonian_operators=[drive_op0, drive_op1],
            hamiltonian_channels=["d0", "d1"],
            channel_carrier_freqs={"d0": 4.99, "d1": 5.01},
            dt=0.1,
            rotating_frame=static_ham_2q,
        )
        self.backend_2q = DynamicsBackend(solver=solver_2q, subsystem_dims=[2, 2])

        # function to discriminate 0 and 1 for default centers.
        self.iq_to_counts = lambda iq_n: dict(
            zip(*np.unique(["0" if iq[0].real > 0 else "1" for iq in iq_n], return_counts=True))
        )

    def test_pi_pulse(self):
        """Test simulation of a pi pulse."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        result = self.simple_backend.run(schedule, seed_simulator=1234567).result()
        self.assertDictEqual(result.get_counts(), {"1": 1024})
        self.assertTrue(result.get_memory() == ["1"] * 1024)

        result = self.simple_backend.run(
            schedule, meas_level=1, meas_return="single", seed_simulator=1234567
        ).result()
        counts = self.iq_to_counts(result.get_memory())
        self.assertDictEqual(counts, {"1": 1024})

    def test_pi_pulse_initial_state(self):
        """Test simulation of a pi pulse with a different initial state."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        result = self.simple_backend.run(
            schedule, seed_simulator=1234567, initial_state=Statevector([0.0, 1.0])
        ).result()
        self.assertDictEqual(result.get_counts(), {"0": 1024})
        self.assertTrue(result.get_memory() == ["0"] * 1024)

    def test_pi_half_pulse_density_matrix(self):
        """Test simulation of a pi/2 pulse with a DensityMatrix."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        result = self.simple_backend.run(
            schedule, seed_simulator=398472, initial_state=DensityMatrix([1.0, 0.0])
        ).result()
        self.assertDictEqual(result.get_counts(), {"0": 505, "1": 519})

        result = result = self.simple_backend.run(
            schedule,
            seed_simulator=398472,
            initial_state=DensityMatrix([1.0, 0.0]),
            meas_level=1,
            meas_return="single",
        ).result()

        counts = self.iq_to_counts(result.get_memory())
        self.assertDictEqual(counts, {"0": 499, "1": 525})

    def test_pi_half_pulse_relabelled(self):
        """Test simulation of a pi/2 pulse with qubit relabelled."""

        self.simple_backend.set_options(subsystem_labels=[1])

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=1, register=pulse.MemorySlot(1))

        result = self.simple_backend.run(schedule, seed_simulator=398472).result()
        self.assertDictEqual(result.get_counts(), {"00": 505, "10": 519})

    def test_circuit_with_pulse_defs(self):
        """Test simulating a circuit with pulse definitions."""

        circ = QuantumCircuit(1, 1)
        circ.x(0)
        circ.measure([0], [0])

        with pulse.build() as x_sched0:
            pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))

        circ.add_calibration("x", [0], x_sched0)

        result = self.simple_backend.run(circ, seed_simulator=1234567).result()
        self.assertDictEqual(result.get_counts(), {"1": 1024})
        self.assertTrue(result.get_memory() == ["1"] * 1024)

    def test_circuit_with_target_pulse_instructions(self):
        """Test running a circuit on a simulator with defined instructions."""

        # build target into simulator
        with pulse.build() as x_sched0:
            pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))

        target = Target()
        inst_sched_map = target.instruction_schedule_map()
        inst_sched_map.add("x", qubits=0, schedule=x_sched0)

        backend = DynamicsBackend(solver=self.simple_solver, target=target)

        # build and run circuit
        circ = QuantumCircuit(1, 1)
        circ.x(0)
        circ.measure([0], [0])

        result = backend.run(circ, seed_simulator=1234567).result()
        self.assertDictEqual(result.get_counts(), {"1": 1024})
        self.assertTrue(result.get_memory() == ["1"] * 1024)

    def test_circuit_memory_slot_num(self):
        """Test correct memory_slot number based on quantum circuit."""

        # build a pair of non-trivial 2q circuits with 5 memoryslots, saving measurements
        # in different memory slots
        circ0 = QuantumCircuit(2, 5)
        circ0.x(0)
        circ0.h(1)
        circ0.measure([0, 1], [0, 1])

        circ1 = QuantumCircuit(2, 5)
        circ1.x(0)
        circ1.h(1)
        circ1.measure([0, 1], [2, 4])

        # add definitions to instruction_schedule_map
        inst_map = self.backend_2q.instruction_schedule_map
        with pulse.build() as x_sched0:
            pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))

        with pulse.build() as h_sched1:
            pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(1))

        inst_map.add("x", qubits=0, schedule=x_sched0)
        inst_map.add("h", qubits=1, schedule=h_sched1)

        # run both
        result0 = self.backend_2q.run(circ0, seed_simulator=1234567).result()
        result1 = self.backend_2q.run(circ1, seed_simulator=1234567).result()

        # extract results form memory slots and validate all others are 0
        result0_dict = {}
        for string, count in result0.get_counts().items():
            self.assertTrue(string[:3] == "000")
            result0_dict[string[3:]] = count

        result1_dict = {}
        for string, count in result1.get_counts().items():
            self.assertTrue(string[-4] + string[-2] + string[-1] == "000")
            result1_dict[string[-5] + string[-3]] = count

        # validate consistent results
        self.assertDictEqual(result0_dict, result1_dict)

    def test_schedule_memory_slot_num(self):
        """Test correct memory_slot number in schedule."""

        with pulse.build() as schedule0:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))
                pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(1))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))
                pulse.acquire(duration=1, qubit_or_channel=1, register=pulse.MemorySlot(1))

        with pulse.build() as schedule1:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))
                pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(1))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(2))
                pulse.acquire(duration=1, qubit_or_channel=1, register=pulse.MemorySlot(4))

        # run both
        result0 = self.backend_2q.run(schedule0, seed_simulator=1234567).result()
        result1 = self.backend_2q.run(schedule1, seed_simulator=1234567).result()

        # extract results form memory slots and validate all others are 0
        result0_dict = result0.get_counts()
        for string in result0_dict:
            self.assertTrue(len(string) == 2)

        result1_dict = {}
        for string, count in result1.get_counts().items():
            self.assertTrue(string[-4] + string[-2] + string[-1] == "000")
            result1_dict[string[-5] + string[-3]] = count

        # validate consistent results
        self.assertDictEqual(result0_dict, result1_dict)

        result0_iq = (
            self.backend_2q.run(schedule0, meas_level=1, seed_simulator=1234567)
            .result()
            .get_memory()
        )
        result1_iq = (
            self.backend_2q.run(schedule1, meas_level=1, seed_simulator=1234567)
            .result()
            .get_memory()
        )

        self.assertTrue(result0_iq.shape == (2,))
        self.assertTrue(result1_iq.shape == (5,))
        self.assertAllClose(result0_iq, result1_iq[[2, 4]])

    def test_measure_higher_levels(self):
        """Test measurement of higher levels."""

        solver = Solver(static_hamiltonian=np.diag([-1.0, 0.0, 1.0]), dt=0.1)
        qutrit_backend = DynamicsBackend(
            solver=solver, max_outcome_level=None, initial_state=Statevector([0.0, 0.0, 1.0])
        )

        circ = QuantumCircuit(1, 1)
        circ.measure([0], [0])

        res = qutrit_backend.run(circ).result()

        self.assertDictEqual(res.get_counts(), {"2": 1024})

    def test_setting_experiment_result_function(self):
        """Test overriding default experiment_result_function."""

        # trivial result function
        # pylint: disable=unused-argument
        def exp_result_function(*args, **kwargs):
            return ExperimentResult(
                data=ExperimentResultData(counts={"3": 1}), shots=1, success=True
            )

        # minimal simulation schedule
        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 1), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        result = self.simple_backend.run(
            schedule,
            seed_simulator=1234567,
            initial_state=Statevector([0.0, 1.0]),
            experiment_result_function=exp_result_function,
        ).result()
        self.assertDictEqual(result.get_counts(), {"3": 1})


class Test_default_experiment_result_function(QiskitDynamicsTestCase):
    """Test default_experiment_result_function."""

    def setUp(self):
        """Build reusable models."""

        static_ham = 2 * np.pi * 5 * np.array([[-1.0, 0.0], [0.0, 1.0]]) / 2
        drive_op = 2 * np.pi * 0.1 * np.array([[0.0, 1.0], [1.0, 0.0]]) / 2

        solver = Solver(
            static_hamiltonian=static_ham,
            hamiltonian_operators=[drive_op],
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 5.0},
            dt=0.1,
            rotating_frame=static_ham,
        )

        self.simple_solver = solver
        self.simple_backend = DynamicsBackend(solver=solver)

    def test_simple_example(self):
        """Test a simple example."""

        output = default_experiment_result_function(
            experiment_name="exp123",
            solver_result=OdeResult(
                t=[0.0, 1.0], y=[Statevector([1.0, 0.0]), Statevector(np.sqrt([0.5, 0.5]))]
            ),
            measurement_subsystems=[0],
            memory_slot_indices=[1],
            num_memory_slots=3,
            backend=self.simple_backend,
            seed=1234567,
        )

        self.assertDictEqual(output.data.counts, {"000": 513, "010": 511})
