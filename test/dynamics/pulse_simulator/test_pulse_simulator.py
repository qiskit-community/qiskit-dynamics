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
"""
Test PulseSimulator.
"""

import numpy as np

from qiskit import QiskitError, pulse, QuantumCircuit
from qiskit.transpiler import Target
from qiskit.quantum_info import Statevector, DensityMatrix

from qiskit_dynamics import Solver, PulseSimulator
from ..common import QiskitDynamicsTestCase


class TestPulseSimulatorValidation(QiskitDynamicsTestCase):
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

        self.simple_simulator = PulseSimulator(solver=solver)

    def test_solver_not_configured_for_pulse(self):
        """Test error is raised if solver not configured for pulse simulation."""

        solver = Solver(
            static_hamiltonian=np.array([[1.0, 0.0], [0.0, -1.0]]),
            hamiltonian_operators=[np.array([[0.0, 1.0], [1.0, 0.0]])],
        )

        with self.assertRaisesRegex(QiskitError, "not configured for Pulse"):
            PulseSimulator(solver=solver)

    def test_run_input_error(self):
        """Test submission of invalid run input."""

        with self.assertRaisesRegex(QiskitError, "not supported by PulseSimulator.run."):
            self.simple_simulator.run([1])

    def test_subsystem_dims_inconsistency(self):
        """Test that setting subsystem_dims inconsistently with solver.model.dim raises error."""

        with self.assertRaisesRegex(QiskitError, "inconsistent"):
            self.simple_simulator.set_options(subsystem_dims=[4])

    def test_max_outcome_level_error(self):
        """Test that invalid max_outcome_level results in error."""

        with self.assertRaisesRegex(QiskitError, "must be a positive integer"):
            self.simple_simulator.set_options(max_outcome_level=0)

        with self.assertRaisesRegex(QiskitError, "must be a positive integer"):
            self.simple_simulator.set_options(max_outcome_level="hi")

    def test_no_measurements_in_schedule(self):
        """Test that running a schedule with no measurements raises an error."""

        with pulse.build() as schedule:
            pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))

        with self.assertRaisesRegex(QiskitError, "At least one measurement"):
            self.simple_simulator.run(schedule)

    def test_no_measurements_with_memory_slots_in_schedule(self):
        """Test that running a schedule without measurements saving results in a MemorySlot
        raises an error."""

        with pulse.build() as schedule:
            pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
            pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.RegisterSlot(0))

        with self.assertRaisesRegex(QiskitError, "At least one measurement"):
            self.simple_simulator.run(schedule)

    def test_multiple_measurements_in_schedule(self):
        """Test error raising when attempting to run a schedule with multiple measurements."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))
                pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        with self.assertRaisesRegex(QiskitError, "only supports measurements at one time"):
            self.simple_simulator.run(schedule)

    def test_measure_nonexistant_subsystem(self):
        """Attempt to measure subsystem that doesn't exist."""

        with pulse.build() as schedule:
            pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
            pulse.acquire(duration=1, qubit_or_channel=1, register=pulse.MemorySlot(0))

        with self.assertRaisesRegex(QiskitError, "Attempted to measure subsystem 1"):
            self.simple_simulator.run(schedule)

    def test_invalid_initial_state(self):
        """Test setting an invalid initial state."""

        with self.assertRaisesRegex(QiskitError, "initial_state must be either"):
            self.simple_simulator.set_options(initial_state=1)

    def test_invalid_meas_level(self):
        """Test setting an invalid meas_level."""

        with self.assertRaisesRegex(QiskitError, "Only meas_level == 2 is supported"):
            self.simple_simulator.set_options(meas_level=1)


class TestPulseSimulator(QiskitDynamicsTestCase):
    """Tests ensuring basic workflows work correctly for PulseSimulator."""

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
        self.simple_simulator = PulseSimulator(solver=solver)

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
        self.simulator_2q = PulseSimulator(solver=solver_2q, subsystem_dims=[2, 2])

    def test_pi_pulse(self):
        """Test simulation of a pi pulse."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        result = self.simple_simulator.run(schedule, seed_simulator=1234567).result()
        self.assertDictEqual(result.get_counts(), {"1": 1024})
        self.assertTrue(result.get_memory() == ["1"] * 1024)

    def test_pi_pulse_initial_state(self):
        """Test simulation of a pi pulse with a different initial state."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        result = self.simple_simulator.run(
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

        result = self.simple_simulator.run(
            schedule, seed_simulator=398472, initial_state=DensityMatrix([1.0, 0.0])
        ).result()
        self.assertDictEqual(result.get_counts(), {"0": 505, "1": 519})

    def test_pi_half_pulse_relabelled(self):
        """Test simulation of a pi/2 pulse with qubit relabelled."""

        self.simple_simulator.set_options(subsystem_labels=[1])

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=1, register=pulse.MemorySlot(1))

        result = self.simple_simulator.run(schedule, seed_simulator=398472).result()
        self.assertDictEqual(result.get_counts(), {"00": 505, "10": 519})

    def test_circuit_with_pulse_defs(self):
        """Test simulating a circuit with pulse definitions."""

        circ = QuantumCircuit(1, 1)
        circ.x(0)
        circ.measure([0], [0])

        with pulse.build() as x_sched0:
            pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))

        circ.add_calibration("x", [0], x_sched0)

        result = self.simple_simulator.run(circ, seed_simulator=1234567).result()
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

        pulse_simulator = PulseSimulator(solver=self.simple_solver, target=target)

        # build and run circuit
        circ = QuantumCircuit(1, 1)
        circ.x(0)
        circ.measure([0], [0])

        result = pulse_simulator.run(circ, seed_simulator=1234567).result()
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
        inst_map = self.simulator_2q.instruction_schedule_map
        with pulse.build() as x_sched0:
            pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))

        with pulse.build() as h_sched1:
            pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(1))

        inst_map.add("x", qubits=0, schedule=x_sched0)
        inst_map.add("h", qubits=1, schedule=h_sched1)

        # run both
        result0 = self.simulator_2q.run(circ0, seed_simulator=1234567).result()
        result1 = self.simulator_2q.run(circ1, seed_simulator=1234567).result()

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
        result0 = self.simulator_2q.run(schedule0, seed_simulator=1234567).result()
        result1 = self.simulator_2q.run(schedule1, seed_simulator=1234567).result()

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

    def test_measure_higher_levels(self):
        """Test measurement of higher levels."""

        solver = Solver(static_hamiltonian=np.diag([-1.0, 0.0, 1.0]), dt=0.1)
        qutrit_simulator = PulseSimulator(
            solver=solver, max_outcome_level=None, initial_state=Statevector([0.0, 0.0, 1.0])
        )

        circ = QuantumCircuit(1, 1)
        circ.measure([0], [0])

        res = qutrit_simulator.run(circ).result()

        self.assertDictEqual(res.get_counts(), {"2": 1024})
